"""
Training script for CNN model
"""
import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.cnn_model import ImageClassifierCNN, create_data_augmentation
from src.preprocessing.image_processor import ImageDataLoader


class ModelTrainer:
    """Handles model training and evaluation"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize trainer
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
        self.history = None
        
    def prepare_data(self) -> Tuple:
        """
        Prepare training and validation data
        
        Returns:
            Tuple of (train_ds, val_ds, test_ds)
        """
        data_dir = self.config['data']['processed_dir']
        img_size = tuple(self.config['model']['input_shape'][:2])
        batch_size = self.config['training']['batch_size']
        num_classes = self.config['model']['num_classes']
        
        # Create datasets
        train_ds = keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=img_size,
            batch_size=batch_size,
            label_mode='categorical'  # One-hot encode labels
        )
        
        val_ds = keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=img_size,
            batch_size=batch_size,
            label_mode='categorical'  # One-hot encode labels
        )
        
        # Normalize pixel values to [0, 1]
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
        
        # Optimize performance
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        
        return train_ds, val_ds
    
    def build_model(self):
        """Build and compile model"""
        model_config = self.config['model']
        training_config = self.config['training']
        
        # Create model
        classifier = ImageClassifierCNN(
            input_shape=tuple(model_config['input_shape']),
            num_classes=model_config['num_classes'],
            architecture=model_config.get('architecture', 'custom')
        )
        
        self.model = classifier.build()
        
        # Compile model
        classifier.compile_model(
            optimizer=training_config['optimizer'],
            learning_rate=training_config['learning_rate'],
            loss=model_config['loss'],
            metrics=model_config.get('metrics', ['accuracy'])
        )
        
        return self.model
    
    def create_callbacks(self) -> list:
        """
        Create training callbacks
        
        Returns:
            List of callbacks
        """
        callbacks = []
        
        # Model checkpoint
        checkpoint_path = Path(self.config['data']['models_dir']) / 'best_model.h5'
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        )
        
        # Early stopping
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config['training'].get('early_stopping_patience', 10),
                restore_best_weights=True,
                verbose=1
            )
        )
        
        # Reduce learning rate
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        )
        
        # TensorBoard
        log_dir = Path(self.config['data']['models_dir']) / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            keras.callbacks.TensorBoard(log_dir=str(log_dir))
        )
        
        return callbacks
    
    def train(self):
        """Train the model"""
        print("Preparing data...")
        train_ds, val_ds = self.prepare_data()
        
        print("Building model...")
        self.build_model()
        self.model.summary()
        
        print("Training model...")
        callbacks = self.create_callbacks()
        
        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.config['training']['epochs'],
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def plot_training_history(self, save_path: str = None):
        """
        Plot training history
        
        Args:
            save_path: Path to save plot
        """
        if not self.history:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        axes[0].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot loss
        axes[1].plot(self.history.history['loss'], label='Train Loss')
        axes[1].plot(self.history.history['val_loss'], label='Val Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def evaluate(self, test_ds):
        """
        Evaluate model on test set
        
        Args:
            test_ds: Test dataset
        """
        if not self.model:
            print("Model not trained yet")
            return
        
        results = self.model.evaluate(test_ds, verbose=1)
        
        print("\nTest Results:")
        for name, value in zip(self.model.metrics_names, results):
            print(f"{name}: {value:.4f}")
        
        return results
    
    def save_final_model(self):
        """Save final trained model"""
        model_path = Path(self.config['data']['models_dir']) / 'final_model.h5'
        self.model.save(str(model_path))
        print(f"Final model saved to {model_path}")


if __name__ == "__main__":
    # Train the model
    trainer = ModelTrainer("config/config.yaml")
    
    # Train
    history = trainer.train()
    
    # Plot results
    plot_path = Path(trainer.config['data']['models_dir']) / 'training_history.png'
    trainer.plot_training_history(save_path=str(plot_path))
    
    # Save final model
    trainer.save_final_model()
    
    print("\nTraining complete!")