"""
CNN Model for Image Classification
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, ResNet50
from typing import Tuple, Optional


class ImageClassifierCNN:
    """Convolutional Neural Network for image classification"""
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        num_classes: int = 10,
        architecture: str = 'custom'
    ):
        """
        Initialize CNN model
        
        Args:
            input_shape: Input image shape (height, width, channels)
            num_classes: Number of output classes
            architecture: Model architecture ('custom', 'vgg16', 'resnet50')
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.architecture = architecture
        self.model = None
        
    def build_custom_cnn(self) -> models.Model:
        """
        Build custom CNN architecture
        
        Returns:
            Keras Model
        """
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def build_transfer_learning_model(self, base_model_name: str) -> models.Model:
        """
        Build model using transfer learning
        
        Args:
            base_model_name: Name of base model ('vgg16' or 'resnet50')
            
        Returns:
            Keras Model
        """
        # Load pre-trained base model
        if base_model_name == 'vgg16':
            base_model = VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif base_model_name == 'resnet50':
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        else:
            raise ValueError(f"Unknown base model: {base_model_name}")
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Build complete model
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def build(self) -> models.Model:
        """
        Build the model based on specified architecture
        
        Returns:
            Compiled Keras Model
        """
        if self.architecture == 'custom':
            self.model = self.build_custom_cnn()
        elif self.architecture in ['vgg16', 'resnet50']:
            self.model = self.build_transfer_learning_model(self.architecture)
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")
        
        return self.model
    
    def compile_model(
        self,
        optimizer: str = 'adam',
        learning_rate: float = 0.001,
        loss: str = 'categorical_crossentropy',
        metrics: list = None
    ):
        """
        Compile the model
        
        Args:
            optimizer: Optimizer name
            learning_rate: Learning rate
            loss: Loss function
            metrics: List of metrics
        """
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall']
        
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer == 'rmsprop':
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            opt = optimizer
        
        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=metrics
        )
    
    def get_model_summary(self):
        """Print model summary"""
        if self.model:
            self.model.summary()
        else:
            print("Model not built yet. Call build() first.")
    
    def save_model(self, filepath: str):
        """Save model to file"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from file"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")


def create_data_augmentation() -> keras.Sequential:
    """
    Create data augmentation pipeline
    
    Returns:
        Sequential model with augmentation layers
    """
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
        layers.RandomBrightness(0.1),
    ])


if __name__ == "__main__":
    # Example usage
    classifier = ImageClassifierCNN(
        input_shape=(224, 224, 3),
        num_classes=10,
        architecture='custom'
    )
    
    model = classifier.build()
    classifier.compile_model(learning_rate=0.001)
    classifier.get_model_summary()