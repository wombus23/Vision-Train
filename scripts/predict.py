"""
Prediction script for trained CNN model
"""
import sys
import yaml
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing.image_processor import ImagePreprocessor


class ImageClassifierPredictor:
    """Handles predictions using trained model"""
    
    def __init__(self, model_path: str, config_path: str = "config/config.yaml"):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load model
        print(f"Loading model from {model_path}")
        self.model = keras.models.load_model(model_path)
        
        # Initialize preprocessor
        img_size = tuple(self.config['model']['input_shape'][:2])
        self.preprocessor = ImagePreprocessor(target_size=img_size)
        
        # Get class names
        self.class_names = self.config['classes']
    
    def predict_single_image(self, image_path: str, top_k: int = 3):
        """
        Predict class for a single image
        
        Args:
            image_path: Path to image
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with predictions
        """
        # Preprocess image
        img = self.preprocessor.preprocess_image(image_path)
        img_batch = np.expand_dims(img, axis=0)
        
        # Make prediction
        predictions = self.model.predict(img_batch, verbose=0)
        probabilities = predictions[0]
        
        # Get top-k predictions
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        
        results = {
            'predictions': [
                {
                    'class': self.class_names[idx],
                    'probability': float(probabilities[idx]),
                    'percentage': float(probabilities[idx] * 100)
                }
                for idx in top_indices
            ],
            'top_class': self.class_names[top_indices[0]],
            'confidence': float(probabilities[top_indices[0]])
        }
        
        return results
    
    def predict_batch(self, image_paths: list, batch_size: int = 32):
        """
        Predict classes for multiple images
        
        Args:
            image_paths: List of image paths
            batch_size: Batch size for prediction
            
        Returns:
            List of prediction results
        """
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            
            # Preprocess batch
            batch_images = []
            for path in batch_paths:
                img = self.preprocessor.preprocess_image(path)
                batch_images.append(img)
            
            batch_images = np.array(batch_images)
            
            # Predict
            predictions = self.model.predict(batch_images, verbose=0)
            
            # Process results
            for j, pred in enumerate(predictions):
                top_idx = np.argmax(pred)
                results.append({
                    'image': batch_paths[j],
                    'class': self.class_names[top_idx],
                    'confidence': float(pred[top_idx])
                })
        
        return results
    
    def visualize_prediction(self, image_path: str, save_path: str = None):
        """
        Visualize prediction with image and probabilities
        
        Args:
            image_path: Path to image
            save_path: Path to save visualization
        """
        # Get prediction
        result = self.predict_single_image(image_path, top_k=5)
        
        # Load and display image
        img = self.preprocessor.load_image(image_path)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Display image
        ax1.imshow(img)
        ax1.axis('off')
        ax1.set_title(f"Predicted: {result['top_class']}\n"
                     f"Confidence: {result['confidence']:.2%}")
        
        # Display probabilities
        classes = [p['class'] for p in result['predictions']]
        probs = [p['probability'] for p in result['predictions']]
        
        colors = ['green' if i == 0 else 'skyblue' 
                 for i in range(len(classes))]
        
        ax2.barh(classes, probs, color=colors)
        ax2.set_xlabel('Probability')
        ax2.set_title('Top 5 Predictions')
        ax2.set_xlim([0, 1])
        
        # Add percentage labels
        for i, (c, p) in enumerate(zip(classes, probs)):
            ax2.text(p + 0.01, i, f'{p:.2%}', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description='Predict image class using trained CNN model'
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to image file'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='data/models/best_model.h5',
        help='Path to trained model'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Visualize prediction'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save visualization'
    )
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = ImageClassifierPredictor(args.model, args.config)
    
    # Make prediction
    print(f"\nPredicting class for: {args.image}")
    result = predictor.predict_single_image(args.image)
    
    # Print results
    print(f"\nTop Class: {result['top_class']}")
    print(f"Confidence: {result['confidence']:.2%}\n")
    
    print("Top 3 Predictions:")
    for i, pred in enumerate(result['predictions'], 1):
        print(f"{i}. {pred['class']}: {pred['percentage']:.2f}%")
    
    # Visualize if requested
    if args.visualize:
        predictor.visualize_prediction(args.image, args.output)


if __name__ == "__main__":
    main()