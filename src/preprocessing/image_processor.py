"""
Image preprocessing and data loading utilities
"""
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
from PIL import Image
import tensorflow as tf
from tqdm import tqdm


class ImagePreprocessor:
    """Handles image preprocessing operations"""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        normalize: bool = True
    ):
        """
        Initialize preprocessor
        
        Args:
            target_size: Target image size (height, width)
            normalize: Whether to normalize pixel values to [0, 1]
        """
        self.target_size = target_size
        self.normalize = normalize
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load image from file
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image as numpy array
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target size
        
        Args:
            image: Input image
            
        Returns:
            Resized image
        """
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image pixel values to [0, 1]
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        return image.astype(np.float32) / 255.0
    
    def augment_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations to image
        
        Args:
            image: Input image
            
        Returns:
            Augmented image
        """
        # Random horizontal flip
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 1)
        
        # Random rotation
        angle = np.random.uniform(-15, 15)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
        
        # Random brightness adjustment
        brightness_factor = np.random.uniform(0.8, 1.2)
        image = np.clip(image * brightness_factor, 0, 255).astype(np.uint8)
        
        return image
    
    def preprocess_image(
        self,
        image_path: str,
        augment: bool = False
    ) -> np.ndarray:
        """
        Complete preprocessing pipeline
        
        Args:
            image_path: Path to image
            augment: Whether to apply augmentation
            
        Returns:
            Preprocessed image
        """
        # Load image
        img = self.load_image(image_path)
        
        # Apply augmentation if requested
        if augment:
            img = self.augment_image(img)
        
        # Resize
        img = self.resize_image(img)
        
        # Normalize
        if self.normalize:
            img = self.normalize_image(img)
        
        return img
    
    def validate_image(self, image_path: str) -> bool:
        """
        Check if image is valid and can be processed
        
        Args:
            image_path: Path to image
            
        Returns:
            True if valid, False otherwise
        """
        try:
            img = Image.open(image_path)
            img.verify()
            
            # Check minimum size
            if img.size[0] < 50 or img.size[1] < 50:
                return False
            
            return True
        except:
            return False


class ImageDataLoader:
    """Handles batch loading and organization of image data"""
    
    def __init__(
        self,
        data_dir: str,
        preprocessor: ImagePreprocessor,
        batch_size: int = 32
    ):
        """
        Initialize data loader
        
        Args:
            data_dir: Directory containing images
            preprocessor: ImagePreprocessor instance
            batch_size: Batch size for loading
        """
        self.data_dir = Path(data_dir)
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.image_paths = []
        self.labels = []
        self.class_names = []
    
    def scan_directory(self):
        """Scan directory and collect image paths"""
        print(f"Scanning directory: {self.data_dir}")
        
        # Get class directories
        class_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        self.class_names = [d.name for d in class_dirs]
        
        print(f"Found {len(self.class_names)} classes: {self.class_names}")
        
        # Collect image paths and labels
        for class_idx, class_dir in enumerate(class_dirs):
            image_files = list(class_dir.glob('*.jpg')) + \
                         list(class_dir.glob('*.png')) + \
                         list(class_dir.glob('*.jpeg'))
            
            for img_path in image_files:
                if self.preprocessor.validate_image(str(img_path)):
                    self.image_paths.append(str(img_path))
                    self.labels.append(class_idx)
        
        print(f"Total valid images: {len(self.image_paths)}")
    
    def load_batch(
        self,
        start_idx: int,
        end_idx: int,
        augment: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a batch of images
        
        Args:
            start_idx: Start index
            end_idx: End index
            augment: Whether to apply augmentation
            
        Returns:
            Batch of images and labels
        """
        batch_paths = self.image_paths[start_idx:end_idx]
        batch_labels = self.labels[start_idx:end_idx]
        
        images = []
        for path in batch_paths:
            img = self.preprocessor.preprocess_image(path, augment=augment)
            images.append(img)
        
        return np.array(images), np.array(batch_labels)
    
    def create_dataset(self, shuffle: bool = True, augment: bool = False):
        """
        Create TensorFlow dataset
        
        Args:
            shuffle: Whether to shuffle data
            augment: Whether to apply augmentation
            
        Returns:
            tf.data.Dataset
        """
        if not self.image_paths:
            self.scan_directory()
        
        def generator():
            indices = list(range(len(self.image_paths)))
            if shuffle:
                np.random.shuffle(indices)
            
            for idx in indices:
                img = self.preprocessor.preprocess_image(
                    self.image_paths[idx],
                    augment=augment
                )
                label = self.labels[idx]
                yield img, label
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(
                    shape=(*self.preprocessor.target_size, 3),
                    dtype=tf.float32
                ),
                tf.TensorSpec(shape=(), dtype=tf.int32)
            )
        )
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset


def organize_images_by_category(
    source_dir: str,
    target_dir: str,
    categories: List[str]
):
    """
    Organize scraped images into category folders
    
    Args:
        source_dir: Source directory with raw images
        target_dir: Target directory for organized images
        categories: List of category names
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Create category directories
    for category in categories:
        (target_path / category).mkdir(parents=True, exist_ok=True)
    
    print(f"Organizing images from {source_dir} to {target_dir}")
    print("This is a manual process - implement your own logic based on filenames or metadata")


if __name__ == "__main__":
    # Example usage
    preprocessor = ImagePreprocessor(target_size=(224, 224))
    
    # Test single image
    img = preprocessor.preprocess_image("path/to/image.jpg")
    print(f"Preprocessed image shape: {img.shape}")