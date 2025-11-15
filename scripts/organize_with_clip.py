"""
Smart Image Organizer using CLIP (OpenAI)
More accurate for news image classification
"""
import os
import sys
import yaml
import shutil
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

import torch
import clip
from PIL import Image

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


class CLIPImageOrganizer:
    """Organize images using CLIP zero-shot classification"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize organizer"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.raw_dir = Path(self.config['data']['raw_dir'])
        self.processed_dir = Path(self.config['data']['processed_dir'])
        self.categories = self.config['classes']
        
        # Load CLIP model
        print("Loading CLIP model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        print(f"✅ CLIP loaded on {self.device}!")
        
        # Create text prompts for each category
        self.text_prompts = self._create_prompts()
    
    def _create_prompts(self) -> torch.Tensor:
        """
        Create text prompts for each category
        
        Returns:
            Tokenized text prompts
        """
        # Create detailed prompts for better classification
        prompts = []
        
        for category in self.categories:
            # Create multiple prompts per category for better accuracy
            category_prompts = [
                f"a photo of {category}",
                f"a news photo about {category}",
                f"an image related to {category}",
            ]
            prompts.extend(category_prompts)
        
        # Tokenize prompts
        text_tokens = clip.tokenize(prompts).to(self.device)
        
        return text_tokens
    
    def predict_category(self, img_path: str) -> tuple:
        """
        Predict category for an image
        
        Args:
            img_path: Path to image
            
        Returns:
            Tuple of (category, confidence)
        """
        try:
            # Load and preprocess image
            image = Image.open(img_path).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Encode image and text
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                text_features = self.model.encode_text(self.text_prompts)
                
                # Calculate similarity
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            # Get predictions for each category (average over prompts)
            num_categories = len(self.categories)
            prompts_per_category = 3  # We created 3 prompts per category
            
            category_scores = []
            for i in range(num_categories):
                start_idx = i * prompts_per_category
                end_idx = start_idx + prompts_per_category
                category_score = similarity[0, start_idx:end_idx].mean().item()
                category_scores.append(category_score)
            
            # Get best category
            best_idx = torch.tensor(category_scores).argmax().item()
            best_category = self.categories[best_idx]
            confidence = category_scores[best_idx]
            
            return best_category, confidence
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return 'unknown', 0.0
    
    def organize_images(
        self,
        min_confidence: float = 0.25,
        review_mode: bool = False,
        batch_size: int = 1,
        force_categorize: bool = True
    ):
        """
        Organize all images
        
        Args:
            min_confidence: Minimum confidence threshold
            review_mode: Ask for confirmation
            batch_size: Process multiple images at once
            force_categorize: Always assign to best category (no unknowns)
        """
        # Create directories (skip unknown if force_categorize)
        dirs_to_create = self.categories if force_categorize else self.categories + ['unknown']
        for category in dirs_to_create:
            (self.processed_dir / category).mkdir(parents=True, exist_ok=True)
        
        # Get images
        image_files = list(self.raw_dir.glob('*.jpg')) + \
                     list(self.raw_dir.glob('*.png')) + \
                     list(self.raw_dir.glob('*.jpeg'))
        
        print(f"\n📁 Found {len(image_files)} images")
        print(f"📂 Target: {self.processed_dir}")
        print(f"🎯 Categories: {', '.join(self.categories)}")
        print(f"🔍 Using CLIP model on {self.device}")
        print(f"🎲 Force categorize: {'YES - No unknowns!' if force_categorize else 'NO - Using threshold'}\n")
        
        # Statistics
        stats = {cat: 0 for cat in self.categories + ['unknown', 'skipped']}
        
        # Process images
        for img_path in tqdm(image_files, desc="Organizing"):
            try:
                # Predict
                category, confidence = self.predict_category(str(img_path))
                
                # Check confidence (only if not forcing categorization)
                if not force_categorize and confidence < min_confidence:
                    category = 'unknown'
                
                # Review mode
                if review_mode:
                    print(f"\n{img_path.name}")
                    print(f"Category: {category} ({confidence:.1%})")
                    resp = input("Accept (y), Skip (s), or enter category: ").strip().lower()
                    
                    if resp == 's':
                        stats['skipped'] += 1
                        continue
                    elif resp and resp != 'y':
                        category = resp
                
                # Copy image
                dest = self.processed_dir / category / img_path.name
                
                # Handle duplicates
                counter = 1
                while dest.exists():
                    stem = img_path.stem
                    suffix = img_path.suffix
                    dest = self.processed_dir / category / f"{stem}_{counter}{suffix}"
                    counter += 1
                
                shutil.copy2(img_path, dest)
                stats[category] += 1
                
            except Exception as e:
                print(f"Error: {img_path.name}: {e}")
                stats['skipped'] += 1
        
        # Print results
        self._print_stats(stats)
    
    def _print_stats(self, stats: Dict[str, int]):
        """Print statistics"""
        print("\n" + "="*60)
        print("✅ ORGANIZATION COMPLETE!")
        print("="*60)
        
        total = sum(stats.values())
        print(f"\nTotal: {total} images\n")
        
        for cat, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total * 100) if total > 0 else 0
            bar = '█' * int(pct / 2)
            print(f"{cat:15s}: {count:4d} ({pct:5.1f}%) {bar}")
        
        if stats['unknown'] > 0:
            print(f"\n⚠️  {stats['unknown']} unknown images - review manually")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Organize images with CLIP')
    parser.add_argument('--config', default='config/config.yaml')
    parser.add_argument('--confidence', type=float, default=0.25,
                       help='Min confidence (0.0-1.0)')
    parser.add_argument('--review', action='store_true',
                       help='Review each classification')
    parser.add_argument('--force', action='store_true', default=True,
                       help='Force categorization (no unknowns)')
    
    args = parser.parse_args()
    
    # Check if CLIP is installed
    try:
        import clip
    except ImportError:
        print("❌ CLIP not installed!")
        print("\nInstall it with:")
        print("  pip install git+https://github.com/openai/CLIP.git")
        return
    
    # Run organizer
    organizer = CLIPImageOrganizer(args.config)
    organizer.organize_images(
        min_confidence=args.confidence,
        review_mode=args.review,
        force_categorize=args.force
    )
    
    print(f"\n✅ Done! Check: {organizer.processed_dir}")


if __name__ == "__main__":
    main()