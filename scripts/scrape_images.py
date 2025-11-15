"""
Script to scrape images from news websites
"""
import sys
import yaml
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.scraper.image_scraper import ImageScraper


def main():
    """Main function for scraping"""
    parser = argparse.ArgumentParser(
        description='Scrape images from news websites'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--pages',
        type=int,
        default=None,
        help='Number of pages to scrape per URL'
    )
    parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help='Maximum images per URL'
    )
    parser.add_argument(
        '--selenium',
        action='store_true',
        help='Use Selenium for dynamic content'
    )
    parser.add_argument(
        '--urls',
        nargs='+',
        default=None,
        help='Custom URLs to scrape (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get URLs
    urls = args.urls if args.urls else config['scraping']['urls']
    
    # Get max images
    max_images = args.max_images if args.max_images else config['scraping']['max_images_per_url']
    
    # Use Selenium?
    use_selenium = args.selenium or config['scraping'].get('use_selenium', False)
    
    # Initialize scraper
    print("="*60)
    print("NEWS IMAGE SCRAPER")
    print("="*60)
    print(f"Target URLs: {len(urls)}")
    print(f"Max images per URL: {max_images}")
    print(f"Using Selenium: {use_selenium}")
    print(f"Save directory: {config['data']['raw_dir']}")
    print("="*60)
    print()
    
    scraper = ImageScraper(
        save_dir=config['data']['raw_dir'],
        use_selenium=use_selenium
    )
    
    try:
        # Scrape images
        scraper.scrape_multiple_urls(urls, max_images_per_url=max_images)
        
        print("\n" + "="*60)
        print("SCRAPING COMPLETE!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\nScraping interrupted by user")
    
    except Exception as e:
        print(f"\n\nError during scraping: {str(e)}")
    
    finally:
        # Cleanup
        scraper.cleanup()
        print("\nCleaned up resources")


if __name__ == "__main__":
    main()