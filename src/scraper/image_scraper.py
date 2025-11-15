"""
Image scraper for news websites
"""
import os
import time
import hashlib
import requests
from typing import List, Dict
from urllib.parse import urljoin, urlparse
from pathlib import Path

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

from tqdm import tqdm


class ImageScraper:
    """Scrapes images from news websites"""
    
    def __init__(self, save_dir: str = "data/raw", use_selenium: bool = False):
        """
        Initialize the scraper
        
        Args:
            save_dir: Directory to save images
            use_selenium: Whether to use Selenium for dynamic content
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.use_selenium = use_selenium
        self.driver = None
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def setup_selenium(self):
        """Setup Selenium WebDriver"""
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        
    def get_page_content(self, url: str) -> str:
        """
        Get page content using requests or Selenium
        
        Args:
            url: URL to scrape
            
        Returns:
            HTML content
        """
        if self.use_selenium:
            if not self.driver:
                self.setup_selenium()
            self.driver.get(url)
            time.sleep(2)  # Wait for dynamic content
            return self.driver.page_source
        else:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.text
    
    def extract_image_urls(self, html: str, base_url: str) -> List[str]:
        """
        Extract image URLs from HTML
        
        Args:
            html: HTML content
            base_url: Base URL for resolving relative paths
            
        Returns:
            List of image URLs
        """
        soup = BeautifulSoup(html, 'lxml')
        image_urls = []
        
        # Find all img tags
        for img in soup.find_all('img'):
            src = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
            if src:
                # Convert relative URLs to absolute
                full_url = urljoin(base_url, src)
                # Filter valid image URLs
                if self._is_valid_image_url(full_url):
                    image_urls.append(full_url)
        
        return list(set(image_urls))  # Remove duplicates
    
    def _is_valid_image_url(self, url: str) -> bool:
        """Check if URL points to an image"""
        valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.webp')
        parsed = urlparse(url)
        return any(parsed.path.lower().endswith(ext) for ext in valid_extensions)
    
    def download_image(self, url: str) -> bool:
        """
        Download a single image
        
        Args:
            url: Image URL
            
        Returns:
            Success status
        """
        try:
            response = self.session.get(url, timeout=10, stream=True)
            response.raise_for_status()
            
            # Generate filename from URL hash
            url_hash = hashlib.md5(url.encode()).hexdigest()
            extension = Path(urlparse(url).path).suffix or '.jpg'
            filename = f"{url_hash}{extension}"
            filepath = self.save_dir / filename
            
            # Save image
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return True
            
        except Exception as e:
            print(f"Error downloading {url}: {str(e)}")
            return False
    
    def scrape_website(self, url: str, max_images: int = 100) -> Dict[str, int]:
        """
        Scrape images from a website
        
        Args:
            url: Website URL
            max_images: Maximum number of images to download
            
        Returns:
            Statistics dictionary
        """
        print(f"Scraping: {url}")
        
        try:
            # Get page content
            html = self.get_page_content(url)
            
            # Extract image URLs
            image_urls = self.extract_image_urls(html, url)
            print(f"Found {len(image_urls)} images")
            
            # Download images
            downloaded = 0
            for img_url in tqdm(image_urls[:max_images], desc="Downloading"):
                if self.download_image(img_url):
                    downloaded += 1
                time.sleep(0.5)  # Rate limiting
            
            return {
                'found': len(image_urls),
                'downloaded': downloaded,
                'failed': len(image_urls[:max_images]) - downloaded
            }
            
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return {'found': 0, 'downloaded': 0, 'failed': 0}
    
    def scrape_multiple_urls(self, urls: List[str], max_images_per_url: int = 100):
        """
        Scrape images from multiple URLs
        
        Args:
            urls: List of URLs
            max_images_per_url: Max images per URL
        """
        total_stats = {'found': 0, 'downloaded': 0, 'failed': 0}
        
        for url in urls:
            stats = self.scrape_website(url, max_images_per_url)
            for key in total_stats:
                total_stats[key] += stats[key]
            print(f"Stats for {url}: {stats}\n")
        
        print(f"\nTotal Stats: {total_stats}")
        
    def cleanup(self):
        """Close resources"""
        if self.driver:
            self.driver.quit()
        self.session.close()


if __name__ == "__main__":
    # Example usage
    scraper = ImageScraper(save_dir="data/raw", use_selenium=False)
    
    urls = [
        "https://www.bbc.com/news",
        # Add more news URLs
    ]
    
    try:
        scraper.scrape_multiple_urls(urls, max_images_per_url=50)
    finally:
        scraper.cleanup()