import cloudscraper
from bs4 import BeautifulSoup
import time
import logging
from typing import Optional, Dict, List, Tuple
import random
import re

class HLTVforumscraper:
    def __init__(self):
        self.scraper = cloudscraper.create_scraper(
            interpreter = "js2py",
            delay = 10,
            browser = {
                'browser': "chrome",
                'platform': "darwin",
                'mobile': False
            })
        self.scraper.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0'
        })
        self.base_url = "https://www.hltv.org"
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def _make_request(self, url: str) -> Optional[BeautifulSoup]:
        """
        Make a request with retry logic and proper delays.
        
        Args:
            url (str): URL to request
            
        Returns:
            Optional[BeautifulSoup]: Parsed HTML if successful, None otherwise
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Add random delay between requests
                time.sleep(random.uniform(2, 5))
                
                response = self.scraper.get(url)
                response.raise_for_status()
                
                # Check if we got a valid response
                if "Cloudflare" in response.text or "Just a moment" in response.text:
                    self.logger.warning("Cloudflare protection detected, retrying...")
                    time.sleep(10)  # Wait longer if we hit Cloudflare
                    continue
                    
                return BeautifulSoup(response.text, 'html.parser')
                
            except Exception as e:
                self.logger.error(f"Request failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))  # Exponential backoff
                else:
                    return None

    def scrape_forum_threads(self, forum_url: str) -> List[Dict[str, str]]:
        """
        Scrape forum threads from the HLTV forum page.
        
        Args:
            forum_url (str): URL of the forum page to scrape
            
        Returns:
            List[Dict[str, str]]: List of thread information dictionaries
        """
        threads = []
        
        try:
            self.logger.info(f"Scraping forum: {forum_url}")
            
            soup = self._make_request(forum_url)
            if not soup:
                self.logger.error("Failed to get forum page")
                return threads
            
            # Find all thread links
            thread_links = soup.select("a[href*='/forums/threads/']")
            self.logger.info(f"Found {len(thread_links)} thread links")
            
            for i, link in enumerate(thread_links, 1):
                try:
                    href = link.get('href')
                    title = link.get_text(strip=True)
                    
                    # Make sure href is absolute
                    if href.startswith('/'):
                        href = f"{self.base_url}{href}"
                    
                    self.logger.info(f"Processing thread {i}/{len(thread_links)}: {title}")
                    
                    # Get thread content
                    thread_content = self._get_thread_content(href)
                    
                    thread_info = {
                        'title': title,
                        'url': href,
                        'thread_id': self._extract_thread_id(href),
                        'content': thread_content
                    }
                    
                    threads.append(thread_info)
                    
                except Exception as e:
                    self.logger.error(f"Error processing thread {i}: {str(e)}")
                    continue
            
            self.logger.info(f"Successfully processed {len(threads)} threads")
            
        except Exception as e:
            self.logger.error(f"Error scraping forum threads: {str(e)}")
        
        return threads

    def _get_thread_content(self, thread_url: str) -> str:
        """
        Get the content from the forum-middle div of a thread.
        
        Args:
            thread_url (str): URL of the thread
            
        Returns:
            str: Content from the forum-middle div
        """
        try:
            soup = self._make_request(thread_url)
            if not soup:
                return "Failed to load thread"
            
            # Find the forum-middle div
            forum_middle = soup.select_one(".forum-middle")
            if forum_middle:
                return forum_middle.get_text(strip=True)
            else:
                return "No forum-middle content found"
                
        except Exception as e:
            self.logger.error(f"Error getting thread content: {str(e)}")
            return f"Error: {str(e)}"

    def _extract_thread_id(self, url: str) -> Optional[str]:
        """
        Extract thread ID from URL.
        
        Args:
            url (str): Thread URL
            
        Returns:
            Optional[str]: Thread ID if found
        """
        match = re.search(r'/forums/threads/(\d+)', url)
        return match.group(1) if match else None

    def save_threads_to_file(self, threads: List[Dict[str, str]], filename: str = "hltv_forum_threads.txt"):
        """
        Save scraped threads to a file.
        
        Args:
            threads (List[Dict[str, str]]): List of thread information
            filename (str): Output filename
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                for i, thread in enumerate(threads, 1):
                    f.write(f"Thread #{i}\n")
                    f.write(f"Title: {thread.get('title', 'N/A')}\n")
                    f.write(f"URL: {thread.get('url', 'N/A')}\n")
                    f.write(f"Thread ID: {thread.get('thread_id', 'N/A')}\n")
                    f.write(f"Content:\n{thread.get('content', 'N/A')}\n")
                    f.write("-" * 50 + "\n")
            
            self.logger.info(f"Saved {len(threads)} threads to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving threads to file: {str(e)}")

# Example usage
if __name__ == "__main__":
    scraper = HLTVforumscraper()
    
    # Scrape threads from the forum
    forum_url = "https://www.hltv.org/forums/counterstrike/120"  # Replace with your forum URL
    threads = scraper.scrape_forum_threads(forum_url)
    
    if threads:
        scraper.save_threads_to_file(threads)