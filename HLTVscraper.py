from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time


options = webdriver.ChromeOptions()
options.add_argument("--headless=new")
options.add_argument("--disable-gpu")
options.add_argument("--window-size=1920,1080")
options.add_argument("--no-sandbox")
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)


driver.get("https://www.hltv.org/forums/counterstrike/120")
time.sleep(2)

thread_cells = driver.find_elements(By.CSS_SELECTOR, "td.name")

thread_links = []

for cell in thread_cells:
    try:
        a_tag = cell.find_element(By.TAG_NAME, "a")
        thread_title = a_tag.text
        thread_url = a_tag.get_attribute("href")
        thread_links.append(thread_title, thread_url)
    except:
        continue

for link in thread_links:
    print(link)

driver.quit()
def scrape_thread(link):
    driver.get(link)
    time.sleep(2)

    try:
        post_contents = driver.find_element(By.CSS_SELECTOR, "div.forum-middle")
        posts = [post.text for post in post_contents if post.text.strip()]
        return posts
    except:
        print (f"Error scraping thread: {link}")


all_posts = []
for title,link in thread_links:
    print(f"Scraping thread: {link}")
    posts = scrape_thread(link)

    for post in posts:
        all_posts.append({
            "thread title": title,
            "post": post
        })

with open("hltv_forum_posts.txt", "w", encoding="utf-8") as f:
    for post in all_posts:
        f.write(f"Thread Title: {post['thread title']}\n")
        f.write(f"Post: {post['post']}\n")
        f.write("="*50 + "\n")

print("Scraping complete.")