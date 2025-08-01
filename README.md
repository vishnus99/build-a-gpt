Building a GPT-2 model trained on text scraped from CS2 website hltv.org. Text was scraped using cloudscraper PyPi package and the GPT-2 model was built according to the LLMs from Scratch tutorial by Sebastian Raschka. The chapters directory shows the corresponding chapters to Sebastian's tutorial, and contains various notes on LLM theory from every point in the development process.

Features:
- Scraper built with BeautifulSoup and Cloudscraper
- Multi-head attention with pseudocode included for other attention types
- Layer normalization (before attention block and before feed-forward network)
- Right-padding and masking applied to dataloader collate function to handle incomplete batches
- Custom loss function to ignore padded tokens 
- Simple training/validation loop


