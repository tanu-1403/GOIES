import feedparser
import httpx
from readability import Document
from bs4 import BeautifulSoup

from extractor import extract_intelligence
from utils import save_graph


RSS_FEEDS = [
    "https://www.reuters.com/world/rss",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://www.aljazeera.com/xml/rss/all.xml",
]


processed_urls = set()


async def fetch_article_text(url: str):
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url)

    html = r.text

    doc = Document(html)

    summary_html = doc.summary()

    soup = BeautifulSoup(summary_html, "html.parser")

    return soup.get_text(separator=" ")


async def ingest_rss_feed(graph, model="llama3.2"):
    for feed_url in RSS_FEEDS:
        feed = feedparser.parse(feed_url)

        for entry in feed.entries[:5]:
            url = entry.get("link")

            if not url or url in processed_urls:
                continue

            try:
                text = await fetch_article_text(url)

                if len(text) < 200:
                    continue

                extractions = extract_intelligence(text, model=model)

                from server import _update_graph

                _update_graph(extractions)

                processed_urls.add(url)

            except Exception as e:
                print("OSINT ingest error:", e)
