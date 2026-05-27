"""
WEB LEARNER — Real web learning via Wikipedia API, Wikipedia search, and fallback sources.
Integrates with safety nets to filter bogus content before learning.
"""

import re
import time
import html as html_lib
from html.parser import HTMLParser
from urllib.request import urlopen, Request
from urllib.parse import quote


class TextExtractor(HTMLParser):
    """Extract visible text from HTML using stdlib only."""

    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.skip_tags = {'script', 'style', 'noscript', 'code', 'pre'}
        self.in_skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.skip_tags:
            self.in_skip += 1

    def handle_endtag(self, tag):
        if tag in self.skip_tags:
            self.in_skip = max(0, self.in_skip - 1)
        if tag in {'p', 'br', 'h1', 'h2', 'h3', 'h4', 'li', 'div', 'tr'}:
            self.text_parts.append(' ')

    def handle_data(self, data):
        if self.in_skip == 0 and data.strip():
            self.text_parts.append(data.strip())

    def get_text(self):
        text = ' '.join(self.text_parts)
        text = re.sub(r'\s+', ' ', text).strip()
        return text


class WikipediaLearner:
    """Learn from Wikipedia via its API."""

    WIKI_API = "https://en.wikipedia.org/w/api.php"
    WIKI_BASE = "https://en.wikipedia.org/wiki/"

    def search(self, query, limit=3):
        """Search Wikipedia for articles matching query."""
        url = (f"{self.WIKI_API}?action=query&list=search&srsearch={quote(query)}"
               f"&format=json&srlimit={limit}")
        try:
            req = Request(url, headers={'User-Agent': 'BiologicLLM/1.0'})
            with urlopen(req, timeout=10) as resp:
                data = resp.read().decode('utf-8')
            import json
            results = json.loads(data)
            pages = []
            for item in results.get('query', {}).get('search', []):
                pages.append({
                    'title': item['title'],
                    'snippet': re.sub(r'<[^>]+>', '', item.get('snippet', '')),
                    'pageid': item['pageid']
                })
            return pages
        except Exception as e:
            return [{'title': query, 'snippet': f'Search error: {e}', 'pageid': 0}]

    def fetch_article(self, title):
        """Fetch full article text by title."""
        url = (f"{self.WIKI_API}?action=query&prop=extracts&exintro&explaintext"
               f"&titles={quote(title)}&format=json")
        try:
            req = Request(url, headers={'User-Agent': 'BiologicLLM/1.0'})
            with urlopen(req, timeout=15) as resp:
                data = resp.read().decode('utf-8')
            import json
            result = json.loads(data)
            pages = result.get('query', {}).get('pages', {})
            for page_id, page_data in pages.items():
                extract = page_data.get('extract', '')
                cleaned = re.sub(r'\s+', ' ', extract).strip()
                return cleaned if cleaned else None
            return None
        except Exception as e:
            return None

    def fetch_full_article(self, title):
        """Fetch full article HTML and extract text (no API extract limit)."""
        url = f"{self.WIKI_BASE}{quote(title.replace(' ', '_'))}"
        try:
            req = Request(url, headers={
                'User-Agent': 'BiologicLLM/1.0',
                'Accept': 'text/html'
            })
            with urlopen(req, timeout=15) as resp:
                html = resp.read().decode('utf-8')

            extractor = TextExtractor()
            extractor.feed(html)

            text = extractor.get_text()
            # Filter to get meaningful content (skip nav, sidebar, etc.)
            lines = text.split('\n')
            content_lines = [l for l in lines if len(l.strip()) > 40]
            content = ' '.join(content_lines)

            return content[:5000] if content else None
        except Exception as e:
            return None


class WebLearner:
    """
    Complete web learning system: search -> fetch -> extract -> clean -> return.
    """

    def __init__(self, safety_system=None):
        self.wikipedia = WikipediaLearner()
        self.safety = safety_system
        self.learned_topics = set()
        self.search_cache = {}

    def learn(self, topic, max_chars=2000):
        """
        Search the web for information about a topic and return cleaned text.
        Returns dict with status and content.
        """
        print(f"  [WEB] Searching for: '{topic}'")

        # Check cache
        cache_key = topic.lower().strip()
        if cache_key in self.search_cache:
            print(f"  [WEB] Using cached result.")
            return self.search_cache[cache_key]

        # Step 1: Search Wikipedia
        results = self.wikipedia.search(topic)
        if not results or results[0].get('pageid', 0) == 0:
            result = {
                'success': False,
                'content': '',
                'source': 'none',
                'error': 'No search results found'
            }
            self.search_cache[cache_key] = result
            return result

        best_result = results[0]
        title = best_result['title']

        # Step 2: Fetch article
        print(f"  [WEB] Fetching: '{title}'")
        content = self.wikipedia.fetch_article(title)

        if not content or len(content) < 20:
            content = self.wikipedia.fetch_full_article(title)

        if not content or len(content) < 20:
            result = {
                'success': False,
                'content': '',
                'source': 'none',
                'error': 'Could not extract article content'
            }
            self.search_cache[cache_key] = result
            return result

        # Step 3: Clean up
        cleaned = self._clean_content(content, max_chars)

        # Step 4: Safety check
        if self.safety:
            allowed, reason, details = self.safety.pre_check(cleaned, source='web')
            if not allowed:
                result = {
                    'success': False,
                    'content': '',
                    'source': title,
                    'error': f'Safety gate: {reason}'
                }
                self.search_cache[cache_key] = result
                return result

        result = {
            'success': True,
            'content': cleaned,
            'source': title,
            'url': f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}",
            'snippet': best_result.get('snippet', '')
        }

        self.learned_topics.add(cache_key)
        self.search_cache[cache_key] = result
        return result

    def learn_multiple(self, topics, max_chars_per=1500):
        """Learn about multiple related topics."""
        all_content = []
        for topic in topics[:3]:  # Max 3 topics
            result = self.learn(topic, max_chars_per)
            if result['success']:
                all_content.append(f"--- {topic} ---\n{result['content']}")
            time.sleep(0.5)  # Be polite to Wikipedia API
        return '\n\n'.join(all_content)

    def _clean_content(self, text, max_chars):
        """Clean extracted text for learning."""
        text = re.sub(r'\[\d+\]', '', text)  # Remove citation markers
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'(?:may be|is a|refers to|edit|This article)', '', text)
        text = text[:max_chars]
        return text

    def available_sources(self):
        return ['Wikipedia (en.wikipedia.org)']


def demo_web_learner():
    """Demonstrate web learning."""
    print("=" * 60)
    print("WEB LEARNER DEMONSTRATION")
    print("=" * 60)

    learner = WebLearner()

    test_topics = [
        "Python programming language",
        "Neural network",
        "Artificial intelligence",
    ]

    for topic in test_topics:
        print(f"\n--- Learning: {topic} ---")
        result = learner.learn(topic, max_chars=1000)

        if result['success']:
            print(f"  Source: {result['source']}")
            print(f"  Content ({len(result['content'])} chars):")
            print(f"  {result['content'][:200]}...")
        else:
            print(f"  Failed: {result.get('error', 'unknown')}")

        time.sleep(0.5)

    print("\n" + "=" * 60)
    print(f"WEB LEARNER DEMO COMPLETE")
    print(f"Learned topics: {learner.learned_topics}")
    print("=" * 60)

    return learner


if __name__ == "__main__":
    demo_web_learner()
