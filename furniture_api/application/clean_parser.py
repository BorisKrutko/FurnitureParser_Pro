from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from bs4 import NavigableString, Tag

# Настройки для запросов
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.google.com/',
    'Accept-Encoding': 'identity'  
}
banned_symbols = {'$', '?', '!', '.', '€', '%', '*', '+', '=', '\\', ':'}

def get_text_limited_depth(element, max_depth=3):
    """Извлекает текст до заданной глубины вложенности."""
    texts = []

    def recurse(node, depth):
        if depth > max_depth:
            return
        if isinstance(node, NavigableString):
            text = str(node).strip()
            if text:
                texts.append(text)
        elif isinstance(node, Tag):
            for child in node.contents:
                recurse(child, depth + 1)

    recurse(element, depth=0)
    return texts


def is_valid_url(url):
    """Проверяет, является ли URL валидным (имеет схему и домен)."""
    parsed = urlparse(url)
    return bool(parsed.scheme) and bool(parsed.netloc)


def check_and_add_tags(text, texts):
    if not any(c.isalpha() for c in text):
            return

    text = ' '.join(text.split())  
    if (2 <= len(text.split()) <= 15 
        and not any(sym in text for sym in banned_symbols)
        and text.lower() != text):
        texts.add(text)


def extract_informative_text(soup):
    main_content = soup.find('main') or soup
    
    # Удаляем ненужные элементы (скрипты, цены и т.д.)
    for element in main_content(['script', 'style', 'meta', 'noscript', 'link']):
        element.decompose()
    
    texts = set()

    # with class "...product.."
    product_elements = main_content.select("""
        [class*='product' i],
        [class*='item' i],
        [class*='title' i],
        [class*='goods' i],
        [class*='card' i],
        [class*='catalog' i]
    """)
    for text in get_text_limited_depth(product_elements, max_depth=3):
        check_and_add_tags(text, texts)
        
    # <a>
    for link in main_content.find_all('a'):
        for link_text in get_text_limited_depth(link, max_depth=4):
            check_and_add_tags(link_text, texts)    

    # <h1><h2>...
    for header in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        header_text = ' '.join(header.get_text(strip=True).split())
        check_and_add_tags(header_text, texts)
    
    return '. '.join(texts) if texts else ""


def parser(url):    
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, 'html.parser')
    return extract_informative_text(soup)
   