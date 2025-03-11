import requests
import nltk
from urllib.parse import urlparse
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag, RegexpParser
from fastapi import FastAPI, Query, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import copy
import uvicorn
from bs4 import BeautifulSoup
from googlesearch import search  # Install with: pip install google
import re
from pydantic import BaseModel 
from openai import OpenAI
from collections import Counter
import string
from flask import Flask, request, jsonify
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import io
import pycurl
import subprocess
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = FastAPI() 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # Allow all origins, or specify a list of allowed origins
    allow_credentials=True,
    allow_methods=["*"],            # Allow all methods (GET, POST, etc.)
    allow_headers=["*"]             # Allow all headers
)
STOPWORDS = set(stopwords.words('english'))

# List of unwanted question phrases
UNWANTED_PHRASES = {
    "how to", "where to", "what are", "which is", "who can", "when to", 
    "why should", "why is", "how do", "can i", "is it", "best way to"
}
STOPWORDS = set(stopwords.words('english'))

def process_list(ul_tag, indent=1):
    """
    Recursively processes a <ul> tag to extract list items.
    Each list item is indented based on its depth.
    """
    lines = []
    indent_str = "  " * indent
    # Process only direct <li> children
    for li in ul_tag.find_all("li", recursive=False):
        # Get the text content of the list item (including nested text)
        li_text = li.get_text(" ", strip=True)
        if li_text:
            lines.append(f"{indent_str}- {li_text}")
        # Look for a nested <ul> inside the current <li>
        nested_ul = li.find("ul")
        if nested_ul:
            lines.extend(process_list(nested_ul, indent=indent+1))
    return lines

def create_chatgpt_context(summary_html: str) -> str:
    """
    Processes the summary HTML and returns a plain text structured summary.
    
    The output organizes the content into:
    - Menu items (from <ul> lists and their nested <li> items)
    - Sections with titles and associated content (from headings and paragraphs)
    """
    soup = BeautifulSoup(summary_html, 'html.parser')
    result_lines = []

    # Process all <ul> tags for menu items
    for ul in soup.find_all("ul"):
        menu_lines = process_list(ul, indent=1)
        if menu_lines:
            result_lines.append("Menu items:")
            result_lines.extend(menu_lines)
            result_lines.append("")  # Blank line for separation 

    # Process headings as section titles and capture immediately following paragraphs as section content.
    for heading_tag in soup.find_all(["h1", "h2", "h3", "h4", "h5"]):
        heading_text = heading_tag.get_text(strip=True)
        if heading_text:
            result_lines.append(f"Section title: {heading_text}")
            # Collect following sibling <p> tags until the next heading (or until no more siblings)
            sibling = heading_tag.find_next_sibling()
            while sibling and sibling.name == "p":
                content_text = sibling.get_text(strip=True)
                if content_text:
                    result_lines.append(f"Section content: {content_text}")
                sibling = sibling.find_next_sibling()
            result_lines.append("")  # Blank line for separation

    return "\n".join(result_lines)
def fetch_with_curl(url):
    command = ['curl', '-A', 'Mozilla/5.0', url]
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout
def fetch_with_pycurl(url):
    buffer = io.BytesIO()
    c = pycurl.Curl()
    c.setopt(c.URL, url)
    c.setopt(c.USERAGENT, 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:55.0) Gecko/20100101 Firefox/55.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36')
    c.setopt(c.WRITEDATA, buffer)
    c.perform()
    c.close()
    
    body = buffer.getvalue().decode('utf-8')
    return body
def summarize_text(text):
    """
    Dummy summarization function.
    Replace this with a call to ChatGPT's API to generate a summary.
    For demonstration, it returns the first 200 characters (if available).
    """
    return text[:200] + "..." if len(text) > 200 else text
def fetch_clean_content(url: str):
    """ Fetches, cleans, and extracts readable content from a webpage. """
    response_text = fetch_with_pycurl(url)

    soup = BeautifulSoup(response_text, 'html.parser')

    # Extract meta title and meta description
    title = soup.find("title").text.strip() if soup.find("title") else "N/A"
    meta_description = soup.find("meta", attrs={"name": "description"})
    meta_description = meta_description["content"].strip() if meta_description else "N/A"

   # Remove non-content elements (keep header, footer, nav intact).
    for tag in soup(['script', 'style', 'aside','form','img']):
        tag.decompose()

    # Extract the full cleaned text from the page.
    body_text = ' '.join(soup.stripped_strings)
    cleaned_text = clean_text(body_text)

    # Define the tags to extract for the summary.
    tags_to_extract = ["p", "span", "h1", "h2", "h3", "h4", "h5", "ul", "li", "a", "div"]
    summary_parts = []

    # Search recursively in the full page for matching tags.
    for tag in soup.find_all(lambda t: t.name in tags_to_extract):
        tag_copy = copy.deepcopy(tag)

        # Remove all inner <img> tags.
        for img in tag_copy.find_all("img"):
            img.decompose()

        # Remove all inner <input>, <select>, <textarea>, and <button> tags.
        for unwanted in tag_copy.find_all(["input", "select", "textarea", "button"]):
            unwanted.decompose()

        # If the tag is a <ul>, recursively remove attributes from its <li> children.
        if tag_copy.name == "ul":
            for li in tag_copy.find_all("li"):
                li.attrs = {}

        # For <div> elements, only include if they contain only text (i.e. no inner tags).
        if tag_copy.name == "div":
            # After removals, if the div contains any inner tags, skip it.
            if tag_copy.find(True) is not None:
                continue

        # Remove all attributes from the tag itself.
        tag_copy.attrs = {}

        # Append the cleaned tag's HTML to the summary.
        summary_parts.append(str(tag_copy))

    # Combine the parts in order to create the summary HTML.
    page_summary = "\n".join(summary_parts)
    page_summary_html = create_chatgpt_context(page_summary);

    return {
        "title": title,
        "description": meta_description,
        "full_cleaned_content": cleaned_text,
        "page_summary": page_summary_html
    }
def generate_keywords(title: str, description: str, content: str):
    """ Uses ChatGPT to generate relevant SEO keywords based on the website content. """
    
    prompt = f"""
    You are an expert SEO strategist and an expert in identifying business types from website information.

    Part 1: Identify the Business Type
    Please analyze the following website details and extract the primary business type (e.g., Legal Operations Support, Digital Marketing Agency, Business Directory Service, etc.) that best describes the business.
    Return only the business type as a short phrase.

    Website Information:
    - Title: {title}
    - Description: {description}

    Part 2: Extract SEO Keywords
    Using the business type you determined above, extract highly relevant SEO-friendly keywords from the website details. Tailor your keyword extraction to the identified business type.

    Website Information:
    - Title: {title}
    - Description: {description}
    - Content Excerpt: {content}

    Instructions:
    1. Focus on the services offered, key issues, and industry-specific topics represented in the content excerpt.
    2. Only consider those menu items that fall under categories like "services", "what we do", "what we offer", or "our products". Ignore other generic or navigational items (e.g. "Menu items", "Contact Us", "Who we are") that do not represent actual services or issues.
    3. Deduplicate and consolidate similar terms.
    4. Do NOT include generic or vague terms like "real change," "navigate disruption," or "unlock growth" unless they are directly tied to a specific service or issue.
    5. Extract SEO keywords that real users would search for to find a business of this type.
    6. Prioritize industry-specific and service-specific terms over broad marketing phrases.
    7. Return ONLY the final list of SEO keywords, separated by commas (NO extra text). Additionally, prepend the extracted business type to the top of the list in the format: "Type: <extracted business type>", followed by a comma and then the rest of the keywords.
    8. Do NOT include generic items like "about", "contact", "blog", "home", "videos", "photos", "pages", "teams", "podcasts" unless they are directly appended with service-specific terms.
    9. Additionally, exclude keywords that are irrelevant to the core focus of the business. For example, if the determined business type is "Legal Operations Support", do not include keywords like "AI Insights", "US Openings", "UK Openings", or "Asia-Pacific Openings".
    10. Finally, if any generic keywords appear (for instance, "Browse States", "Listing Id.", "Directions", "Maps"), modify them to be industry specific by appending or prepending the business type. For example, if the business type is "Business Directory Service":
    - "Browse States" should become "Browse States for business directory"
    - "Listing Id." should become "business directory Listing Id."
    - "Directions" should become "Directions for business directory"
    - "Maps" should become "Maps for business directory"

    Example Output:
    Type: Business Directory Service, keyword1, keyword2, keyword3, keyword4
    """
    chat_response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    keywords = chat_response.choices[0].message.content.strip()
    
    updated_keywords = keywords.split(", ")
    return updated_keywords
    
def clean_text(text: str):
    """ Cleans text by removing special characters, stopwords, and unnecessary spaces. """
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    words = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    words = [word for word in words if word not in STOPWORDS and len(word) > 2]  # Remove stopwords
    return ' '.join(words)
def fetch_keywords(url: str):
    """ Fetches keywords from a webpage. """
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        return {"error": f"Unable to access {url}"}

    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract title
    title = soup.find("title").text.strip() if soup.find("title") else "N/A"

    # Remove unnecessary tags
    for tag in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
        tag.decompose()

    # Extract visible body text
    body_text = ' '.join(soup.stripped_strings)

    # Extract meaningful keyword phrases
    search_phrases = extract_search_phrases(body_text)
    phrase_freq = keyword_frequency(search_phrases)
    tfidf_keywords = calculate_tfidf(body_text)

    return {
        "title": title,
        "top_keywords_by_frequency": dict(phrase_freq.most_common(10)),
        "top_keywords_by_tfidf": {k: round(v, 4) for k, v in tfidf_keywords[:10]}
    }

def extract_search_phrases(text):
    """ Extracts meaningful search queries and removes question words. """
    sentences = sent_tokenize(text.lower())  # Split into sentences
    search_phrases = []

    for sentence in sentences:
        words = word_tokenize(sentence)
        filtered_words = [word for word in words if word not in STOPWORDS]
        phrase = " ".join(filtered_words)

        # Remove phrases that start with unwanted words
        if not any(phrase.startswith(bad) for bad in UNWANTED_PHRASES) and len(filtered_words) >= 4:
            search_phrases.append(phrase)

    return search_phrases

def keyword_frequency(phrases):
    """ Counts phrase occurrences in the text. """
    return Counter(phrases)

def calculate_tfidf(text):
    """ Computes TF-IDF scores for multi-word search phrases. """
    vectorizer = TfidfVectorizer(ngram_range=(4,6), stop_words='english')  # Extract 4-6 word search queries
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()

    tfidf_scores = dict(zip(feature_names, tfidf_matrix.toarray()[0]))
    return sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
class FetchRequest(BaseModel):
    website_url: str
def fetch_meta_data(url):
    """
    Fetches the title, meta keywords, and meta description from a website.
    """
    headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except Exception as e:
        raise Exception(f"Error fetching URL: {e}")
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Extract title
    title_tag = soup.find("title")
    title = title_tag.text.strip() if title_tag else ""
    
    # Extract meta keywords
    meta_keywords = ""
    meta_kw = soup.find("meta", attrs={"name": "keywords"})
    if meta_kw and meta_kw.get("content"):
        meta_keywords = meta_kw["content"].strip()
    
    # Extract meta description
    meta_description = ""
    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc and meta_desc.get("content"):
        meta_description = meta_desc["content"].strip()
    
    return title, meta_keywords, meta_description

def split_and_clean(text):
    """
    Helper function to split a string using both comma and pipe as separators,
    then clean up whitespace.
    """
    # First, split by comma and pipe
    parts = []
    for sep in [',', '|']:
        # If parts is empty, split the original text;
        # otherwise, further split each current part.
        if not parts:
            parts = text.split(sep)
        else:
            new_parts = []
            for part in parts:
                new_parts.extend(part.split(sep))
            parts = new_parts
    # Clean up and filter out empty strings
    return [p.strip() for p in parts if p.strip()]

def extract_keywords(title, meta_keywords,meta_description):
    """
    Extract candidate keywords by splitting the title and meta keywords.
    
    1. Splits the title by the pipe ('|') and comma (',') characters.
    2. Splits the meta keywords by commas.
    3. Combines and returns a unique list of keywords.
    """
    keywords = []
    
    # Process title: split by '|' then by ','
    if title:
        for part in title.split("|"):
            for subpart in part.split(","):
                keyword = subpart.strip()
                if keyword and keyword not in keywords:
                    keywords.append(keyword)

    """
    Extract candidate keywords by processing:
      1. The title (split by '|' and ',')
      2. Meta keywords (split by ',' and '|')
      3. Main objects (noun phrases) extracted from the combined title and meta description.
      
    Additional processing:
      - Further split any keyword that contains commas or pipes.
      - Remove undesired keywords such as standalone locations (e.g., "Michigan")
        or fillers (e.g., "a").
      - Ensure uniqueness ignoring case.
    """
    keywords_set = {}
    
    def add_keyword(kw):
        kw_clean = kw.strip()
        if not kw_clean:
            return
        lower_kw = kw_clean.lower()
        # Remove undesired standalone tokens (modify the set as needed)
        if lower_kw in {"michigan", "a"}:
            return
        # Remove keywords that are too short (could be noise)
        if len(kw_clean) < 2:
            return
        # Use the lower case version as key to ensure uniqueness
        if lower_kw not in keywords_set:
            keywords_set[lower_kw] = kw_clean
    
    # Process title: split by both ',' and '|'
    if title:
        for part in split_and_clean(title):
            add_keyword(part)
    
    # Process meta keywords: split by both ',' and '|'
    if meta_keywords:
        for part in split_and_clean(meta_keywords):
            add_keyword(part)
    
    # Combine title and meta description for noun phrase extraction
    combined_text = ""
    if title:
        combined_text += title + " "
    if meta_description:
        combined_text += meta_description
    
    # Extract noun phrases (main objects) using a simple grammar
    sentences = sent_tokenize(combined_text)
    grammar = "NP: {<DT>?<JJ>*<NN.*>+}"
    cp = RegexpParser(grammar)
    extracted_objects = set()
    
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        tagged = pos_tag(tokens)
        tree = cp.parse(tagged)
        for subtree in tree.subtrees():
            if subtree.label() == 'NP':
                phrase = " ".join(word for word, tag in subtree.leaves())
                # Only keep phrases with more than one word to avoid noise
                if len(phrase.split()) > 1:
                    extracted_objects.add(phrase)
    
    # Further process extracted noun phrases by splitting on commas and pipes
    for phrase in extracted_objects:
        for sub in split_and_clean(phrase):
            add_keyword(sub)
                    
    # Process meta keywords: split by ','
    if meta_keywords:
        for keyword in meta_keywords.split(","):
            keyword = keyword.strip()
            if keyword and keyword not in keywords:
                keywords.append(keyword)
    
    return list(keywords_set.values())
    
def get_google_ranking(keyword, domain, num_results=20):
    """
    Searches Google for the given keyword and returns an array containing:
    
    1) search_result: an array of the top 10 results.
    2) ranking: the ranking position of the website (if found within the top num_results), 
       or "Above 20" if not found.
    """
    try:
        results = list(search(keyword, num_results=num_results, region="us"))
        search_result = results[:10]  # Top 10 results
        
        ranking = None
        for idx, result in enumerate(results):
            if domain in result:
                ranking = idx + 1  # Rankings are 1-indexed
                break
        if ranking is None:
            ranking = "Above 20"
        
        return [search_result, ranking]
    except Exception as e:
        return [[], f"Error: {e}"]

def get_google_ranking_list(keyword, num_results=10):
    """
    Searches Google for the given keyword and returns a list of domain names
    extracted from the search results (limited to the top num_results).
    """
    try:
        results = list(search(keyword, num_results=num_results, region="us"))
        domains = []
        for result in results:
            url = ""
            # If the result is a string, assume it's a URL.
            if isinstance(result, str):
                url = result
            # If the result is a dictionary, try to extract the "url" key.
            elif isinstance(result, dict) and "url" in result:
                url = result["url"]
            if url:
                domain = urlparse(url).netloc
                if domain:
                    domains.append(domain)
        return {"search_result": domains}
    except Exception as e:
        return f"Error: {e}"
@app.post("/api/fetch_keywords")
def fetch_keywords_endpoint(req: FetchRequest):
    """ FastAPI endpoint to fetch SEO keywords from a given URL. """
    website_url = req.website_url
    site_data = fetch_clean_content(website_url)
    if "error" in site_data:
        return site_data

    keywords = generate_keywords(site_data["title"], site_data["description"], site_data["page_summary"])

    return {
        "title": site_data["title"],
        "description": site_data["description"],
        "full_cleaned_content": site_data["full_cleaned_content"],
        "full_summary": site_data["page_summary"],
        "generated_keywords": keywords
    }
@app.post("/api/fetch")
def extract(req: FetchRequest):


    website_url = req.website_url
    
    try:
        # Fetch meta data
        site_data = fetch_clean_content(website_url)
        if "error" in site_data:
            return site_data

        top_keywords = generate_keywords(site_data["title"], site_data["description"], site_data["page_summary"])
        title = site_data["title"]
        meta_description = site_data["description"]
       
        
        # For each keyword, get the Google ranking
        results = []
        for keyword in top_keywords:
            ranking = get_google_ranking(keyword, website_url)
            results.append({"keyword": keyword, "google_ranking": ranking[1],"search_result":ranking[0]})
        
        response_payload = {
            "website": website_url,
            "meta": {
                "title": title,
                "meta_description": meta_description,
            },
            "top_keywords": top_keywords,
            "rankings": results
        }
        return response_payload
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
   uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
