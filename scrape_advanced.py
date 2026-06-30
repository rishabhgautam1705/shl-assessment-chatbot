import requests
from bs4 import BeautifulSoup
import json
import urllib.parse
import re
import os

BASE_URL = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/products/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def extract_text(element):
    return element.get_text(strip=True, separator=' ') if element else ""

def try_extract_field(text_content, field_name):
    """Attempt to extract fields like 'Duration: 15 mins' using regex."""
    pattern = re.compile(rf"{field_name}[:\s]+([^.\n]+)", re.IGNORECASE)
    match = pattern.search(text_content)
    return match.group(1).strip() if match else "Varies / Not Specified"

def scrape_catalog():
    print("Fetching catalog page...")
    response = requests.get(CATALOG_URL, headers=HEADERS)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    
    product_links = set()
    for a in soup.find_all("a"):
        href = a.get("href")
        if not href:
            continue
        if "/products/assessments/" in href and href != "/products/assessments/":
            full_url = urllib.parse.urljoin(BASE_URL, href)
            product_links.add(full_url)
    
    print(f"Found {len(product_links)} potential assessment links.")
    
    catalog = []
    
    for url in product_links:
        print(f"Scraping: {url}")
        try:
            res = requests.get(url, headers=HEADERS)
            res.raise_for_status()
            p_soup = BeautifulSoup(res.text, "html.parser")
            
            # Find the title
            title_element = p_soup.find("h1")
            name = extract_text(title_element) if title_element else url.split("/")[-2].replace("-", " ").title()
            
            if not name or "Assessments" in name:
                continue
                
            # Extract main text content to search for fields
            main_content = p_soup.get_text(separator=' ')
            
            # Extract description
            paragraphs = p_soup.find_all("p")
            desc_parts = [extract_text(p) for p in paragraphs if len(extract_text(p)) > 50]
            description = " ".join(desc_parts[:2])
            
            # Try to infer test_type
            test_type = "Unknown"
            url_lower = url.lower()
            if "personality" in url_lower:
                test_type = "Personality Assessment"
            elif "cognitive" in url_lower:
                test_type = "Cognitive Assessment"
            elif "behavior" in url_lower:
                test_type = "Behavioral Assessment"
            elif "skill" in url_lower or "simulation" in url_lower:
                test_type = "Skills & Simulations"
                
            # Extract skills tested, duration, level
            skills_tested = try_extract_field(main_content, "Skills")
            duration = try_extract_field(main_content, "Duration")
            level = try_extract_field(main_content, "Level")
            
            # Clean up fallbacks based on URL or title
            if skills_tested == "Varies / Not Specified":
                if "coding" in url_lower: skills_tested = "Programming, Software Engineering"
                elif "language" in url_lower: skills_tested = "Language proficiency (spoken/written)"
                else: skills_tested = "Job-specific competencies"
                
            if duration == "Varies / Not Specified":
                duration = "Typically 15-30 minutes"
                
            if level == "Varies / Not Specified":
                if "executive" in url_lower: level = "Executive"
                elif "manager" in url_lower: level = "Manager"
                else: level = "All Levels"

            catalog.append({
                "name": name,
                "description": description,
                "skills_tested": skills_tested,
                "duration": duration,
                "level": level,
                "test_type": test_type,
                "url": url
            })
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")
    
    os.makedirs("data", exist_ok=True)
    with open("data/catalog.json", "w") as f:
        json.dump(catalog, f, indent=2)
    print(f"Scraped {len(catalog)} assessments into data/catalog.json")

if __name__ == "__main__":
    scrape_catalog()
