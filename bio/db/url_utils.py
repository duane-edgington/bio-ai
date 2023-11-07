# bio-ai, Apache-2.0 license
# Filename: bio/db/utils.py
# Description: utilities for bio-ai url inspection

import re
import requests
from bs4 import BeautifulSoup


def is_valid_url(url):
    try:
        response = requests.get(url)
        return response.status_code == 200  # HTTP status code 200 indicates success
    except requests.RequestException:
        return False  # URL is not reachable or invalid


def extract_image_links(url):
    # Fetch the HTML content
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        # Find all hrefs that end in .jpg or .png
        links = soup.find_all(href=re.compile(r'\.(jpg|png)$'))

        # Extract the href attribute values
        image_links = [link.get('href') for link in links]
        return image_links
    else:
        print("Failed to fetch the URL.")
        return []