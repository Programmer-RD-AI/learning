from urllib.request import urlopen
from recipe_scrapers import scrape_html

# Example recipe URL
url = "https://www.structlog.org/en/stable/"
# retrieve the recipe webpage HTML
html = urlopen(url).read().decode("utf-8")

# pass the html alongside the url to our scrape_html function
scraper = scrape_html(html, org_url=url)

print(html, scraper)
from markdownify import markdownify as md


print(md(html))
