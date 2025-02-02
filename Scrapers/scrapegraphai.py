from scrapegraph_py import Client

client = Client(api_key="sgai-d1b0fe77-25ca-43a6-8541-da22c9a86761")

response = client.markdownify(
    website_url="https://docs.scrapy.org/en/latest/topics/commands.html"
)

print(response)
