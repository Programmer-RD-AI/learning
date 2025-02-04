import requests
from bs4 import *

url = "https://coinmarketcap.com/"
result = requests.get(url).text
doc = BeautifulSoup(result, "html.parser")
tbody = doc.tbody
trs = tbody.contents

prices = {}
for tr in trs[:10]:
    name, price = tr.contents[2:4]
    name = name.p.string
    price = price.find("a").text
    prices[name] = price
print(prices)
