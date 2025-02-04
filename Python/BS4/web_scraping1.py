from bs4 import BeautifulSoup
import requests

url = "https://www.newegg.com/gigabyte-geforce-rtx-3080-gv-n3080gaming-oc-10gd/p/N82E16814932459?Description=3080&cm_re=3080-_-14-932-459-_-Product"

result = requests.get(url)

doc = BeautifulSoup(result.text, "html.parser")

prices = doc.find_all(text="$")
prices = prices[0].parent.text
