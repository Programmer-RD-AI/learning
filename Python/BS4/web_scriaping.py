from bs4 import BeautifulSoup

with open('index.html','r') as f:
    doc = BeautifulSoup(f,'html.parser')

# tag = doc.title
# print(tag.string)

tag = doc.find_all('p')[0]
print(tag.find_all('b'))
