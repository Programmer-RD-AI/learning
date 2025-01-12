from bs4 import *
import re
with open("index2.html", "r") as f:
    doc = BeautifulSoup(f, "html.parser")

# tag = doc.find('option')
# tag['value'] = 'new val'
# tag['required'] = True
# print(tag)
# print(tag.attrs)
# tag = doc.find_all(['p','div','li'])
# print(tag)


# tag = doc.find_all(['option'],text='Undergraduate')
# print(tag[0].text)

# tag = doc.find_all(class_='btn-item')
# print(tag[0].text)

# tag = doc.find_all(text=re.compile('\$.*'))
# print(tag[1].text.strip())


tag = doc.find_all(text=re.compile('\$.*'),limit=1)
print(tag[0].text.strip())
