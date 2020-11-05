from urllib.request import urlopen

html = urlopen(
    "http://mofanpy.com/static/scraping/basic-structure.html").read().decode('utf-8')

print(html)