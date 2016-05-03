# coding=utf-8
import urllib
from bs4 import BeautifulSoup

url = "http://trend.baidu.lecai.com/ssq/redBaseTrend.action?startPhase=2016016&endPhase=2016045#chartTableWrapper"
html = urllib.urlopen(url).read()
print html

soup = BeautifulSoup(html)

print soup.prettify()
