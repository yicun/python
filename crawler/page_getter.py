# coding=utf-8
import urllib


def get_html(url):
    page = urllib.urlopen(url)
    html = page.read()
    return html


html = get_html("http://tieba.baidu.com/p/2738151262")

print html

html = get_html("http://trend.baidu.lecai.com/ssq/redBaseTrend.action?startPhase=2016016&endPhase=2016045#chartTableWrapper")

print html
