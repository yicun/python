# coding=utf-8
import re
import sys

from bs4 import BeautifulSoup
from selenium.common.exceptions import NoSuchElementException

reload(sys)
sys.setdefaultencoding("utf-8")

from selenium import webdriver

driver = webdriver.Chrome('/Users/chyc/Workspaces/Codes/mine/python/crawler/chromedriver')

# 设置等待时间
driver.implicitly_wait(30)

# 打开有赞登入界面
driver.get(url='https://login.youzan.com/sso/index?service=kdt')

# 输入用户名和密码
account = raw_input('手机号码:')
# account = '15201926956'
driver.find_element_by_name('account').send_keys(account)
password = raw_input('登路密码:')
# password = 'explicit'
driver.find_element_by_name('password').send_keys(password)
captcha_code = raw_input('验证码:')
driver.find_element_by_name('captcha_code').send_keys(captcha_code)

# 点击登录
driver.find_element_by_css_selector('.login-btn').click()
driver.forward()

# 进入商户
while driver.find_element_by_css_selector('div.team-icon.done') is None:
    driver.implicitly_wait(1)
    pass

driver.find_element_by_css_selector('div.team-icon.done').click()
driver.forward()

# 进入客户管理
while driver.find_element_by_css_selector('i.sidebar-icon.sidebar-icon-fans') is None:
    driver.implicitly_wait(1)
    pass

driver.find_element_by_css_selector('i.sidebar-icon.sidebar-icon-fans').click()
driver.forward()

# 进入标签管理
while driver.find_element_by_link_text('标签管理') is None:
    driver.implicitly_wait(1)
    pass

driver.find_element_by_link_text('标签管理').click()
driver.forward()

# 输出文件
output = open('weixin_fans.csv', 'w')
output.write('flag_id,flag_name,fan_id,fan_weixin_name,fan_level,fan_point,fan_start_time,fan_last_time,fan_last_order,fan_order_cnt,fan_avg_price\n')

# 抓取标签列表
flag_html = BeautifulSoup(driver.page_source, 'html.parser')
flag_list = flag_html.findAll('tr', attrs={'data-tag-id': re.compile('^[\d]+$')})
for flag in flag_list:
    try:
        flag_id = flag.attrs['data-tag-id']
        flag_name = flag.find_all('td', recursive=False, limit=1)[0].text.encode('utf-8').strip()
        flag_url = 'https:' + flag.find_all('a', attrs={'class': 'new_window'}, limit=1)[0].attrs['href']
        print '\n%s, %s, %s' % (flag_id, flag_name, flag_url)
        # 打开标签
        driver.get(flag_url)
        driver.forward()

        while True:
            fans_html = BeautifulSoup(driver.page_source, 'html.parser')
            fans_list = fans_html.findAll('tr', attrs={'data-fans-id': re.compile('^[\d]+$')})
            # 抓取微信名称
            for fan in fans_list:
                fan_id = fan.attrs['data-fans-id']
                fan_name = fan.findAll('a', attrs={'class': re.compile('js-fans-nickname')}, limit=1)[0].text.encode('utf-8').strip()
                td_list = fan.findAll('td', recursive=False)
                fan_name = td_list[1].text.encode('utf-8').strip()
                fan_level = td_list[2].text.encode('utf-8').strip()
                fan_point = td_list[3].text.encode('utf-8').strip()
                fan_start_time = td_list[4].text.encode('utf-8').strip()
                fan_last_time = td_list[5].text.encode('utf-8').strip()
                fan_last_order = td_list[6].text.encode('utf-8').strip()
                fan_order_cnt = td_list[7].text.encode('utf-8').strip()
                fan_avg_price = td_list[8].text.encode('utf-8').strip()

                print '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s' \
                      % (flag_id, flag_name, fan_id, fan_name, fan_level, fan_point, fan_start_time,
                         fan_last_time, fan_last_order, fan_order_cnt, fan_avg_price)
                output.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n'
                             % (flag_id, flag_name, fan_id, fan_name, fan_level, fan_point, fan_start_time,
                                fan_last_time, fan_last_order, fan_order_cnt, fan_avg_price))
                output.flush()
            try:
                next_page = fans_html.find_all('a', attrs={'class': 'next'}, limit=1)[0].attrs['href']
                driver.get(next_page)
                driver.forward()
            except NoSuchElementException:
                break
    except IndexError:
        continue

output.close()

# 关闭浏览器
driver.implicitly_wait(100)
driver.quit()
