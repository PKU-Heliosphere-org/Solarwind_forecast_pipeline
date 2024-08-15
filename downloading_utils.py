# DOWNLOADING dependencies: modules, constants and functions
import os, gzip
import requests
from bs4 import BeautifulSoup
import datetime
    
URL_HEADER = 'https://gong2.nso.edu/oQR/zqs/'
FOLDER = 'gong_data/'

def find_nearest_file(time_filename_dict, input_datetime, time_diff=0.5):
    # time diff is in hour(s)
    # the dict: key is time
    # find corresponding value in dict, with the key-time nearest to the input_datetime
    if not time_filename_dict:
        return None
    nearest_dt = min(time_filename_dict.keys(), key=lambda dt: abs(dt - input_datetime))
    if abs(nearest_dt - input_datetime) > datetime.timedelta(hours=time_diff):
        return None
    else:
        return time_filename_dict[nearest_dt]
        
def get_url_of_downloading_page(date_inp, _year4_month_str):
    year2_month_day_str = str(date_inp.year)[-2:] + '%02d' % date_inp.month + '%02d' % date_inp.day
    url = URL_HEADER + _year4_month_str + '/mrzqs' + year2_month_day_str + '/'
    return url
    
def get_downloading_page_dict(url,date_inp):
    # analyse the downlaoding page, and get the filename list with corresponding file url
    r0 = requests.get(url)
    soup0 = BeautifulSoup(r0.content,features="html.parser")
    links0 = soup0.findAll('a')
    # build dict for that page
    res = dict()
    for link0 in links0:
        if link0['href'].startswith('mrzqs'):
            url1 = url + link0['href']
            splitted_href = link0['href'].split('t')
            hours_str = splitted_href[1][:2]
            mins_str = splitted_href[1][2:4]
            file_time = datetime.datetime(date_inp.year,date_inp.month,date_inp.day,int(hours_str),int(mins_str))
            res[file_time] = url1
    return res    