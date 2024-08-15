# Solar wind forecast pipeline program
# Author: Rong Lin
# Date: 2024/08/016

# please use py39env on RONG's windows laptop.
import pandas as pd
import numpy as np
import datetime
import pfsspy

from downloading_utils import *
from gong_and_pfss_utils import *
from torch_utils import *

import torch
print(f"Torch version: {torch.__version__}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # input: epoches to be forecast
    # TODO: make the slide buttton...
    date_dict = {'date':[datetime.datetime(2021, 6, 11, 14),
                        datetime.datetime(2021, 6, 11, 15),
                        datetime.datetime(2021, 6, 12, 15),
                        datetime.datetime(2021, 6, 12, 16),
                        datetime.datetime(2021, 6, 12, 17),
                        datetime.datetime(2021, 6, 13, 16)]}

    # CALCULATE the date for 3, 4, 5 and 6 days ago, for each epoch
    date_df = pd.DataFrame(date_dict)
    date_df['3_days_ago'] = date_df['date'] + datetime.timedelta(days=-3)
    date_df['4_days_ago'] = date_df['date'] + datetime.timedelta(days=-4)
    date_df['5_days_ago'] = date_df['date'] + datetime.timedelta(days=-5)
    date_df['6_days_ago'] = date_df['date'] + datetime.timedelta(days=-6)

    # MERGE all dates from 3, 4, 5, and 6 days ago,
    # so that we don't need to download for multiple times
    melted_df = pd.melt(date_df, value_vars=['3_days_ago', '4_days_ago', '5_days_ago', '6_days_ago'], value_name='uniq_date')
    unique_dates = pd.to_datetime(melted_df['uniq_date']).drop_duplicates()
    unique_sorted_dates = unique_dates.sort_values().reset_index(drop=True)
    unique_sorted_dates = pd.DataFrame(unique_sorted_dates)
    unique_sorted_dates['path'] =''

    # DOWNLOAD required files
    year_month_day_previous_i = None
    for i_date in range(len(unique_sorted_dates)):
        date_input = unique_sorted_dates['uniq_date'][i_date]
        date_input = date_input.to_pydatetime()
        year4_month_str = str(date_input.year) + '%02d' % date_input.month
        year2_month_day_str = str(date_input.year)[-2:]+'%02d'%date_input.month+'%02d'%date_input.day
        if (year_month_day_previous_i is not None) and (year2_month_day_str == year_month_day_previous_i):
            pass
            # still use the "time_filename_dict_this_day"
        else:
            # get url of downloading page
            url0 = get_url_of_downloading_page(date_input, year4_month_str)
            
            time_filename_dict_this_day = get_downloading_page_dict(url0,date_input)   
            
            # analyse the downlaoding page, and get the filename list with corresponding file url
            r0 = requests.get(url0)
            soup0 = BeautifulSoup(r0.content,features="html.parser")
            links0 = soup0.findAll('a')

            # build dict for that page
            time_filename_dict_this_day = dict()
            for link0 in links0:
                if link0['href'].startswith('mrzqs'):
                    url1 = url0 + link0['href']
                    splitted_href = link0['href'].split('t')
                    hours_str = splitted_href[1][:2]
                    mins_str = splitted_href[1][2:4]
                    file_time = datetime.datetime(date_input.year,date_input.month,date_input.day,int(hours_str),int(mins_str))
                    time_filename_dict_this_day[file_time] = url1
            year_month_day_previous_i = year2_month_day_str        
        
        gz_fileurl = find_nearest_file(time_filename_dict_this_day, date_input)
        if gz_fileurl is None:
            gz_filename = None
            path_file_full = None
        else:
            gz_filename = gz_fileurl.split('/')[-1] # NoneType does not have it
            path = FOLDER + year4_month_str + '/' + gz_filename[:11]+'/'
            if not os.path.exists(path):
                os.makedirs(path)
            path_file = path + gz_filename
            if not os.path.exists(path_file):
                r2 = requests.get(gz_fileurl)
                data = r2.content
                with open(path_file, 'wb') as f:
                    f.write(data)
                path_file_full = path_file.replace('.gz', '')
                file_data = gzip.GzipFile(path_file)
                with open(path_file_full, "wb+") as pfu:
                    pfu.write(file_data.read())
                file_data.close()
            else:
                path_file_full = path_file.replace('.gz', '')
                pass
        # print(date_input, year2_month_day_str, url0, gz_filename, path_file_full)
        unique_sorted_dates.loc[i_date,'path'] = path_file_full

    # ARRANGE dataframe
    date_df = date_df.merge(unique_sorted_dates, left_on='3_days_ago', right_on='uniq_date',
                            how='left').drop(columns=['uniq_date']).rename(columns={'path': 'path_3_days_ago'})
    date_df = date_df.merge(unique_sorted_dates, left_on='4_days_ago', right_on='uniq_date',
                            how='left').drop(columns=['uniq_date']).rename(columns={'path': 'path_4_days_ago'})
    date_df = date_df.merge(unique_sorted_dates, left_on='5_days_ago', right_on='uniq_date',
                            how='left').drop(columns=['uniq_date']).rename(columns={'path': 'path_5_days_ago'})
    date_df = date_df.merge(unique_sorted_dates, left_on='6_days_ago', right_on='uniq_date',
                            how='left').drop(columns=['uniq_date']).rename(columns={'path': 'path_6_days_ago'})

    conditions = date_df[['path_3_days_ago', 'path_4_days_ago', 'path_5_days_ago', 'path_6_days_ago']].notnull().all(axis=1)
    date_df['all_paths_present'] = conditions

    # CHECK if can forecast
    can_forecast_indices = date_df.index[date_df['all_paths_present'] == True].tolist()
    can_forecast_indices
    input_for_model = list()
    for i in can_forecast_indices:
        input_for_model.append(get_ss_graphs(date_df,i))        
    input_for_model = np.array(input_for_model)
    input_for_model = torch.from_numpy(input_for_model)

    # MODELS
    MODEL_1 = Net().to(DEVICE)
    MODEL_1.load_state_dict(torch.load("./Models/published_model_1.pt",weights_only=True))
    MODEL_2 = Net().to(DEVICE)
    MODEL_2.load_state_dict(torch.load("./Models/published_model_2.pt",weights_only=True))
    MODEL_3 = Net().to(DEVICE)
    MODEL_3.load_state_dict(torch.load("./Models/published_model_3.pt",weights_only=True))
    MODEL_4 = Net().to(DEVICE)
    MODEL_4.load_state_dict(torch.load("./Models/published_model_4.pt",weights_only=True))
    MODEL_5 = Net().to(DEVICE)
    MODEL_5.load_state_dict(torch.load("./Models/published_model_5.pt",weights_only=True))
    MODEL_6 = Net().to(DEVICE)
    MODEL_6.load_state_dict(torch.load("./Models/published_model_6.pt",weights_only=True))
    MODEL_7 = Net().to(DEVICE)
    MODEL_7.load_state_dict(torch.load("./Models/published_model_7.pt",weights_only=True))
    for MODEL in [MODEL_1, MODEL_2, MODEL_3, MODEL_4, MODEL_5, MODEL_6, MODEL_7]:
        MODEL.eval()
        
    # INPUT
    x = input_for_model
    x = x.type(torch.FloatTensor)
    x = x.to(DEVICE)

    # RESULT DATAFRAME
    new_columns = ['date'] + [f'forecast_{i}' for i in range(8)]
    new_df = pd.DataFrame(columns=new_columns)

    new_df['date'] = date_df['date']
    for col in new_df.columns[1:]:
        new_df[col] = np.nan

    result_df = new_df

    # FORECAST
    for i in range(1,8):
        model = eval(f"MODEL_{i}")

        forecast_values = model(x)
        forecast_values = forecast_values.cpu().detach().numpy().flatten()

        result_df.loc[can_forecast_indices, f'forecast_{i}'] = forecast_values

    result_df.to_csv(f'./Result_{year2_month_day_str}.csv')