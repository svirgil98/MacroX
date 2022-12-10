import numpy as np
import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

CLIMATES_CONST = ['Tropical Savanna','Mid-Latitude Steppe and Desert','Humid Subtropical','Mediterran','Tropical Monsoon','Oceanic Subtropical Highland','Marine West Coast']
OW_COLS_CONST = ['CO','NO2','SO2','PM2.5']
TEST_COLS = ['CO_govt','CO_sat','NO2_govt','NO2_sat','SO2_govt','SO2_sat','PM2.5_govt']

if __name__ == "__main__":
	print('Retrieving data.')
	data_path = 'C:/Users/shrey/OneDrive/Documents/Georgia Tech OMSA/MacroXProject/Data/Cleaned/full_dataset.csv'
	data_set = pd.read_csv(data_path)
	grouping_path = 'C:/Users/shrey/OneDrive/Documents/Georgia Tech OMSA/MacroXProject/Data/Cleaned/city_groupings.csv'
	grouping_set = pd.read_csv(grouping_path,names=['location','climate'])
	
	full_df = data_set.join(grouping_set.set_index('location'),on='location')
	full_df['climate'] = full_df['climate'].replace(['Oceanic Subtropical Highland Climate'], 'Oceanic Subtropical Highland')
	full_df = full_df.rename(columns={'PM2_5_openweather' : 'PM2.5_openweather'})
	full_df['date'] = pd.to_datetime(full_df['date'])
	agg_df = full_df.groupby(['date','climate']).agg("mean").sort_values(by='date')
	agg_df = agg_df.reset_index()
	open_weather_df = agg_df[['date','climate','CO_openweather', 'NO2_openweather','SO2_openweather','PM2.5_openweather']].copy()
	
	# print(open_weather_df)
	true_data_mask = (open_weather_df['date'] > '2020-11-24') & (open_weather_df['date'] < '2022-7-25')
	empty_data_mask = (open_weather_df['date'] < '2020-11-24')
	open_weather_true = open_weather_df.loc[true_data_mask]
	open_weather_empty = open_weather_df.loc[empty_data_mask]
	# print(open_weather_true)


	clim_data_path = 'C:/Users/shrey/OneDrive/Documents/Georgia Tech OMSA/MacroXProject/Python Code/'
	for clim in CLIMATES_CONST:
		if clim == 'Marine West Coast':

			## LOAD DATA
			clim_data_set = pd.read_csv(clim_data_path+clim+'.csv')
			mask = (clim_data_set['date'] > '2020-11-24') & (clim_data_set['date'] < '2022-7-25')
			clim_data_set = clim_data_set.loc[mask]
			clim_data_set['date'] = pd.to_datetime(clim_data_set['date'])
			clim_ow_set = open_weather_true.copy().loc[open_weather_true['climate'] == clim]
			clim_ow_set['date'] = pd.to_datetime(clim_ow_set['date'])
			clim_full_set = clim_ow_set.join(clim_data_set.set_index('date'),on='date')
			clim_full_set = clim_full_set[clim_full_set['CO_openweather'].notna()]

			clim_empty_set = pd.read_csv(clim_data_path+clim+'.csv')
			mask = (clim_empty_set['date'] < '2020-11-24')
			clim_empty_set = clim_empty_set.loc[mask]
			clim_empty_set['date'] = pd.to_datetime(clim_empty_set['date'])
			
			ow_est_df = pd.DataFrame(data=clim_empty_set['date'],columns=['date','CO_openweather','NO2_openweather','SO2_openweather','PM2.5_openweather'])
			print(ow_est_df)
			## RANDOM FOREST CROSS VALIDATION
			param_data = pd.read_csv('C:/Users/shrey/OneDrive/Documents/Georgia Tech OMSA/MacroXProject/Python Code/rf_optimal_params.csv');
			for emission in OW_COLS_CONST:
				filter_col = [col for col in clim_data_set if col in TEST_COLS]
				X = clim_full_set[filter_col]
				y = clim_full_set[emission+'_openweather']
				params = param_data.loc[param_data['Climate/Emission'] == clim][emission].values[0]
				depth = int(params.split("max_depth: ",1)[1].split(',')[0])
				ests = int(params.split("n_estimators: ",1)[1])
				print(depth,ests)
				rfmod = RandomForestRegressor(max_depth=7,n_estimators=ests).fit(X,y)
				y_pred = rfmod.predict(clim_empty_set[filter_col])
				ow_est_df[emission+'_openweather'] = y_pred
				# print(clim+' '+emission+':', grid_clf.best_params_)
			print(ow_est_df)
			concat_df = pd.concat([ow_est_df,clim_ow_set]).drop(columns='climate')
			concat_df.to_csv(clim+'_FINAL_data.csv')


	print('Done.')