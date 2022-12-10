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
	scores = {k:{} for k in CLIMATES_CONST}
	errors = {k:{} for k in CLIMATES_CONST}

	for clim in CLIMATES_CONST:

		## LOAD DATA
		clim_data_set = pd.read_csv(clim_data_path+clim+'.csv')
		mask = (clim_data_set['date'] > '2020-11-24') & (clim_data_set['date'] < '2022-7-25')
		clim_data_set = clim_data_set.loc[mask]
		clim_data_set['date'] = pd.to_datetime(clim_data_set['date'])
		clim_ow_set = open_weather_true.copy().loc[open_weather_true['climate'] == clim]
		clim_ow_set['date'] = pd.to_datetime(clim_ow_set['date'])
		clim_full_set = clim_ow_set.join(clim_data_set.set_index('date'),on='date')
		clim_full_set = clim_full_set[clim_full_set['CO_openweather'].notna()]
		# print(clim_full_set)

		## SPLIT INTO TRAINING AND TESTING DATASET
		clim_full_set = clim_full_set.drop(['date','index','climate'],axis=1)
		sc = StandardScaler()
		sc.fit_transform(clim_full_set)
		clim_train, clim_test = train_test_split(clim_full_set, test_size=0.2)

		
		# ## LINEAR REGRESSION
		# scores[clim]['linreg'] = {}
		# errors[clim]['linreg'] = {}
		# for emission in OW_COLS_CONST:
		# 	filter_col = [col for col in clim_train if col in TEST_COLS]
		# 	X = clim_train[filter_col]
		# 	y = clim_train[emission+'_openweather']
		# 	linreg = LinearRegression()
		# 	linreg.fit(X,y)
		# 	scores[clim]['linreg'][emission] = linreg.score(X,y)
		# 	y_pred = linreg.predict(clim_test[filter_col])
		# 	errors[clim]['linreg'][emission] = mean_absolute_error(clim_test[emission+'_openweather'].values,y_pred)

		# ## LASSO 
		# scores[clim]['lasso'] = {}
		# errors[clim]['lasso'] = {}
		# for emission in OW_COLS_CONST:
		# 	filter_col = [col for col in clim_train if col in TEST_COLS]
		# 	X = clim_train[filter_col]
		# 	y = clim_train[emission+'_openweather']
		# 	lassoreg = LassoCV(cv=5).fit(X,y)
		# 	scores[clim]['lasso'][emission] = lassoreg.score(X,y)
		# 	y_pred = lassoreg.predict(clim_test[filter_col])
		# 	errors[clim]['lasso'][emission] = mean_absolute_error(clim_test[emission+'_openweather'].values,y_pred)

		# ## RFE
		# scores[clim]['rfe'] = {}
		# errors[clim]['rfe'] = {}
		# for emission in OW_COLS_CONST:
		# 	filter_col = [col for col in clim_train if col in TEST_COLS]
		# 	X = clim_train[filter_col]
		# 	y = clim_train[emission+'_openweather']
		# 	estimator = SVR(kernel="linear")
		# 	rfe_selector = RFE(estimator,step=1)
		# 	rfe_selector = rfe_selector.fit(X,y)
		# 	scores[clim]['rfe'][emission] = rfe_selector.score(X,y)
		# 	y_pred = rfe_selector.predict(clim_test[filter_col])
		# 	errors[clim]['rfe'][emission] = mean_absolute_error(clim_test[emission+'_openweather'].values,y_pred)

		# ## RANDOM FOREST
		# scores[clim]['rf'] = {}
		# errors[clim]['rf'] = {}
		# for emission in OW_COLS_CONST:
		# 	filter_col = [col for col in clim_train if col in TEST_COLS]
		# 	X = clim_train[filter_col]
		# 	y = clim_train[emission+'_openweather']
		# 	rfmod = RandomForestRegressor().fit(X,y)
		# 	scores[clim]['rf'][emission] = rfmod.score(X,y)
		# 	y_pred = rfmod.predict(clim_test[filter_col])
		# 	errors[clim]['rf'][emission] = mean_absolute_error(clim_test[emission+'_openweather'].values,y_pred)

		# ## SVR
		# scores[clim]['svr'] = {}
		# errors[clim]['svr'] = {}
		# for emission in OW_COLS_CONST:
		# 	filter_col = [col for col in clim_train if col in TEST_COLS]
		# 	X = clim_train[filter_col]
		# 	y = clim_train[emission+'_openweather']
		# 	svr = SVR().fit(X,y)
		# 	scores[clim]['svr'][emission] = svr.score(X,y)
		# 	y_pred = svr.predict(clim_test[filter_col])
		# 	errors[clim]['svr'][emission] = mean_absolute_error(clim_test[emission+'_openweather'].values,y_pred)

		## RANDOM FOREST CROSS VALIDATION
		param_grid = {'n_estimators':[50,100,200,500],'max_depth':[2,5,7,9]}
		# param_grid = {'n_estimators':[50,100]}
		for emission in OW_COLS_CONST:
			filter_col = [col for col in clim_train if col in TEST_COLS]
			X = clim_train[filter_col]
			y = clim_train[emission+'_openweather']
			rfmod = RandomForestRegressor()
			grid_clf = GridSearchCV(rfmod, param_grid, cv=10)
			grid_clf.fit(X,y)
			print(clim+' '+emission+':', grid_clf.best_params_)


	# print(scores)
	# print(errors)
	print('Done.')
	# score_df = pd.DataFrame.from_dict({(i,j):scores[i][j] for i in scores.keys() for j in scores[i].keys()},orient="index")
	# score_df.to_csv('rf_scores.csv')
	# error_df = pd.DataFrame.from_dict({(i,j):errors[i][j] for i in errors.keys() for j in errors[i].keys()},orient="index")
	# error_df.to_csv('rf_errors.csv')