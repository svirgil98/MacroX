import numpy as np
import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

COLS_CONST = ['CO_openweather', 'NO2_openweather', 'SO2_openweather', 'PM2_5_openweather', 'NO2_govt', 'SO2_govt','CO_govt', 'PM2.5_govt', 'CO_sat', 'NO2_sat','SO2_sat']
CLIMATES_CONST = ['Tropical Savanna','Mid-Latitude Steppe and Desert','Humid Subtropical','Mediterran','Tropical Monsoon','Oceanic Subtropical Highland','Marine West Coast']

if __name__ == "__main__":
	print('Retrieving data.')
	data_path = 'C:/Users/shrey/OneDrive/Documents/Georgia Tech OMSA/MacroXProject/Data/Cleaned/full_dataset.csv'
	data_set = pd.read_csv(data_path)
	
	grouping_path = 'C:/Users/shrey/OneDrive/Documents/Georgia Tech OMSA/MacroXProject/Data/Cleaned/city_groupings.csv'
	grouping_set = pd.read_csv(grouping_path,names=['location','climate'])
	
	full_df = data_set.join(grouping_set.set_index('location'),on='location')
	full_df = full_df.drop(columns=['NH3_govt','BC_sat'])	
	full_df['climate'] = full_df['climate'].replace(['Oceanic Subtropical Highland Climate'], 'Oceanic Subtropical Highland')
	full_df['NO2_sat'] = full_df['NO2_sat'].apply(lambda x: x * 1.88e-15)
	# full_df.plot()

	full_df['date'] = pd.to_datetime(full_df['date'])
	agg_df = full_df.groupby(['date','climate']).agg("mean").sort_values(by='date')

	# for col in COLS_CONST:
	# 	agg_df[col].unstack().plot()
	# 	plt.show()

	agg_df = agg_df.reset_index()
	# print(agg_df)
	for clim in CLIMATES_CONST:
		temp_df = agg_df.loc[agg_df['climate']==clim]
		temp_df = temp_df.drop(columns=COLS_CONST[0:4])
		date_col = temp_df['date'].reset_index()
		temp_df = temp_df.drop(columns=['date', 'climate'])
		temp_df = temp_df.dropna(axis='columns', how='all')
		col_names = temp_df.columns
		print(col_names)
		imputer = IterativeImputer(random_state=0)
		imputed = imputer.fit_transform(temp_df)
		print(imputed)
		imputed_df = pd.DataFrame(imputed,columns=col_names)
		imputed_df = imputed_df.join(date_col).set_index('date').reset_index()
		print(imputed_df)
		imputed_df.to_csv(clim+'.csv')
		# imputed_df.plot(x='date',y=COLS_CONST[4:])
		# plt.title(clim)
		# plt.show()