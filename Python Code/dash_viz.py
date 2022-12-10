from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from sklearn.preprocessing import MinMaxScaler


raw_data = pd.read_csv("C:/Users/shrey/OneDrive/Documents/Georgia Tech OMSA/MacroXProject/Data/Aggreagted Data/subtropical_imp_agg.csv",index_col=0)
raw_data.sort_values("Date", inplace=True)
raw_data.set_index('Date')
data = raw_data.copy()
scaler = MinMaxScaler()
data[['CO','NO2','SO2','PM2_5']] = scaler.fit_transform(data[['CO','NO2','SO2','PM2_5']])
max_unemployment = data['Unemployment_Rate'].max()
data[['CO','NO2','SO2','PM2_5']] = data[['CO','NO2','SO2','PM2_5']].apply(lambda x: x*max_unemployment)

app = Dash()

fig=go.Figure()
fig.add_trace(go.Scatter(x=data["Date"],y=data["CO"],customdata=raw_data['CO'], name="CO (scaled)",hovertemplate='<br>'.join([
            'Scaled Emission: $%{y:.2f}',
            'Month: %{x}',
            'Original Emission: %{customdata}',
        ])))
fig.add_trace(go.Scatter(x=data["Date"],y=data["NO2"],name="NO2 (scaled)"))
fig.add_trace(go.Scatter(x=data["Date"],y=data["SO2"],name="SO2 (scaled)"))
fig.add_trace(go.Scatter(x=data["Date"],y=data["PM2_5"],name="PM2.5 (scaled)"))
fig.add_trace(go.Bar(x=data['Date'],y=data['Unemployment_Rate']))


app.layout = html.Div(children=[
    html.H1(children='Trends between different emissions and unemployment rate, scaled'),
    dcc.Dropdown(
        id='my-dropdown',
        options=[
            {'label': 'Humid Subtropical', 'value': 'subtropical'},
            {'label': 'Mediterran', 'value': 'med'},
            {'label': 'Mid-Lattitude Steppe and Desert', 'value': 'desert'},
            {'label': 'Oceanic Subtropical Highland', 'value': 'highland'},
            {'label': 'Tropical Monsoon', 'value': 'monsoon'},
            {'label': 'Tropical Savanna', 'value': 'sav'},
        ],
        value='subtropical'
    ),
    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])

@app.callback(
Output('example-graph', 'figure'), 
[Input('my-dropdown', 'value')]
)
def update_graph(value):
	raw_data = pd.read_csv("C:/Users/shrey/OneDrive/Documents/Georgia Tech OMSA/MacroXProject/Data/Aggreagted Data/"+value+"_imp_agg.csv",index_col=0)
	raw_data.sort_values("Date", inplace=True)
	raw_data.set_index('Date')
	data = raw_data.copy()
	scaler = MinMaxScaler()
	data[['CO','NO2','SO2','PM2_5']] = scaler.fit_transform(data[['CO','NO2','SO2','PM2_5']])
	max_unemployment = data['Unemployment_Rate'].max()
	data[['CO','NO2','SO2','PM2_5']] = data[['CO','NO2','SO2','PM2_5']].apply(lambda x: x*max_unemployment)
	return{'data': [go.Scatter(x=data["Date"],y=data["CO"],customdata=raw_data['CO'],name="CO (scaled)",hovertemplate='<br>'.join([
            'Scaled Emission: %{y:.2f}',
            'Month: %{x}',
            'Original Emission: %{customdata}',
        ])),
	go.Scatter(x=data["Date"],y=data["NO2"],name="NO2 (scaled)",customdata=raw_data['NO2'],hovertemplate='<br>'.join([
            'Scaled Emission: %{y:.2f}',
            'Month: %{x}',
            'Original Emission: %{customdata}',
        ])),
	go.Scatter(x=data["Date"],y=data["SO2"],name="SO2 (scaled)",customdata=raw_data['SO2'],hovertemplate='<br>'.join([
            'Scaled Emission: %{y:.2f}',
            'Month: %{x}',
            'Original Emission: %{customdata}',
        ])),
	go.Scatter(x=data["Date"],y=data["PM2_5"],name="PM2.5 (scaled)",customdata=raw_data['PM2_5'],hovertemplate='<br>'.join([
            'Scaled Emission: %{y:.2f}',
            'Month: %{x}',
            'Original Emission: %{customdata}',
        ])),
	go.Bar(x=data['Date'],y=data['Unemployment_Rate'],name="Unemployment Rate")]}

if __name__ == '__main__':
    app.run_server(debug=True)