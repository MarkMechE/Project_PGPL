# PGL GIS Dashboard (Busan EDC Map)
# pip install dash dash-leaflet pandas

import dash
from dash import html, dcc
import dash_leaflet as dl
import pandas as pd

# Load Processed Data
df = pd.read_csv('data/processed_real.csv')  # From processor
alerts = df[df['metric'] > 30].to_dict('records')  # P1/P2 only

app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("PGL Busan Eco-Delta Leak Map"),
    html.P(f"Active Alerts: {len(alerts)} | RMSE: 5.53m"),
    dl.Map(style={'height': '600px'}, 
           center=[35.1796, 129.0756], zoom=17,
           children=[dl.GeoJSON(id="alerts", data={
               "type": "FeatureCollection",
               "features": [{"type": "Point", "geometry": {"coordinates": [a["lon"], a["lat"]]},
                            "properties": {"p_level": a["p_level"], "metric": a["metric"]}}
                           for a in alerts]
           })]),
    dcc.Interval(id='interval', interval=5000, n_intervals=0)  # Refresh
])

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)