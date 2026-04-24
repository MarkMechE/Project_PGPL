# PGL GIS Dashboard (Busan EDC Map)
# pip install dash dash-leaflet pandas

import dash
import dash_leaflet as dl
from dash import html
import pandas as pd

df = pd.read_csv('data/processed.csv')  # From processor
alerts = df[df['sev'] >= 0.6].to_dict('records')

app = dash.Dash()
app.layout = html.Div([
    html.H1("PGL Busan Map"),
    dl.Map(center=[35.1796, 129.0756], zoom=17, children=[
        dl.GeoJSON(data={"type": "FeatureCollection", "features": [
            {"type": "Point", "geometry": {"coordinates": [a['lon'], a['lat']]},
             "properties": {"style": {"color": "red" if a['sev']>=0.8 else "orange"}}}
            for a in alerts
        ]})
    ])
])

app.run_server(debug=True)