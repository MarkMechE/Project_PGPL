import folium

# Create a map centered on Incheon
m = folium.Map(location=[37.4563, 126.7052], zoom_start=12)

# Add a marker for the detected leak
folium.Marker(
    [37.4563, 126.7052], 
    popup="Detected Leak (Error < 1m)",
    icon=folium.Icon(color='red', icon='info-sign')
).add_to(m)

m.save('outputs/leak_map.html')
print("✅ Interactive map generated: outputs/leak_map.html")