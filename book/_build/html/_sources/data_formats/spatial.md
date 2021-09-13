# Spatial Files
Example file: `synthetic_data/prefectures.geojson` <br>
```{note}
Spatial data is loaded with geopandas, and can be stored as a shapefile or geojson. 
```
| region   | geometry          |   |
|----------|-------------------|---|
| Region A | Multipolygon(...) |   |
| Region B | Multipolygon(...) |   |
| Region C | Multipolygon(...) |   |
| Region D | Multipolygon(...) |   |  


## Fields <br>
***region**: _string_ <br>
Unique ID for each polygon in the shapefile. Must be unique. 


**amount**: _float_ <br>
Amount of the recharge (in any currency)

***geometry**: _polygon_ <br>
Shapely polygon, as loaded from a shapefile or geojson - usually in WKT format


```{note}
Columns without a preceding asterisk '*' are optional
```
