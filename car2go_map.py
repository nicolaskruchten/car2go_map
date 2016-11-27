# This script by [Nicolas Kruchten](http://nicolas.kruchten.com/) was used to
# generate a map for a [CBC.ca article](http://ici.radio-canada.ca/nouvelles/special/2016/11/montreal-car2go-covoiturage-vignettes-stationnement/)
# showing the spatio-temporal distribution of available
# [Car2Go](https://www.car2go.com/CA/en/montreal/) vehicles in Montreal over a
# 30-day period during the summer of 2016.
# <small>[more info &raquo;](http://nicolas.kruchten.com/)</small>

# [`pandas`](http://pandas.pydata.org/) and
# [`scikit-learn`](http://scikit-learn.org/stable/) are used to manipulate the
# data,  [`folium`](https://github.com/python-visualization/folium) to make the
# map, and [`vincent`](https://vincent.readthedocs.io/en/latest/) to make the
# popup charts.

import pandas, math
from sklearn.cluster import MiniBatchKMeans
from folium import Map, CircleMarker, Vega, Popup
from vincent import Bar

# The dataset contains approximately 3 million `(lat, lon, hod)` tuples
# where `hod` stands for 'hour of day'.

df = pandas.read_csv("lat_lon_hod.csv.gz")

# `carh` stands for 'car-hours', and since the dataset was made by interrogating
# the Car2Go API every 5 minutes, each row represents an observation which, on
# average, means that a car was present for one twelfth of an hour during the
# given hour of day at the given latitude and longitude.

samples_per_hour = 12
df["carh"] = 1.0/samples_per_hour

# The map is divided into 250 zones, each containing a comparable numbers of
# observations, using
# [K-means clustering](https://en.wikipedia.org/wiki/K-means_clustering).

df["zone"] = MiniBatchKMeans(250).fit_predict(df[["lat", "lon"]].values)

# A dataset is generated with one row per zone containing the latitude and
# longitude of the zone centroid, and the total number of car-hours observed.

zones = df.groupby("zone").agg(dict(carh='sum', lon='mean', lat='mean'))

# With the data manipulation complete, a map is initialized, centered on the
# centroid of all the zone centroids.

map = Map(zoom_start=12, tiles="cartodbpositron",
             location=list(zones[["lat","lon"]].mean().values))

# For each zone...
for i, zone in zones.iterrows():

    # ...a dataset is created containing the average number of car-hours/day by
    # hour of day. (1 car-hour/day at a given hour of day in a given zone
    # represents either 1 car available in the zone for the entire hour, or 3
    # cars for 20 minutes each, or 1 car for 20 minutes and 1 for 40 minutes,
    # and so on.)

    num_days = 30
    carh_by_hod = df.query("zone==@i").groupby("hod").carh.sum()/num_days

    # ...a bar chart is created with this dataset, which shows the daily pattern
    # of car availabilities. This is done by first generating a JSON description
    # of a bar-chart in the [Vega grammar](https://vega.github.io/vega/), and
    # then rendering it to a graphic.

    vega = Bar(carh_by_hod, width=450,
               height=150).axis_titles(x='Hour of Day', y='Cars Available')
    chart = Vega(vega.to_json(), width=vega.width+50, height=vega.height+50)

    # ... and a circle is placed on the map:
    map.add_child( CircleMarker(

        # &bullet; circles are located on the zone centroids.
        location = [zone["lat"], zone["lon"]],

        # &bullet; circles are scaled such that the area is proportional to the
        # total number of car-hours observed in the zone.
        radius = int(6*math.sqrt(zone["carh"])),

        # &bullet; circles are coloured according to the hour of day with peak
        # car availability: blues representing night-time, yellows representing
        # day-time and reds representing evenings.
        fill_opacity = 0.8, color=None,
        fill_color = ('#274cc9','#274cc9','#274cc9','#274cc9',
            '#274cc9','#3959bf','#647aa6','#909b8c','#bcbc73',
            '#e8dd5a','#f1e455','#f1e455','#f1e455','#f1e455',
            '#f0df56','#ecc45a','#e7a95f','#e28e63','#de7467',
            '#c46576','#9d5f8a','#76599f','#4e52b4','#274cc9'
            )[carh_by_hod.idxmax()],

        # &bullet; clicking on the circles pops up the daily-pattern chart.
        popup = Popup(max_width=chart.width[0]).add_child(chart)
    ) )

# The map is then saved to disk as a [self-contained HTML file](map.html)
# (except for the map tiles, of course) in that the Javascript required to place
# the circles on the map and generate the popup charts is inlined. This is very
# convenient for collaborating with colleagues without sending a large number of
# files or requiring them to run this script.
map.save('map.html')
