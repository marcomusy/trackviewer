#!/usr/bin/env python3
#
from vedo.io import download
from trackviewer import TrackViewer

# Data ##################################################################################
csv_path = download("https://vedo.embl.es/examples/data/sox9_tracks_test.csv.gz")
tif_path = download("https://vedo.embl.es/examples/data/sox9_labels_test.tif")
########################################################################################

# Create the viewer
tv = TrackViewer()
tv.nchannels = 3

# Customize the viewer
tv.channel = 2
tv.nclosest = 5
tv.monitor = "MEAN_INTENSITY_CH2"
tv.yrange = [90, 130]

# Load the data
tv.load_tracks(csv_path)
tv.load_volume(tif_path)

# Start the application:
tv.start(interactive=False)

# Programmatic usage:
# newid = tv.split_track(13, frame=15)
# tv.join_tracks(13, newid)

# tv.split_track(13834, frame=404)
# tv.join_tracks(13834, 15570) # this will give error

# tv.write("test.csv")

tv.plotter.interactive().close()
