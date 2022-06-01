#!/usr/bin/env python3
#
from trackviewer import TrackViewer

# Data ##################################################################################
from vedo.io import download
csv_path = download("https://vedo.embl.es/examples/data/sox9_tracks_test.csv.gz")
tif_path = download("https://vedo.embl.es/examples/data/sox9_labels_test.tif")

# csv_path = "data/72h_spots.csv"
# tif_path = "data/72h_masks_sox9_mem.tif"
########################################################################################

# Create the viewer
tv = TrackViewer()

# Customize the viewer
tv.channel = 2
tv.sox9name = "MEAN_INTENSITY_CH3"
tv.nclosest = 10
tv.lscale = 6   # size of labels

# Load the data
tv.loadTracks(csv_path)
tv.loadVolume(tif_path)

# Start the application
tv.start()
