#!/usr/bin/env python3
#
from trackviewer import TrackViewer

# import vedo
# csv_path = vedo.io.download("https://vedo.embl.es/examples/data/sox9_tracks_test.csv")
# tif_path = vedo.io.download("https://vedo.embl.es/examples/data/sox9_labels_test.tif")

csv_path = "data/all_spots.csv"
tif_path = "data/Composite_downsized_sox9_labels_membrane.tif"

# csv_path = "data/72h_spots.csv"
# tif_path = "data/72h_masks_sox9_mem.tif"


# Create the viewer
tv = TrackViewer()
tv.channel = 2

# Customize the viewer:
# tv.cmap='Greys_r'
# tv.nclosest = 10

# Load the data
tv.loadTracks(csv_path)
tv.loadVolume(tif_path)

# Start the application
tv.start()
