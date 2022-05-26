#!/usr/bin/env python3
#
import vedo

csv_path = vedo.io.download("https://vedo.embl.es/examples/data/sox9_tracks_test.csv")
tif_path = vedo.io.download("https://vedo.embl.es/examples/data/sox9_labels_test.tif")
# csv_path = "data/all_spots_reduced.csv"
# tif_path = "data/sox9_labels_test.tif"


from trackviewer import TrackViewer

# Create the viewer
tv = TrackViewer()

# Customize the viewer:
# tv.cmap='Greys_r'
# tv.nclosest = 10

# Load the data
tv.loadTracks(csv_path)
tv.loadVolume(tif_path)

# Start the application
tv.start()
