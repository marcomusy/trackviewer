#!/usr/bin/env python3
#

from trackviewer import TrackViewer

tv = TrackViewer()
tv.cmap='Greens_r'
tv.loadTracks("data/all_spots_reduced.csv")
tv.loadVolume("data/Composite_downsized_sox9_labels_membrane.tif")
tv.start()
