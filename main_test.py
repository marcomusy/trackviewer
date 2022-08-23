#!/usr/bin/env python3
#
from trackviewer import TrackViewer

# Data ##################################################################################
csv_path = "data/72h_spots.csv"
tif_path = "data/72h_masks_sox9_mem.tif"
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
tv.loadTracks(csv_path)
tv.loadVolume(tif_path)

tv.track = 1571
tv.frame = 15

# Start the application:
tv.start(interactive=False)

# Programmatic usage:
# newid = tv.splitTrack(13, frame=15)
# tv.joinTracks(13, newid)

# tv.splitTrack(13834, frame=404)
# tv.joinTracks(13834, 15570) # this will give error

# tv.write("test.csv")

tv.plotter.interactive().close()
