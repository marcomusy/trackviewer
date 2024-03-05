#!/usr/bin/env python3
#
from trackviewer import TrackViewer

# Data ##################################################################################
csv_path = "data/72h_spots.csv"
tif_path = "data/72h_masks_sox9_mem.tif"
########################################################################################
# csv_path = "/g/sharpe/scratch/ForMarco_From_Xavi/trackviewer_dataset/all_spots.csv"
# tif_path = "/g/sharpe/scratch/ForMarco_From_Xavi/trackviewer_dataset/72h_masks_sox9_mem.tif"
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

tv.track = 1571
tv.frame = 15

# Start the application but don't hold it:
tv.start(interactive=False)

# Programmatic usage:
# newid = tv.split_track(13, frame=15)
# tv.join_tracks(13, newid)
#
# tv.split_track(13834, frame=404)
# tv.join_tracks(13834, 15570)  # this will give error
# tv.write("test.csv")

# hold it here
tv.plotter.interactive().close()
