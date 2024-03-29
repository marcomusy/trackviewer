#!/usr/bin/env python3
# https://github.com/marcomusy/trackviewer
# Created on Wed Jun 10 17:45:55 2022
# @author: musy

"""Press:
- arrows to navigate
    left/right to change frame
    up / down  to change track
- drag mouse to rotate the scene in the left panel
- right-click and drag to zoom in and out
- click in right panel to show closest tracks
- 1-9 (on keypad) or + to change volume channel
- l to show track line
- c to show closest ids
- x to jump to the closest track
- t to manually input a track id
- J to join the current track to a specified one
- S to split the current track in half
- W to write the edited track to disk
- o to enable/disable drawing a reference spline
- O to find tracks in the (anti-clock-wise) spline
- p to print current mouse coordinates (in pixels)
- r to reset camera
- q to quit"""

import os
import numpy as np
import pandas
from rich.table import Table
from rich.console import Console
import vedo

version = "1.0"

######################################################
class TrackViewer:
    """Track Viewer"""

    def __init__(self):

        self.cmap = "Greys_r"     # color mapping of the volume slices
        self.frame = 0            # current frame number
        self.track = 0            # current track number
        self.itrack = 0           # track nr selected by slider
        self.nframes = 0          # total number of time frames
        self.ntracks = 0          # total nr of tracks
        self.dataframe = None     # the pandas dataframe
        self.volume = None        # the vedo Volume object
        self.maxvelocity = 10     # a saturation value for velocity coloring
        self.nclosest = 10        # max nr of tracks positions to visualize
        self.skiprows = (1,2,3)   # skip these rows in the csv file
        self.channel = 1          # the tif channel we want to see as slices
        self.nchannels = 3        # total nr of channels in the tif stack
        self.lcolor = 'white'     # color of the labels when clicking
        self.lscale = 6           # size of the labels
        self.fieldname = 'RADIUS' # the variable name to be shown in the terminal table
        self.monitor = 'MEAN_INTENSITY_CH3'  # field name to show in the plot
        self.range = ()           # scalar range of the sox9 expression (automatic)
        self.yrange = (None,None) # y plot range, None=automatic

        self.uniquetracks = []
        self.closer_trackid = None
        self.text2d = None
        self.volumes = [None] * self.nchannels
        self.filename = ""

        self.input_mode = False
        self.input_text2d = None
        self.input_string = ""

        self.draw_mode = False
        self.spline = None
        self.spline_cpoints = []
        self.spline_points = None
        self.spline_tracks = None

        self.camera = dict(
            pos=(1145, -1405, 1200),
            focal_point=(285, 255, 370),
            viewup=(-0.18, 0.36, 0.915),
        )

        vedo.settings.enable_default_mouse_callbacks = False
        vedo.settings.enable_default_keyboard_callbacks = False
        vedo.settings.default_font = "Calco"

        custom_shape = [
            dict(bottomleft=(0  ,0), topright=(0.5,1), bg='white', bg2='lightcyan'), # renderer0 (3d)
            dict(bottomleft=(0.5,0), topright=(1.0,1), bg='white'),            # renderer1 (2d scene)
            dict(bottomleft=(0.29,0.76), topright=(0.498,0.998), bg='k9'),     # renderer2 (2d plot)
        ]
        self.plotter = vedo.Plotter(
            shape=custom_shape,
            sharecam=False,
            title=f"Track Viewer v{version}",
            size=(2200, 1100),
        )

        self._callback1 = self.plotter.add_callback("left click mouse", self._on_left_click)
        self._callback2 = self.plotter.add_callback("key press", self._on_keypress)
        self._callback3 = self.plotter.add_callback("right click mouse", self._on_right_click)
        self.slider1 = None
        self.slider2 = None


    ######################################################
    def load_tracks(self, filename):
        """Load the track data from a cvs file"""
        vedo.printc("Loading track data from", filename, c="y", end="")
        self.dataframe = pandas.read_csv(filename, skip_blank_lines=True, skiprows=self.skiprows)

        self.uniquetracks = np.unique(self.dataframe["TRACK_ID"].to_numpy()).astype(int)
        self.ntracks = len(self.uniquetracks)
        vedo.printc(f"  (found {self.ntracks} tracks)", c="y")

        self.slider2 = self.plotter.at(1).add_slider(
            self._slider_track,
            0,
            self.ntracks - 1,
            value=self.itrack,
            pos=([0.05, 0.94], [0.45, 0.94]),
            title="Track ID",
            show_value=False,
            c="blue3",
        )

    ######################################################
    def load_volume(self, filename=""):
        """Load the 3-channel tif stack"""

        # channels are interleaved in the tif stack so self.nchannels is the step
        ch = (self.channel + 2) % self.nchannels
        if self.volumes[ch] is not None:
            self.volume = self.volumes[ch]
            if len(self.range) == 0:
                self.range = self.volume.scalar_range()
                self.range[1] = self.range[1] * 0.7
            return self.volume

        if filename:
            self.filename = filename
        else:
            filename = self.filename
        vedo.printc(f"Loading volumetric dataset {filename}, channel-{self.channel}", c="y", end='')
        dataset = vedo.Volume(filename)

        arr = dataset.tonumpy()
        self.volume = vedo.Volume(arr[:, :, ch::self.nchannels])
        self.volumes[ch] = self.volume

        dims = self.volume.dimensions()
        self.nframes = int(dims[2])
        vedo.printc(f"  (found {self.nframes} frames)", c="y")

        if len(self.range) == 0:
            self.range = self.volume.scalar_range()
            self.range[1] = self.range[1] * 0.7

        if self.slider1 is None:
            self.slider1 = self.plotter.at(1).add_slider(
                self._slider_time,
                0,
                self.nframes - 1,
                value=self.frame,
                pos=([0.05, 0.06], [0.45, 0.06]),
                title="Frame number",
                c="orange3",
            )

    ######################################################
    def get_points(self, track=None):
        """Get point coords for a track"""
        if track is None:
            track = self.track
        track_df = self.dataframe.loc[self.dataframe["TRACK_ID"] == track]
        if len(track_df) == 0:
            return ()
        line_pts = np.c_[track_df["POSITION_X"], track_df["POSITION_Y"], track_df["FRAME"]]
        return line_pts

    ######################################################
    def get_velocity(self):
        """Get velocity for a track"""
        line_pts = self.get_points()
        line_pts0 = np.array(line_pts[:-1])
        line_pts1 = np.array(line_pts[1:])
        delta = line_pts1 - line_pts0
        delta = [delta[0]] + delta.tolist()
        return vedo.mag(np.array(delta))

    ######################################################
    def get_closest(self, pt=None):
        frame = self.frame
        track = self.track
        df = self.dataframe

        if pt is None:
            track_frame_df = df.loc[(df["FRAME"] == frame) & (df["TRACK_ID"] == track)]
            px, py, pz = (
                track_frame_df["POSITION_X"],
                track_frame_df["POSITION_Y"],
                track_frame_df["FRAME"],
            )
            pt = np.c_[px, py, pz]
            if len(pt) == 0:
                return
            pt = pt[0]

        frame_df = df.loc[df["FRAME"] == frame]
        tx, ty, tz = frame_df["POSITION_X"], frame_df["POSITION_Y"], frame_df["FRAME"]
        trackpts_at_frame = np.c_[tx, ty, tz]

        spotid, trackid = frame_df["ID"], frame_df["TRACK_ID"]
        area, circ, sox9, finfo = (
            frame_df["AREA"],
            frame_df["CIRCULARITY"],
            frame_df[self.monitor],
            frame_df[self.fieldname],
        )
        ids = np.c_[spotid, trackid, area, circ, sox9, finfo]

        vpts = vedo.Points(trackpts_at_frame)
        cids = vpts.closest_point(pt, n=self.nclosest, return_point_id=True)
        self.closer_trackid = trackid.to_numpy()[cids[0]]

        for row in ids[cids]:
            self.plotter.at(0).remove("closeby_trk")
        for row in ids[cids]:
            tpts = self.get_points(row[1])
            if len(tpts):
                closeby_track = vedo.Line(tpts, c="indigo8")
                closeby_track.name = "closeby_trk"
                self.plotter.at(0).add(closeby_track)

        trackpts_at_frame[:, 2] = 0
        cpts = vedo.Points(trackpts_at_frame[cids], c="w")
        labels1 = cpts.labels("id", c=self.lcolor, justify="center", scale=self.lscale).shift(0,0,1)
        labels1.name = "labels1"
        labels0 = labels1.clone(deep=False).z(self.frame + 0.1).pickable(False)
        labels0.name = "labels0"

        self.plotter.at(0).remove("labels0").add(labels0)
        self.plotter.at(1).remove("labels1").add(labels1).render()

        rtable = Table(title_justify="left")
        rtable.add_column(header="Index", style="yellow", no_wrap=True)
        rtable.add_column(header="spotID", style="yellow", no_wrap=True)
        rtable.add_column(header="trackID", style="yellow", no_wrap=True)
        rtable.add_column(header="Area", style="yellow", no_wrap=True)
        rtable.add_column(header="Circularity", style="yellow", no_wrap=True)
        rtable.add_column(header="SOX9", style="yellow", no_wrap=True)
        rtable.add_column(header=self.fieldname, style="yellow", no_wrap=True)
        for i, (a, b, c, d, e, f) in enumerate(ids[cids]):
            try:
                d = vedo.utils.precision(d, 4)
                rtable.add_row(str(i), str(int(a)), str(int(b)), str(int(c)), d, str(int(e)), str(f))
            except ValueError:
                pass
        console = Console()
        console.print(rtable)
        return ids[cids]

    ######################################################
    def update(self):
        """Update visualization"""
        self.track = min(self.track, max(self.uniquetracks))
        self.track = max(self.track, min(self.uniquetracks))
        self.frame = min(self.frame, self.nframes - 1)
        self.frame = max(self.frame, 0)

        slc1 = self.volume.zslice(self.frame).lighting("off").z(-self.frame)
        slc1.cmap(self.cmap, vmin=self.range[0], vmax=self.range[1])

        slc1.name = "slice1"
        slc0 = slc1.clone(deep=False).z(self.frame).pickable(False)
        slc0.name = "slice0"
        self.plotter.at(0).remove("slice0").add(slc0)
        self.plotter.at(1).remove("slice1").add(slc1).render()

        line_pts = self.get_points()
        if len(line_pts) == 0:
            return

        frames = line_pts[:, 2]
        minframe, maxframe = np.min(frames).astype(int), np.max(frames).astype(int)
        trackline = vedo.Line(line_pts, lw=3, c="orange5")
        trackline.name = "track"
        vel = self.get_velocity()
        trackline.cmap("autumn_r", vel, vmin=0, vmax=self.maxvelocity)
        self.plotter.at(0).remove("track", "labels0").add(trackline)

        # for _ in range(self.nclosest):
        self.plotter.remove("closeby_trk")

        self.plotter.at(1).remove("pt2d", "track2d", "labels1")
        pt2d = None
        if minframe <= self.frame <= maxframe:
            res = np.where(frames == self.frame)[0]
            if len(res):  # some frames might be missing
                pt2d = vedo.Point(line_pts[res[0]], c="red6", r=10).z(0)
                pt2d.pickable(False).use_bounds(False)
                pt2d.name = "pt2d"
                self.plotter.add(pt2d)

        # work out the average level of the self.monitor variable over the whole frame
        mframes, mon_levels = [], []
        for fr in range(minframe, maxframe, 5):
            mm = self.dataframe.loc[(self.dataframe["FRAME"] == fr)][self.monitor].mean()
            mframes.append(fr)
            mon_levels.append(mm)

        level = self.dataframe.loc[self.dataframe["TRACK_ID"]==self.track][self.monitor].to_numpy()
        title = self.monitor.replace("_","\_")
        mplot = vedo.pyplot.plot(frames, level, 'o', ylim=self.yrange, title=title, aspect=16/9)
        mplot+= vedo.Line(np.c_[np.array(frames), vel+mplot.ylim[0]-1], c='tomato', lw=2)
        if len(mon_levels) > 2:
            mplot += vedo.Line(np.c_[np.array(mframes), np.array(mon_levels)], c="k4", lw=1)

        if pt2d is not None:  # some frames might be missing
            mplot += vedo.Point([self.frame, level[res][0]], r=9, c="red6")
        self.plotter.at(2).remove("PlotXY").add(mplot).reset_camera(tight=0.05)

        self.text2d.text("Press h for help")
        self.slider1.value = self.frame
        self.slider2.title = f"Track ID {self.track}"
        self.plotter.render()

    ######################################################
    def _slider_time(self, obj, _):
        self.frame = int(self.slider1.value)
        self.update()

    def _slider_track(self, obj, _):
        self.itrack = int(self.slider2.value)
        self.track = self.uniquetracks[self.itrack]
        self.slider2.title = f"Track ID {self.track}"
        self.update()

    ######################################################
    def _update_spline(self, closed=False):
        self.plotter.remove([self.spline, self.spline_points])  # remove old points and spline
        self.spline_points = vedo.Points(self.spline_cpoints).ps(10).c("yellow4")
        self.spline_points.pickable(False)  # avoid picking the same point
        if len(self.spline_cpoints) > 2:
            self.spline = vedo.Spline(self.spline_cpoints, closed=closed).c("yellow5").lw(3)
            self.plotter.add(self.spline_points, self.spline)
        else:
            self.plotter.add(self.spline_points)
        self.plotter.render()

    ######################################################
    def _on_right_click(self, evt):
        if self.draw_mode and evt.actor and len(self.spline_cpoints) > 0:
            self.spline_cpoints.pop()  # pop removes from the list the last pt
            self._update_spline()

    ######################################################
    def _on_left_click(self, evt):
        """Clicking on the right image will dump some info"""
        if not evt.actor:
            return

        if self.draw_mode:
            p = evt.picked3d + [0, 0, 1]
            self.spline_cpoints.append(p)
            self._update_spline()

        else:
            pt = np.array([evt.picked3d[0], evt.picked3d[1], self.frame])
            self.get_closest(pt)

    ######################################################
    def _interactive_keypress(self, key):
        # used to input values from window
        if key == "Return":
            self.input_mode = False
            if self.input_string.isdigit():
                self.track = int(self.input_string)
                line_pts = self.get_points()
                if len(line_pts) == 0:
                    self.input_text2d.text(f"Track {self.track} does not exist!")
                    self.plotter.render()
                    return
                self.frame = int(np.min(line_pts[:, 2]))
            self.input_text2d.text("")
            self.update()

        elif key == "BackSpace":
            if self.input_string:
                self.input_string = self.input_string[:-1]
                self.input_text2d.text("Jump to TrackID: " + self.input_string)
                self.plotter.render()

        elif key.isdigit():
            self.input_string += key
            self.input_text2d.text("Jump to TrackID: " + self.input_string)
            self.plotter.render()

    def _on_keypress(self, evt):
        """Press keys to perform some action"""
        k = evt.keypress
        if not k: return

        # HACK for vedo2022.4.1: needed by vtk9.2 because key is kept lowercase wrt vtk9.0
        # if hasattr(self.plotter, "keyheld") and "Shift_" in self.plotter.keyheld and len(k)==1:
        #     k = k.upper()
        #     self.plotter.keyheld = ''

        # ------------------------------------------------
        if self.input_mode:
            self._interactive_keypress(k)
            return
        # ------------------------------------------------

        if k == "t":
            self.input_mode = True
            self.input_string = ""
            self.input_text2d.text("Jump to TrackID: " + self.input_string)
            self.plotter.render()

        elif k == "Up":
            self.itrack += 1
            if self.itrack >= self.ntracks:
                self.itrack = self.ntracks - 1
                return
            self.track = self.uniquetracks[self.itrack]
            line_pts = self.get_points()
            self.frame = int(np.min(line_pts[:, 2]))
            self.slider2.value = self.itrack

        elif k == "Down":
            self.itrack -= 1
            if self.itrack < 0:
                self.itrack = 0
                return
            self.track = self.uniquetracks[self.itrack]
            line_pts = self.get_points()
            self.frame = int(np.min(line_pts[:, 2]))
            self.slider2.value = self.itrack

        elif k == "Right":
            self.frame += 1

        elif k == "Left":
            self.frame -= 1

        elif k == "l":
            line_pts = self.get_points()
            if line_pts is None:
                return
            trackline2d = vedo.Line(line_pts[:, (0, 1)], c="o6", alpha=0.5).z(0.1)
            trackline2d.name = "track2d"
            self.plotter.at(1).remove("track2d").add(trackline2d).render()
            return

        elif k == "x":
            if self.closer_trackid is None:
                vedo.printc("Please click a point or press c before x", c="r")
                return
            vedo.printc(" -> jumping to track id", self.closer_trackid)
            self.track = self.closer_trackid
            self.closer_trackid = None

        elif k == "c":
            self.get_closest()
            return

        elif "KP_" in k:
            kp = k.replace("KP_", "").replace("End", "1").replace("Down", "2")
            kp = kp.replace("Next", "3").replace("Left", "4").replace("Begin", "5")
            if kp.isdigit():
                self.channel = int(kp)
                self.load_volume()

        elif "plus" in k:
            self.channel = (self.channel + 1) % self.nchannels
            self.load_volume()

        elif k == "r":
            dx, dy, _ = self.volume.dimensions()
            cam = dict(
                pos=(dx/2, dy/2, (dx+dy)/2*3),
                focal_point=(dx/2, dy/2, 0),
                viewup=(0, 1, 0),
            )
            self.plotter.at(1).remove(self.spline)
            self.spline_cpoints = []
            self.spline = None
            self.plotter.at(0).reset_camera()
            self.plotter.at(1).remove("track2d")
            self.plotter.at(1).show(camera=cam)

        elif k == "h":
            if len(self.text2d.text())>20:
                self.text2d.text("Press h for help")
            else:
                self.text2d.text(__doc__)
            self.plotter.render()
            return

        elif k == "J":
            trid = vedo.io.ask(
                f"Insert the TRACKID to be joined to current track {self.track}",
                c='g', invert=True,
            )
            try:
                self.join_tracks(self.track, int(trid))
            except ValueError:
                vedo.printc("Could not join tracks. Skipped.")
            return

        elif k == "S":
            try:
                self.split_track(self.track)
            except ValueError:
                vedo.printc(f"Could not split track {self.track}. Skipped.")
            return

        elif k == "W":
            self.write()
            return

        elif k == "o" and self.draw_mode is False:
            self.draw_mode = True
            vedo.printc("Spline drawing mode is now enabled", c="y", invert=True)
            self.plotter.at(0).remove("closeby_trk")
            self.plotter.at(1).remove(self.spline).render()
            self.spline_cpoints = []
            self.spline = None
            return

        elif k == "o":
            self._update_spline(closed=True)
            self.draw_mode = False
            self.plotter.remove(self.spline_points).render()
            vedo.printc("Spline drawing mode disabled", c="y", invert=True)
            return

        elif k == "O":
            if not self.spline:
                vedo.printc("There is no spline! Press o to draw it.", c="r", invert=True)
                return
            if self.draw_mode:
                vedo.printc("Press again o to close the line.", c="r", invert=True)
                return

            if not self.spline_tracks:
                tracks = []
                trackIDs = []
                vedo.printc("Selecting tracks..", c="y")
                ltracks = set(self.dataframe["TRACK_ID"].to_numpy())
                pb = vedo.ProgressBar(0, len(ltracks), c="y")
                for t in ltracks:
                    pb.print()
                    pts = self.get_points(t)
                    if len(pts) == 0:  # some TRACK_ID can be empty
                        continue
                    tracks.append(pts)
                    trackIDs += [t] * len(pts)
                # tracks = np.array(tracks)
                self.spline_tracks = vedo.Lines(tracks)
                self.spline_tracks.pointdata["trackID"] = np.array(trackIDs, dtype=int)

            vtracks = self.spline_tracks.clone().cut_with_line(self.spline, invert=False)
            rtracks = list(set(vtracks.pointdata["trackID"]))
            for i in range(1, 100):
                fn = f"splinetracks_0{i}.txt" if i < 10 else f"splinetracks_{i}.txt"
                if not os.path.isfile(fn):
                    with open(fn, "w") as f:
                        for tr in rtracks:
                            f.write(str(tr) + "\n")
                        break

            vedo.printc("Visualising tracks..", c="y")
            for t in rtracks:
                tpts = self.get_points(t)
                if len(tpts):
                    closeby_track = vedo.Line(tpts, c="indigo8")
                    closeby_track.name = "closeby_trk"
                    self.plotter.at(0).add(closeby_track)
            tube = self.spline.extrude(vtracks.zbounds(1)).lw(0).c("yellow5").alpha(0.25)
            tube.name = "closeby_trk"
            self.plotter.add(tube).render()

            vedo.printc(f"Tracks in current spline written to {fn}", c="g", invert=1)
            self.plotter.screenshot(fn.replace(".txt", ".png"))
            return

        elif k == "p" and evt.picked3d is not None:
            vedo.printc(
                f"Mouse position: [{int(np.round(evt.picked3d[0]))},",
                f"{int(np.round(evt.picked3d[1]))}]",
                f"frame={self.frame}",
                c="c",
                invert=True,
            )
            return

        elif k == "S":
            self.plotter.screenshot()
            return

        elif k == "q":
            self.plotter.break_interaction()
            return

        self.update()

    ######################################################
    def join_tracks(self, track1, track2):
        """Join two tracks. The ID of the second input is overwritten by the first"""
        z0 = self.get_points(track1)[:, 2]
        z1 = self.get_points(track2)[:, 2]

        # Sanity checks
        tt = f" is overlapping with current track {track1}. Skip."
        if z0[0] < z1[0] < z0[-1]:
            vedo.printc(f"ERROR: start frame of track {track2} {tt}", c="r", invert=True)
            return
        if z0[0] < z1[-1] < z0[-1]:
            vedo.printc(f"ERROR: end frame of track {track2} {tt}", c="r", invert=True)
            return
        if z1[0] <= z0[0] and z1[-1] >= z0[-1]:
            vedo.printc(f"ERROR: track you want to join {tt}", c="r", invert=True)
            return

        df = self.dataframe
        df["TRACK_ID"] = np.where(df["TRACK_ID"] == track2, track1, df["TRACK_ID"])
        vedo.printc(f"..joined tracks IDs {track1} and {track2} to ID {track1}", c='g', invert=True)
        self.uniquetracks = np.unique(df["TRACK_ID"].to_numpy()).astype(int)
        self.ntracks = len(self.uniquetracks)
        self.slider2.range = [0, self.ntracks - 1]
        self.update()

    ######################################################
    def split_track(self, trid, frame=None):
        """Split a track starting from current/specified frame and return a new ID"""
        if frame is None:
            frame = self.frame
        df = self.dataframe
        mask = (df["TRACK_ID"] == trid) & (df["FRAME"] >= frame)
        newid = df["TRACK_ID"].max() + 1
        df["TRACK_ID"] = np.where(mask, newid, df["TRACK_ID"])
        vedo.printc("..created new track with ID", newid, c="g", invert=True)
        self.uniquetracks = np.unique(df["TRACK_ID"].to_numpy()).astype(int)
        self.ntracks = len(self.uniquetracks)
        self.slider2.range = [0, self.ntracks - 1]
        self.update()
        return newid

    ######################################################
    def write(self, filename="new_spots.csv"):
        """Write to file the current dataframe for the (edited) tracks"""
        vedo.printc(f"Writing tracks to {filename} (this will take time)...", end="", invert=True)
        self.dataframe.to_csv(filename)
        vedo.printc(" done!", invert=True)

    ######################################################
    def start(self, interactive=True):
        """Initialize and start the interactive application"""
        slc = self.volume.zslice(self.frame).lighting("off").z(-self.frame)
        slc.cmap(self.cmap, vmin=self.range[0], vmax=self.range[1])

        self.text2d = vedo.Text2D("Press h for help", font="Calco", bg="yellow9", alpha=1)
        self.input_text2d = vedo.Text2D(font="Calco", c='w', bg="blue3", alpha=1, pos='bottom-left', s=1.3)

        axes = vedo.Axes(
            self.volume, xtitle="x /pixel", ytitle="y /pixel", ztitle="frame nr.",
            xtitle_backface_color='t', ytitle_backface_color='t', ztitle_backface_color='t',
        )

        self.plotter.at(0).show(axes, self.text2d, self.input_text2d, camera=self.camera, bg2="light cyan")
        self.plotter.at(1).show(slc, resetcam=True, zoom=1.1)
        self.update()
        if interactive:
            self.plotter.interactive()

