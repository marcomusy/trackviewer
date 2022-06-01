"""Press:
- arrows to navigate
    left/right to change frame
    up / down  to change track
- drag mouse to rotate the scene in the left panel
- right-click and drag to zoom in and out
- left-click in right panel to show closest tracks
- l to show track line
- c to show closest ids
- x to jump to the closest track
- t to manually input a track id in terminal
- r to reset camera
- q to quit"""
import numpy as np
import pandas
from rich.table import Table
from rich.console import Console
import vedo

version = 0.4

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
        self.channel = 2          # the tif channel we want to see as slices
        self.nchannels = 3        # total nr of channels in the tif stack
        self.lcolor = 'white'     # color of the labels when clicking
        self.lscale = 6           # size of the labels
        self.fieldname = 'RADIUS' # the variable name to be shown in the terminal table
        self.sox9name = 'MEAN_INTENSITY_CH1'  # field name where the sox9 info is held
        self.range = ()           # scalar range of the sox9 expression (automatic)
        self.yrange = (None,None) # y plot range, None=automatic

        self.uniquetracks = []
        self.closer_trackid = None
        self.text2d = None

        self.camera = dict(
            pos=(1147, -1405, 1198),
            focalPoint=(284.3, 254.2, 366.9),
            viewup=(-0.1758, 0.3662, 0.9138),
        )

        vedo.settings.enableDefaultMouseCallbacks = False
        vedo.settings.enableDefaultKeyboardCallbacks = False

        custom_shape = [
            dict(bottomleft=(0  ,0), topright=(0.5,1), bg='white', bg2='lightcyan'), # renderer0 (3d)
            dict(bottomleft=(0.5,0), topright=(1.0,1), bg='white'),             # renderer1 (2d scene)
            dict(bottomleft=(0.29,0.76), topright=(0.498,0.998), bg='k9'),      # renderer2 (plot)
        ]
        self.plotter = vedo.Plotter(
            shape=custom_shape, sharecam=False, title=f"Track Viewer v{version}", size=(2200,1100),
        )

        self._callback1 = self.plotter.addCallback("click mouse", self.on_click)
        self._callback2 = self.plotter.addCallback("key press", self.on_keypress)
        self._slider1 = None
        self._slider2 = None


    ######################################################
    def loadTracks(self, filename):
        """Load the track data from a cvs file"""
        vedo.printc("Loading track data from", filename, c="y", end='')
        self.dataframe = pandas.read_csv(filename, skip_blank_lines=True, skiprows=self.skiprows)

        self.uniquetracks = np.unique(self.dataframe["TRACK_ID"].to_numpy()).astype(int)
        self.ntracks = len(self.uniquetracks)
        vedo.printc(f"  (found {self.ntracks} tracks)", c="y")

        self._slider2 = self.plotter.at(1).addSlider2D(
            self._slider_track,
            0,  self.ntracks - 1,
            value=self.itrack,
            pos=([0.05, 0.94], [0.45, 0.94]),
            title="track id",
            showValue=False,
            c='blue3',
        )

    ######################################################
    def loadVolume(self, filename):
        """Load the 3-channel tif stack"""
        vedo.printc("Loading volumetric dataset", filename, c="y", end='')

        dataset = vedo.Volume(filename)
        arr = dataset.tonumpy(transpose=False)
        # channels are interleaved in the tif stack so self.nchannels is the step:
        self.volume = vedo.Volume(arr[self.channel::self.nchannels])

        dims = self.volume.dimensions()
        self.nframes = int(dims[2])
        vedo.printc(f"  (found {self.nframes} frames)", c="y")

        if len(self.range) == 0:
            self.range = self.volume.scalarRange()
            self.range[1] = self.range[1]*0.8

        self._slider1 = self.plotter.at(1).addSlider2D(
            self._slider_time,
            0, self.nframes - 1,
            value=self.frame,
            pos=([0.05, 0.06], [0.45, 0.06]),
            title="frame number",
            c='orange3',
        )

    ######################################################
    def getpoints(self):
        """Get point coords for a track"""
        track_df = self.dataframe.loc[self.dataframe["TRACK_ID"] == self.track]
        if len(track_df) == 0:
            return ()
        line_pts = np.c_[track_df["POSITION_X"], track_df["POSITION_Y"], track_df["FRAME"]]
        return line_pts

    ######################################################
    def getvelocity(self):
        """Get velocity for a track"""
        line_pts = self.getpoints()
        line_pts0 = np.array(line_pts[:-1])
        line_pts1 = np.array(line_pts[1:])
        delta = line_pts1 - line_pts0
        delta = [delta[0]] + delta.tolist()
        return vedo.mag(np.array(delta))

    ######################################################
    def getclosest(self, pt=None):
        frame = self.frame
        track = self.track
        df = self.dataframe

        if pt is None:
            track_frame_df = df.loc[(df["FRAME"]==frame) & (df["TRACK_ID"]==track)]
            px,py,pz = track_frame_df["POSITION_X"], track_frame_df["POSITION_Y"], track_frame_df["FRAME"]
            pt = np.c_[px,py,pz]
            if len(pt) == 0:
                return
            pt = pt[0]

        # this selects the single frame and the +1 is a trick that removes the NaNs
        frame_df = df.loc[(df["FRAME"] == frame) & df["TRACK_ID"]+1]
        tx,ty,tz = frame_df["POSITION_X"], frame_df["POSITION_Y"], frame_df["FRAME"]
        trackpts_at_frame = np.c_[tx,ty,tz]

        spotid, trackid = frame_df["ID"], frame_df['TRACK_ID']
        area, circ, sox9, finfo = (
            frame_df["AREA"],
            frame_df['CIRCULARITY'],
            frame_df[self.sox9name],
            frame_df[self.fieldname],
        )
        ids = np.c_[spotid, trackid, area, circ, sox9, finfo]

        vpts = vedo.Points(trackpts_at_frame)
        cids = vpts.closestPoint(pt, N=self.nclosest, returnPointId=True)
        self.closer_trackid = trackid.to_numpy()[cids[0]]

        trackpts_at_frame[:,2] = 0
        cpts = vedo.Points(trackpts_at_frame[cids], c='w')
        labels = cpts.labels("id", c=self.lcolor, justify='center', scale=self.lscale).shift(0,0,1)
        labels.name = "closest_info"
        self.plotter.at(1).remove("closest_info").add(labels)

        rtable = Table(title_justify='left')
        rtable.add_column(header='Index', style="yellow", no_wrap=True)
        rtable.add_column(header='spotID', style="yellow", no_wrap=True)
        rtable.add_column(header='trackID', style="yellow", no_wrap=True)
        rtable.add_column(header='Area', style="yellow", no_wrap=True)
        rtable.add_column(header='Circularity', style="yellow", no_wrap=True)
        rtable.add_column(header='SOX9', style="yellow", no_wrap=True)
        rtable.add_column(header=self.fieldname, style="yellow", no_wrap=True)
        for i, (a,b,c,d,e,f) in enumerate(ids[cids]):
            d = vedo.utils.precision(d, 4)
            rtable.add_row(str(i), str(int(a)), str(int(b)), str(int(c)), d, str(int(e)), str(f))
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

        slc1 = self.volume.zSlice(self.frame).lighting("off").z(-self.frame)
        slc1.cmap(self.cmap, vmin=self.range[0], vmax=self.range[1])

        slc1.name = "slice1"
        slc0 = slc1.clone(transformed=1).z(self.frame).pickable(False)
        slc0.name = "slice0"
        self.plotter.at(0).remove("slice0").add(slc0, render=False)
        self.plotter.at(1).remove("slice1").add(slc1, render=False)

        line_pts = self.getpoints()
        if len(line_pts)==0:
            return
        frames = line_pts[:, 2]
        minframe, maxframe = np.min(frames).astype(int), np.max(frames).astype(int)
        trackline = vedo.Line(line_pts, lw=3, c="orange5")
        trackline.name = "track"
        vel = self.getvelocity()
        trackline.cmap('autumn_r', vel, vmin=0, vmax=self.maxvelocity)
        self.plotter.at(0).remove("track").add(trackline, render=False)

        self.plotter.at(1).remove("pt2d", "track2d", "closest_info")
        pt2d = None
        if minframe <= self.frame <= maxframe:
            res = np.where(frames == self.frame)[0]
            if len(res):  # some frames might be missing
                pt2d = vedo.Point(line_pts[res[0]], c="red6", r=10).z(0)
                pt2d.pickable(False).useBounds(False)
                pt2d.name = "pt2d"
                self.plotter.add(pt2d, render=False)

        self._slider1.GetRepresentation().SetValue(self.frame)
        self._slider2.GetRepresentation().SetTitleText(f"track id {self.track}")

        sox9level = self.dataframe.loc[self.dataframe["TRACK_ID"]==self.track][self.sox9name].to_numpy()
        title = self.sox9name.replace("_","-")
        sox9plot = vedo.pyplot.plot(frames, sox9level, 'o', ylim=self.yrange, title=title, aspect=16/9)
        sox9plot+= vedo.Line(np.c_[np.array(frames), vel+sox9plot.ylim[0]-1], c='tomato', lw=2)

        if pt2d is not None:  # some frames might be missing
            sox9plot += vedo.Point([self.frame, sox9level[res]], r=9, c='red6')
        self.plotter.at(2).remove("PlotXY").add(sox9plot, render=False).resetCamera(tight=0.05)

        self.text2d.text("Press h for help")
        self.plotter.render()

    ######################################################
    def _slider_time(self, obj, _):
        self.frame = int(obj.GetRepresentation().GetValue())
        self.update()

    def _slider_track(self, obj, _):
        self.itrack = int(obj.GetRepresentation().GetValue())
        self.track = self.uniquetracks[self.itrack]
        obj.GetRepresentation().SetTitleText(f"track id {self.track}")
        self.update()

    ######################################################
    def on_click(self, evt):
        """Clicking on the right image will dump some info"""
        if not evt.actor:
            return
        pt = np.array([evt.picked3d[0], evt.picked3d[1], self.frame])
        self.getclosest(pt)

    ######################################################
    def on_keypress(self, evt):
        """Press keys to perform some action"""
        if evt.keyPressed == "t":
            try:
                self.track = int(vedo.io.ask("Input track:", c="y"))
                line_pts = self.getpoints()
                if len(line_pts)==0:
                    vedo.printc("no points for this track!", c='r')
                    return
                self.frame = int(np.min(line_pts[:, 2]))
            except ValueError:
                pass

        elif evt.keyPressed == "Up":
            self.itrack += 1
            self.track = self.uniquetracks[self.itrack]
            line_pts = self.getpoints()
            self.frame = int(np.min(line_pts[:, 2]))
            self._slider2.GetRepresentation().SetValue(self.itrack)

        elif evt.keyPressed == "Down":
            self.itrack -= 1
            self.track = self.uniquetracks[self.itrack]
            line_pts = self.getpoints()
            self.frame = int(np.min(line_pts[:, 2]))
            self._slider2.GetRepresentation().SetValue(self.itrack)

        elif evt.keyPressed == "Right":
            self.frame += 1

        elif evt.keyPressed == "Left":
            self.frame -= 1

        elif evt.keyPressed == "l":
            line_pts = self.getpoints()
            if line_pts is None:
                return
            trackline2d = vedo.Line(line_pts[:, (0,1)], c="o6", alpha=0.5).z(0.1)
            trackline2d.name = "track2d"
            self.plotter.at(1).remove("track2d").add(trackline2d)
            return

        elif evt.keyPressed == "x":
            if self.closer_trackid is None:
                vedo.printc("Please click a point or press c before x", c='r')
                return
            vedo.printc(" -> jumping to track id", self.closer_trackid)
            self.track = self.closer_trackid
            self.closer_trackid = None

        elif evt.keyPressed == "c":
            self.getclosest()
            return

        elif evt.keyPressed == "r":
            dx, dy, _ = self.volume.dimensions()
            cam = dict(
                pos=(dx/2, dy/2, (dx+dy)/2*3),
                focalPoint=(dx/2, dy/2, 0),
                viewup=(0, 1, 0),
            )
            self.plotter.at(1).show(camera=cam)

        elif evt.keyPressed == "h":
            self.text2d.text(__doc__)
            self.plotter.render()
            return

        elif evt.keyPressed == "q":
            self.plotter.close()
            return
        self.update()

    ######################################################
    def start(self):

        slc = self.volume.zSlice(self.frame).lighting("off").z(-self.frame)
        slc.cmap(self.cmap, vmin=self.range[0], vmax=self.range[1])

        self.text2d = vedo.Text2D("Press h for help", font="Calco", bg="yellow9", alpha=1)

        axes = vedo.Axes(self.volume, xtitle="x /pixel", ytitle="y /pixel", ztitle="frame nr.")

        self.plotter.at(0).show(axes, self.text2d, camera=self.camera, bg2='light cyan')
        self.plotter.at(1).show(slc, resetcam=True, zoom=1.1)
        self.update()
        self.plotter.interactive().close()
