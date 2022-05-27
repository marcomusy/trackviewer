import numpy as np
import pandas
import vedo
from rich.table import Table
from rich.console import Console

_version = 0.2

######################################################
class TrackViewer:
    """Track viewer"""

    def __init__(self):

        self.cmap = "Greys_r"
        self.frame = 0  # init values
        self.track = 0
        self.itrack = 0
        self.nframes = 0
        self.ntracks = 0
        self.dataframe = None
        self.volume = None
        self.maxvelocity = 20
        self.nclosest = 10
        self.skiprows = (1,2,3)
        self.channel = 2
        self.lcolor = 'white'
        self.lscale = 4
        self.fieldname = 'RADIUS'
        self.sox9name = 'MEAN_INTENSITY_CH1'
        self.rng = ()
        self.uniquetracks = []
        self.closer_trackid = None

        self.camera = dict(
            pos=(1147, -1405, 1198),
            focalPoint=(284.3, 254.2, 366.9),
            viewup=(-0.1758, 0.3662, 0.9138),
        )

        self.info = (
            "Press:\n"
            "- arrows to navigate\n"
            "- t to input track\n"
            "- r to reset camera\n"
            "- l to show track line\n"
            "- c to show closest ids\n"
            "- x to jump to track\n"
            "- q to quit"
        )
        self._slider1 = None
        self._slider2 = None
        self._cornerplot = None

        vedo.settings.enableDefaultMouseCallbacks = False
        vedo.settings.enableDefaultKeyboardCallbacks = False

        self.plt = vedo.Plotter(N=2, sharecam=False, title=f"Track Viewer v{_version}", size=(2200,1100))
        self._callback1 = self.plt.addCallback("click mouse", self.on_click)
        self._callback2 = self.plt.addCallback("key press", self.on_keypress)


    ######################################################
    def loadTracks(self, filename):
        """Load the track data from a cvs file"""
        vedo.printc("Loading track data from", filename, c="y", end='')
        self.dataframe = pandas.read_csv(filename, skip_blank_lines=True, skiprows=self.skiprows)

        self.uniquetracks = np.unique(self.dataframe["TRACK_ID"].to_numpy())
        self.ntracks = len(self.uniquetracks)
        vedo.printc(f"  (found {self.ntracks} tracks)", c="y")

        self._slider2 = self.plt.at(1).addSlider2D(
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
        self.volume = vedo.Volume(arr[self.channel::3])

        dims = self.volume.dimensions()
        self.nframes = int(dims[2])
        vedo.printc(f"  (found {self.nframes} frames)", c="y")

        if len(self.rng) == 0:
            self.rng = self.volume.scalarRange()
            self.rng[1] = self.rng[1]*0.8

        self._slider1 = self.plt.at(1).addSlider2D(
            self._slider_time,
            0, self.nframes - 1,
            value=self.frame,
            pos=([0.05, 0.06], [0.45, 0.06]),
            title="frame nr. (time)",
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
        # print("point track",pt ," at frame", frame)

        # this selects the single frame and the +1 is a trick that removes the NaNs
        frame_df = df.loc[(df["FRAME"] == frame) & df["TRACK_ID"]+1]
        tx,ty,tz = frame_df["POSITION_X"], frame_df["POSITION_Y"], frame_df["FRAME"]
        trackpts_at_frame = np.c_[tx,ty,tz]
        # print("frame_df", frame_df)

        spotid, trackid = frame_df["ID"], frame_df['TRACK_ID']
        area, circ, sox9, finfo = (
            frame_df["AREA"],
            frame_df['CIRCULARITY'],
            frame_df[self.sox9name],
            frame_df[self.fieldname],
        )
        ids = np.c_[spotid, trackid, area, circ, sox9, finfo]
        # print("spotid, trackid", ids)

        vpts = vedo.Points(trackpts_at_frame)
        cids = vpts.closestPoint(pt, N=self.nclosest, returnPointId=True)
        self.closer_trackid = trackid.to_numpy()[cids[0]]

        # print("closestPoint ids", cids)
        # print("closestPoint spotid, trackid", ids[cids])

        trackpts_at_frame[:,2] = 0
        cpts = vedo.Points(trackpts_at_frame[cids], c='w')
        labels = cpts.labels("id", c=self.lcolor, justify='center', scale=self.lscale).shift(0,0,1)
        labels.name = "closest_info"
        self.plt.at(1).remove("closest_info").add(labels)

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
        slc1.cmap(self.cmap, vmin=self.rng[0], vmax=self.rng[1])

        slc1.name = "slice1"
        slc0 = slc1.clone(transformed=1).z(self.frame).pickable(False)
        slc0.name = "slice0"
        self.plt.at(0).remove("slice0").add(slc0, render=False)
        self.plt.at(1).remove("slice1").add(slc1, render=False)

        line_pts = self.getpoints()
        if len(line_pts)==0:
            return
        minframe, maxframe = np.min(line_pts[:, 2]), np.max(line_pts[:, 2])
        trackline = vedo.Line(line_pts, lw=3, c="orange5")
        trackline.name = "track"
        trackline.cmap('autumn_r', self.getvelocity(), vmin=0, vmax=self.maxvelocity)
        self.plt.at(0).remove("track").add(trackline, render=False)

        self.plt.at(1).remove("pt2d", "track2d", "closest_info")
        if minframe <= self.frame <= maxframe:
            res = np.where(line_pts[:, 2] == self.frame)[0]
            if len(res):  # some frames might be missing
                pt2d = vedo.Point(line_pts[res[0]], c="red6", r=10).z(0)
                pt2d.pickable(False).useBounds(False)
                pt2d.name = "pt2d"
                self.plt.add(pt2d, render=False)

        self._slider1.GetRepresentation().SetValue(self.frame)
        self._slider2.GetRepresentation().SetTitleText(f"track id {self.track}")

        df = self.dataframe
        sox9level = df.loc[df["TRACK_ID"]==self.track][self.sox9name].to_numpy()
        x = list(range(len(sox9level)))
        self.plt.at(0).remove(self._cornerplot)
        self._cornerplot = vedo.pyplot.CornerPlot([x, sox9level], pos='top-right', c='dg', lines=False)
        self._cornerplot.GetXAxisActor2D().SetFontFactor(0.5)
        self._cornerplot.GetYAxisActor2D().SetFontFactor(0.5)
        self.plt.add(self._cornerplot, render=False)

        self.plt.render()

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
            self.plt.at(1).remove("track2d").add(trackline2d)
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
            self.plt.resetCamera()

        elif evt.keyPressed == "q":
            self.plt.close()
            return
        self.update()

    ######################################################
    def start(self):

        slc = self.volume.zSlice(self.frame).lighting("off").z(-self.frame)
        slc.cmap(self.cmap, vmin=self.rng[0], vmax=self.rng[1])

        txt = vedo.Text2D(self.info, font="Calco", bg="yellow7")

        axes = vedo.Axes(
            self.volume,
            xtitle="x /pixel",
            ytitle="y /pixel",
            ztitle="frame nr.",
        )

        self.plt.at(0).show(axes, txt, camera=self.camera, bg2='light cyan')
        self.plt.at(1).show(slc, resetcam=True, zoom=1.1)
        self.update()
        self.plt.interactive().close()
