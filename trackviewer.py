import numpy as np
import pandas
import vedo

######################################################
class TrackViewer:
    """Track viewer"""

    def __init__(self):

        self.cmap = "Greys_r"
        self.frame = 0  # init values
        self.track = 0
        self.nframes = 0
        self.ntracks = 0
        self.slider1 = None
        self.slider2 = None
        self.dataframe = None
        self.volume = None
        self.volspot = None
        self.slices = []

        self.info = ("Press:\n"
                    "- arrows to navigate\n"
                    "- t to input track\n"
                    "- r to reset camera\n"
                    "- l to show track line\n"
                    "- q to quit")

        vedo.settings.enableDefaultMouseCallbacks = False
        vedo.settings.enableDefaultKeyboardCallbacks = False

        self.plt = vedo.Plotter(N=2, sharecam=False, title="Track Viewer")
        self.plt.addCallback("click mouse", self.on_click)
        self.plt.addCallback("key press", self.on_keypress)


    ######################################################
    def loadTracks(self, filename):
        """Load the track data from a cvs file"""
        vedo.printc("Loading track data from", filename, c="y")
        self.dataframe = pandas.read_csv(filename)
        self.ntracks = int(max(self.dataframe["TRACK_ID"]))

        self.slider2 = self.plt.at(1).addSlider2D(
            self._slider_track,
            0, self.ntracks - 1,
            value=self.track,
            pos=([0.1, 0.92], [0.4, 0.92]),
            title="track id",
        )

    ######################################################
    def loadVolume(self, filename):
        """Load the 3-channel tif stack"""
        vedo.printc("Loading volumetric dataset", filename, c="y")

        dataset = vedo.Volume(filename)
        arr = dataset.tonumpy(transpose=False)
        self.volume  = vedo.Volume(arr[2::3])
        self.volspot = vedo.Volume(arr[1::3])

        dims = self.volume.dimensions()
        self.nframes = int(dims[2] / 10)

        rng = self.volume.scalarRange()
        pb = vedo.ProgressBar(0, self.nframes, c="y")
        for i in range(self.nframes):
            s = self.volume.zSlice(i).lighting("off").cmap(self.cmap).z(-i)
            s.cmap(self.cmap, vmin=rng[0], vmax=rng[1]*0.8)
            self.slices.append(s)
            pb.print("slicing")

        self.slider1 = self.plt.at(1).addSlider2D(
            self._slider_time,
            0, self.nframes - 1,
            value=self.frame,
            pos=([0.1, 0.08], [0.4, 0.08]),
            title="time",
        )

    ######################################################
    def getpoints(self):
        """Get point coords for a track"""
        track_df = self.dataframe.loc[self.dataframe["TRACK_ID"] == self.track]
        if len(track_df) == 0:
            return None
        line_pts = np.c_[track_df["POSITION_X"], track_df["POSITION_Y"], track_df["FRAME"]]
        return line_pts

    ######################################################
    def update(self):
        """Update visualization"""
        self.track = min(self.track, self.ntracks - 1)
        self.track = max(self.track, 0)
        self.frame = min(self.frame, self.nframes - 1)
        self.frame = max(self.frame, 0)

        slc1 = self.slices[self.frame]
        slc1.name = "slice1"
        slc0 = slc1.clone(transformed=1).z(self.frame)
        slc0.name = "slice0"
        self.plt.at(0).remove("slice0").add(slc0, render=False)
        self.plt.at(1).remove("slice1").add(slc1, render=False)

        line_pts = self.getpoints()
        if line_pts is None:
            return
        minframe, maxframe = np.min(line_pts[:, 2]), np.max(line_pts[:, 2])
        trackline = vedo.Line(line_pts, lw=3, c="orange5")
        trackline.name = "track"
        self.plt.at(0).remove("track").add(trackline, render=False)

        self.plt.at(1).remove(["pt2d", "track2d"])
        if minframe <= self.frame <= maxframe:
            res = np.where(line_pts[:, 2] == self.frame)[0]
            if len(res):  # some frames might be missing
                pt2d = vedo.Point(line_pts[res[0]], c="red6", r=9).z(0)
                pt2d.pickable(False).useBounds(False)
                pt2d.name = "pt2d"
                self.plt.add(pt2d, render=False)

        self.slider1.GetRepresentation().SetValue(self.frame)
        self.slider2.GetRepresentation().SetValue(self.track)
        self.plt.render()

    ######################################################
    def _slider_time(self, obj, _):
        self.frame = int(obj.GetRepresentation().GetValue())
        self.update()

    def _slider_track(self, obj, _):
        self.track = int(obj.GetRepresentation().GetValue())
        self.update()

    ######################################################
    def on_click(self, evt):
        """Clicking on the right image will dump some info"""
        if not evt.actor:
            return
        sid = self.volspot.zSlice(self.frame)
        sid.name = "slice1"
        pid = sid.closestPoint(evt.picked3d, returnPointId=True)
        spid = int(self.volspot.pointdata[0][pid])
        vedo.printc("SpotID:", spid, "TrackID:", self.track, invert=True)

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
            self.track += 1
            line_pts = self.getpoints()
            self.frame = int(np.min(line_pts[:, 2]))

        elif evt.keyPressed == "Down":
            self.track -= 1
            line_pts = self.getpoints()
            self.frame = int(np.min(line_pts[:, 2]))

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

        elif evt.keyPressed == "r":
            self.plt.resetCamera()

        elif evt.keyPressed == "q":
            self.plt.close()
            return
        self.update()

    ######################################################
    def start(self):

        s0 = self.volume.zSlice(0).lighting("off").cmap(self.cmap).alpha(0.2)
        s1 = self.volume.zSlice(self.nframes-1).lighting("off").cmap(self.cmap).alpha(0.2)

        txt = vedo.Text2D(self.info, font="Calco", bg="yellow7")

        axes = vedo.Axes(
            self.volume,
            xtitle="x /pixel",
            ytitle="y /pixel",
            ztitle="frame nr.",
            xyGrid=False,
        )

        cam = dict(
            pos=(999.2, -1247, 991.1),
            focalPoint=(227.6, 236.1, 248.5),
            viewup=(-0.1758, 0.3662, 0.9138),
        )
        self.plt.at(0).show(s0, s1, axes, txt, camera=cam)
        self.plt.at(1).show(self.slices[self.frame], resetcam=True)
        self.plt.interactive().close()
