# Track Viewer

Browse interactively cell tracks from segmented samples

## To install
```bash
pip install rich
pip install -U vedo
pip install vtk==9.0.3  # optional
git clone https://github.com/marcomusy/trackviewer.git
```

## To run

```bash
cd trackviewer
python main.py
```

![](https://user-images.githubusercontent.com/32848391/194614159-a4ad615a-dd3d-40c0-9381-6c83e6d1cc56.png)


## Mouse and Keyboard controls

While in the application press:
```
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
- q to quit
```

