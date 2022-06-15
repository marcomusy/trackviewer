# Track Viewer

Browse interactively cell tracks from segmented samples

## To install
```bash
pip install rich
pip install -U git+https://github.com/marcomusy/vedo.git
git clone https://github.com/marcomusy/trackviewer.git
cd trackviewer
```

## To run

Adjust the paths in `main.py` then:

```bash
python main.py
```

![](https://user-images.githubusercontent.com/32848391/171412909-fb28f53d-aa42-4987-be4e-7cd6fb62d5da.png)


## Mouse and Keyboard controls

While in the application press:
```
- arrows to navigate
  (left/right to change frame, up/down to change track)
- mouse to rotate the scene in the left panel
- right-click and drag to zoom in and out
- left-click in the right panel to show closest tracks
- l to show track line
- c to show closest ids
- x to jump to track
- t to input a specific track id (you need to click in your terminal)
- r to reset camera
- q to quit
```

