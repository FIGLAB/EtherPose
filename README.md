# EtherPose

## About

## Installation

```
python3 -m venv venv
```

```
source venv/bin/activate
```

```
pip install -r requirements.txt
```

`venv/lib/python3.*/site-packages/OpenGL/platform/ctypesloader.py`

`line35`

```
fullName = util.find_library( name )
```
to 
```
fullName = '/System/Library/Frameworks/OpenGL.framework/OpenGL'
```

Change `venv/lib/python3.*/site-packages/pyrender/viewer.py` to `etherpose_viewer/viewer.py`.

## How to use

```
python etherpose_demo.py -S 1.36e9 -E 1.4e9 -N 21 --plot
```

`c`: calibration

`t`: time serial data visualization

`p`: trackpad

`h`: handpose

`space`: record data

`enter`: prediction start/stop

## Credit
