# Guideline for EtherPose

## About

This is research code for EtherPose: Continuous Hand Pose Tracking with Wrist-Worn Antenna Impedance Characteristic Sensing (UIST 2022). It contains base data collection, regression, and visualization pipeline. More info can be found [here](https://daehwa.github.io/paper/EtherPose_paper.pdf) (To appear).

![](./img/etherpose_example.gif)

<!-- <p align="center">
    <video preload="none" autoplay loop muted playsinline poster="https://daehwa.github.io/img/etherpose.png">
      <source src="https://daehwa.github.io/img/etherpose.mp4" type="video/mp4">
    </video>
</p>
 -->

## Installation

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

After installing dependencies, we have two files to change.


First, change `line35` of the file `venv/lib/python3.*/site-packages/OpenGL/platform/ctypesloader.py`
<br />
From
```
fullName = util.find_library( name )
```
to 
```
fullName = '/System/Library/Frameworks/OpenGL.framework/OpenGL'
```


Second, we modified pyrender library to visualize hand mesh. Change `venv/lib/python3.*/site-packages/pyrender/viewer.py` to `etherpose_viewer/viewer.py`.

Download [inverse kinematics solver](https://github.com/CalciferZh/Minimal-IK) and place files in `mano_ik/inverse_kinematics/`.

## How to use

```
python etherpose_demo.py -S 1.36e9 -E 1.4e9 -N 21 --plot
```
`-S` is for setting the start frequency. `-E` is for setting the end freqeuncy. `-N` is for setting the number of data points. `--plot` is for turning on the signal plot.

`c`: Calibration. It finds the most "local" biggest return-loss value in and centeralizes the corresponding frequency.
<br />
`t`: Time-domain data visualization.
<br />
`p`: Trackpad mode.
<br />
`h`: Hand pose estimation mode.
<br />
`space`: Recording train data.
<br />
`enter`: Prediction start / stop.

## Hardware Configuration

We used [Cloverleaf antenna](https://www.amazon.com/gp/product/B072V1F8V3/ref=ppx_yo_dt_b_search_asin_title?ie=UTF8&psc=1) with a single-layered copper ground plane. After soldering the [SMA Female Connector](https://www.amazon.com/gp/product/B07XWL597P/ref=ppx_yo_dt_b_search_asin_image?ie=UTF8&psc=1), the antenna was connected to [NanoVNA V2](https://www.amazon.com/gp/product/B08F9MR3NT/ref=ppx_yo_dt_b_search_asin_title?ie=UTF8&psc=1) through the [50ohm RF coaxial Cable](https://www.amazon.com/gp/product/B07YHLGVT9/ref=ppx_yo_dt_b_search_asin_title?ie=UTF8&psc=1) and some [connectors](https://www.amazon.com/gp/product/B07Z2QB9M8/ref=ppx_yo_dt_b_search_asin_title?ie=UTF8&psc=1). For more detail on the antenna geometries, please refer to the paper.

## Credit

[NanoVNA](https://github.com/nanovna-v2/NanoVNA2-firmware)
<br />
[Minimal-IK by CalciferZh](https://github.com/CalciferZh/Minimal-IK)
