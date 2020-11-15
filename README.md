# Yolov5 + Deep Sort with PyTorch

[![HitCount](http://hits.dwyl.com/{mikel-brostrom}/{Yolov5_DeepSort_Pytorch}.svg)](http://hits.dwyl.com/{mikel-brostrom}/{Yolov5_DeepSort_Pytorch})


![](Town.gif)

## Introduction

This repository contains a moded version of PyTorch YOLOv5 (https://github.com/ultralytics/yolov5). It filters out every detection that is not a person. The detections of persons are then passed to a Deep Sort algorithm (https://github.com/ZQPei/deep_sort_pytorch) which tracks the persons. The reason behind the fact that it just tracks persons is that the deep association metric is trained on a person ONLY datatset.

## Description

The implementation is based on two papers:

- Simple Online and Realtime Tracking with a Deep Association Metric
https://arxiv.org/abs/1703.07402
- YOLOv4: Optimal Speed and Accuracy of Object Detection
https://arxiv.org/pdf/2004.10934.pdf

## Requirements

Python 3.8 or later with all requirements.txt dependencies installed, including torch>=1.6. To install run:

`pip install -U -r requirements.txt`

All dependencies are included in the associated docker images. Docker requirements are: 
- `nvidia-docker`
- Nvidia Driver Version >= 440.44

Do not forget to install the requirements specified in the `deep_sort` and `yolov5` directories.


## Before you run the tracker

Github block pushes of files larger than 100 MB (https://help.github.com/en/github/managing-large-files/conditions-for-large-files). Hence you need to download two different weights: the ones for yolo and the ones for deep sort

- download the yolov5 weight from https://drive.google.com/drive/folders/1Drs_Aiu7xx6S-ix95f9kNsA6ueKRpN2J. Place the downlaoded `.pt` file under `yolov5/weights/`
- download the deep sort weights from https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6. Place ckpt.t7 file under`deep_sort/deep/checkpoint/`

## Tracking

Tracking can be run on most video formats

```bash
python3 track.py --source ...
```

- Video:  `--source file.mp4`
- Webcam:  `--source 0`
- RTSP stream:  `--source rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa`
- HTTP stream:  `--source http://wmccpinetop.axiscam.net/mjpg/video.mjpg`

MOT compliant results can be saved to `inference/output` by 

```bash
python3 track.py --source ... --save-txt
```

## Other information

For more detailed information about the algorithms and their corresponding lisences used in this project access their official github implementations.


## Real Time People Counter         [These are my changes.]

This repository also contains code to track people entering and exiting your reference location.
Follow the below steps to test with your custom video feed.

- Run the script by passing `--list` as an additional argument.

```bash
python3 track.py --source ... --list
```
- This opens up a window with the first frame of the video source.
- You need to draw the reference line used to compute the entry/exit of a person. (aka drawing the perdiction border).
- Left click on the frame to select the one end of the prediction border and middle click on the frame for selecting the other end of the prediction border.
- People moving from the bottom of the prediction border to the top of the prediction are considered as people entering your reference location(store/building, etc).
- Bottom co-ordinates of the boudning box is used to estimate the movement direction.
