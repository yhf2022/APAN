# APAN

## Datesets
We will provide links to our data sets as soon as possible

### Data directory structure in APAN
Our data directory structure is the same as that of VOC dataset

```
.
├── ...
├── VOCdevkit                   
│   └── VOC2007
│       ├── Annotations
│       │       └── *.xml           
│       ├── ImageSets 
│       │       └── *.txt
│       └── JPEGImages
│               └── *.jpg
└── ...
```
## Train and Test

### Train

Train a APAN model
- Firstly, change directory to  VOCdevkit/VOC2007 folder
- Change the classes of targets which you want to detect in `classes.txt` 
- Then run `python train.py`, you can adjust learning parameters in `train.py`.

### Test

Test a APAN model
run `python test.py`, you can adjust learning parameters in `test.py`
