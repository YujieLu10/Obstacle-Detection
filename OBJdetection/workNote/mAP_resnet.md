### scale [1920, 1200] range [-1, 90]

Running per image evaluation...

DONE (t=139.67s).

Accumulating evaluation results...

DONE (t=24.34s).

~~~ Mean and per-category AP @ IoU=0.50,0.95] ~~~~

all              26.0
> self.classes ['__background__', u'car', u'van', u'bus', u'truck', u'forklift', u'person', u'person-sitting', u'bicycle', u'motor', u'open-tricycle', u'close-tricycle', u'water-block', u'cone-block', u'other-block', u'crash-block', u'triangle-block', u'warning-block', u'small-block', u'large-block', u'bicycle-group', u'person-group', u'motor-group', u'parked-bicycle', u'parked-motor', u'cross-bar']

car              49.7

van              45.9

bus              45.0

truck            49.6

forklift         42.0

person           25.8

person-sitting   32.3

bicycle          17.6

motor            41.9

open-tricycle    41.0

close-tricycle   28.7

water-block      29.5

cone-block       16.2

other-block      14.2

crash-block      27.5

triangle-block    0.2

warning-block    13.9

/home/luyujie/.local/lib/python2.7/site-packages/numpy/core/fromnumeric.py:2957: RuntimeWarning: Mean of empty slice.
  out=out, **kwargs)
/home/luyujie/.local/lib/python2.7/site-packages/numpy/core/_methods.py:80: RuntimeWarning: invalid value encountered in double_scalars
  ret = ret.dtype.type(ret / rcount)
  
  small-block       nan

large-block       nan

bicycle-group     nan

person-group      nan

motor-group       nan

parked-bicycle    0.0

parked-motor      0.0

cross-bar         0.0

~~~ Summary metrics ~~~~

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.260
 
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.437
 
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.268
 
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.199
 
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.633
 
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.064
 
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.232
 
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.367
 
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.390
 
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.349
 
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.711
 
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.073
 
 coco eval results saved to ./output/sniper_res101_bn/results/detections_test-dev2015_results.pkl

All done!


### scale [1280, 800] range [32, 180]
Accumulating evaluation results...

DONE (t=15.21s).

~~~ Mean and per-category AP @ IoU=0.50,0.95] ~~~~

all              14.3

> self.classes ['__background__', u'car', u'van', u'bus', u'truck', u'forklift', u'person', u'person-sitting', u'bicycle', u'motor', u'open-tricycle', u'close-tricycle', u'water-block', u'cone-block', u'other-block', u'crash-block', u'triangle-block', u'warning-block', u'small-block', u'large-block', u'bicycle-group', u'person-group', u'motor-group', u'parked-bicycle', u'parked-motor', u'cross-bar']

car              25.2

van              25.1

bus              28.2

truck            32.8

forklift         42.4

person            9.7

person-sitting    9.2

bicycle           5.7

motor            24.3

open-tricycle    20.4

close-tricycle   13.5

water-block      16.1

cone-block        8.3

other-block       4.9

crash-block      14.5

triangle-block    0.0

warning-block     6.0

/home/luyujie/.local/lib/python2.7/site-packages/numpy/core/fromnumeric.py:2957: RuntimeWarning: Mean of empty slice.
  out=out, **kwargs)
/home/luyujie/.local/lib/python2.7/site-packages/numpy/core/_methods.py:80: RuntimeWarning: invalid value encountered in double_scalars
  ret = ret.dtype.type(ret / rcount)
  
  small-block       nan

large-block       nan

bicycle-group     nan

person-group      nan

motor-group       nan

parked-bicycle    0.0

parked-motor      0.0

cross-bar         0.0

~~~ Summary metrics ~~~~

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.143
 
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.224
 
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.151
 
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.044
 
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.512
 
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.748
 
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.146
 
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.211
 
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.215
 
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.110
 
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.639
 
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.786
 
 coco eval results saved to ./output/sniper_res101_bn/results/detections_test-dev2015_results.pkl

All done!


### scale [640, 400] range [75, -1]

Running per image evaluation...

DONE (t=69.45s).

Accumulating evaluation results...

DONE (t=18.03s).

~~~ Mean and per-category AP @ IoU=0.50,0.95] ~~~~

all              14.3

>>>> self.classes ['__background__', u'car', u'van', u'bus', u'truck', u'forklift', u'person', u'person-sitting', u'bicycle', u'motor', u'open-tricycle', u'close-tricycle', u'water-block', u'cone-block', u'other-block', u'crash-block', u'triangle-block', u'warning-block', u'small-block', u'large-block', u'bicycle-group', u'person-group', u'motor-group', u'parked-bicycle', u'parked-motor', u'cross-bar']

car              25.2

van              25.1

bus              28.2

truck            32.8

forklift         42.4

person            9.7

person-sitting    9.2

bicycle           5.7

motor            24.3

open-tricycle    20.4

close-tricycle   13.5

water-block      16.1

cone-block        8.3

other-block       4.9

crash-block      14.5

triangle-block    0.0

warning-block     6.0

/home/luyujie/.local/lib/python2.7/site-packages/numpy/core/fromnumeric.py:2957: RuntimeWarning: Mean of empty slice.
  out=out, **kwargs)
/home/luyujie/.local/lib/python2.7/site-packages/numpy/core/_methods.py:80: RuntimeWarning: invalid value encountered in double_scalars
  ret = ret.dtype.type(ret / rcount)
  
  small-block       nan

large-block       nan

bicycle-group     nan

person-group      nan

motor-group       nan

parked-bicycle    0.0

parked-motor      0.0

cross-bar         0.0

~~~ Summary metrics ~~~~

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.143
 
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.224
 
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.151
 
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.044
 
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.512
 
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.748
 
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.146
 
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.211
 
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.215
 
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.110
 
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.639
 
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.786
 
 coco eval results saved to ./output/sniper_res101_bn/results/detections_test-dev2015_results.pkl

All done!

