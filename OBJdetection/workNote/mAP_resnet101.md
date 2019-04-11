DONE (t=163.18s).

Accumulating evaluation results...

DONE (t=35.43s).

~~~ Mean and per-category AP @ IoU=0.50,0.95] ~~~~

all              37.4

> self.classes ['__background__', u'car', u'van', u'bus', u'truck', u'forklift', u'person', u'person-sitting', u'bicycle', u'motor', u'open-tricycle', u'close-tricycle', u'water-block', u'cone-block', u'other-block', u'crash-block', u'triangle-block', u'warning-block', u'small-block', u'large-block', u'bicycle-group', u'person-group', u'motor-group', u'parked-bicycle', u'parked-motor', u'cross-bar']

car              62.8

van              61.4

bus              61.2

truck            65.8

forklift         77.0

person           38.5

person-sitting   41.4

bicycle          33.5

motor            53.8

open-tricycle    53.3

close-tricycle   42.4

water-block      39.2

cone-block       28.1

other-block      26.7

crash-block      38.1

triangle-block    0.1

warning-block    24.6

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

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.374
 
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.597
 
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.400
 
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.316
 
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.651
 
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.816
 
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.314
 
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.493
 
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.523
 
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.481
 
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.736
 
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.875
 
 coco eval results saved to ./output/sniper_res101_bn/results/detections_test-dev2015_results.pkl

All done!

