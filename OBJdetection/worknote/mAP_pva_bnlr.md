Running per image evaluation...

DONE (t=182.48s).

Accumulating evaluation results...

DONE (t=61.94s).

~~~ Mean and per-category AP @ IoU=0.50,0.95] ~~~~

all              16.3
> self.classes ['__background__', u'car', u'van', u'bus', u'truck', u'forklift', u'person', u'person-sitting', u'bicycle', u'motor', u'open-tricycle', u'close-tricycle', u'water-block', u'cone-block', u'other-block', u'crash-block', u'triangle-block', u'warning-block', u'small-block', u'large-block', u'bicycle-group', u'person-group', u'motor-group', u'parked-bicycle', u'parked-motor', u'cross-bar']

car              50.7

van              42.9

bus              38.0

truck            38.9

forklift         39.8

person            9.9

person-sitting    0.1

bicycle           2.9

motor            35.4

open-tricycle    15.6

close-tricycle    2.2

water-block       3.5

cone-block        4.6

other-block      13.2

crash-block      15.8

triangle-block    0.0

warning-block     9.3

/home/luyujie/.local/lib/python2.7/site-packages/numpy/core/fromnumeric.py:2957: RuntimeWarning: Mean of empty slice.
  out=out, **kwargs)
/home/luyujie/.local/lib/python2.7/site-packages/numpy/core/_methods.py:80: RuntimeWarning: invalid value encountered in double_scalars
  ret = ret.dtype.type(ret / rcount)
  
  small-block       nan

large-block       nan

bicycle-group     nan

person-group      nan

motor-group       nan

parked-bicycle    2.6

parked-motor      0.4

cross-bar         0.0

~~~ Summary metrics ~~~~

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.163
 
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.279
 
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.169
 
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.119
 
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.311
 
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.474
 
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.173
 
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.279
 
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.294
 
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.244
 
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.510
 
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.626
 
 coco eval results saved to ./output/pvalite_b5_bn/results/detections_test-dev2015_results.pkl

All done!

