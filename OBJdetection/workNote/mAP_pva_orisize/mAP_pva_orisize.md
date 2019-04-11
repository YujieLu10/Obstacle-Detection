Running per image evaluation...
DONE (t=192.45s).
Accumulating evaluation results...
DONE (t=44.83s).
~~~~ Mean and per-category AP @ IoU=0.50,0.95] ~~~~
all              27.3
>>>> self.classes ['__background__', u'car', u'van', u'bus', u'truck', u'forklift', u'person', u'person-sitting', u'bicycle', u'motor', u'open-tricycle', u'close-tricycle', u'water-block', u'cone-block', u'other-block', u'crash-block', u'triangle-block', u'warning-block', u'small-block', u'large-block', u'bicycle-group', u'person-group', u'motor-group', u'parked-bicycle', u'parked-motor', u'cross-bar']
car              56.6
van              50.5
bus              51.1
truck            52.1
forklift         56.2
person           21.2
person-sitting   16.5
bicycle          13.6
motor            45.6
open-tricycle    33.5
close-tricycle   16.2
water-block      18.1
cone-block       14.4
other-block      17.1
crash-block      26.8
triangle-block    0.0
warning-block    18.2
/home/luyujie/.local/lib/python2.7/site-packages/numpy/core/fromnumeric.py:2957: RuntimeWarning: Mean of empty slice.
  out=out, **kwargs)
/home/luyujie/.local/lib/python2.7/site-packages/numpy/core/_methods.py:80: RuntimeWarning: invalid value encountered in double_scalars
  ret = ret.dtype.type(ret / rcount)
small-block       nan
large-block       nan
bicycle-group     nan
person-group      nan
motor-group       nan
parked-bicycle   17.3
parked-motor      8.7
cross-bar        12.5
~~~~ Summary metrics ~~~~
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.273
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.452
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.291
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.069
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.321
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.479
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.263
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.430
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.458
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.195
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.559
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.682
coco eval results saved to ./output/pvalite_b5_bn/results/detections_test-dev2015_results.pkl
All done!