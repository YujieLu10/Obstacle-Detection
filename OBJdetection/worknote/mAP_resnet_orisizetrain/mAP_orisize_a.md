~~~~ Mean and per-category AP @ IoU=0.50,0.95] ~~~~
all              33.0

self.classes ['__background__', u'car', u'van', u'bus', u'truck', u'forklift', u'person', u'person-sitting', u'bicycle', u'motor', u'open-tricycle', u'close-tricycle', u'water-block', u'cone-block', u'other-block', u'crash-block', u'triangle-block', u'warning-block', u'small-block', u'large-block', u'bicycle-group', u'person-group', u'motor-group', u'parked-bicycle', u'parked-motor', u'cross-bar']

car              59.6
van              55.1
bus              55.8
truck            58.9
forklift         66.2
person           27.0
person-sitting   16.5
bicycle          22.7
motor            49.5
open-tricycle    46.8
close-tricycle   32.8
water-block      28.1
cone-block       21.3
other-block      20.4
crash-block      30.0
triangle-block    0.0
warning-block    19.1
/home/luyujie/.local/lib/python2.7/site-packages/numpy/core/fromnumeric.py:2957: RuntimeWarning: Mean of empty slice.
  out=out, **kwargs)
  /home/luyujie/.local/lib/python2.7/site-packages/numpy/core/_methods.py:80: RuntimeWarning: invalid value encountered in double_scalars
    ret = ret.dtype.type(ret / rcount)
    small-block       nan
    large-block       nan
    bicycle-group     nan
    person-group      nan
    motor-group       nan
    parked-bicycle   25.8
    parked-motor     13.8
    cross-bar        10.5

    ~~~ Summary metrics ~~~~

     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.330
      Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.529
       Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.354
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.123
         Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.383
          Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.572
           Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.308
            Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.495
             Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.523
              Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.261
               Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.633
                Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.750

                coco eval results saved to ./output/sniper_res101_bn/results/detections_test-dev2015_results.pkl
                All done!
