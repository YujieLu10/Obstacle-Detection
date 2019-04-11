 ### scale [640, 400] range [75, -1]

 SCALES:

  - !!python/tuple [640, 400]

  VALID_RANGES:

  - !!python/tuple [75,-1]

  Accumulating evaluation results...

DONE (t=0.05s).

~~~ Mean and per-category AP @ IoU=0.50,0.95] ~~~~

all              12.5

> self.classes ['__background__', u'car', u'van', u'bus', u'truck', u'forklift', u'person', u'person-sitting', u'bicycle', u'motor', u'open-tricycle', u'close-tricycle', u'water-block', u'cone-block', u'other-block', u'crash-block', u'triangle-block', u'warning-block', u'small-block', u'large-block', u'bicycle-group', u'person-group', u'motor-group', u'parked-bicycle', u'parked-motor', u'cross-bar']

car              20.2

van               0.1

bus               0.0

truck            39.8

/home/luyujie/.local/lib/python2.7/site-packages/numpy/core/fromnumeric.py:2957: RuntimeWarning: Mean of empty slice.
  out=out, **kwargs)
/home/luyujie/.local/lib/python2.7/site-packages/numpy/core/_methods.py:80: RuntimeWarning: invalid value encountered in double_scalars
  ret = ret.dtype.type(ret / rcount)

forklift          nan

person            0.0

person-sitting    nan

bicycle           nan

motor            15.1

open-tricycle     nan

close-tricycle    nan

water-block       nan

cone-block        nan

other-block       nan

crash-block       nan

triangle-block    nan

warning-block     nan

small-block       nan

large-block       nan

bicycle-group     nan

person-group      nan

motor-group       nan

parked-bicycle    nan

parked-motor      nan

cross-bar         nan

~~~ Summary metrics ~~~~

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.125

 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.182

 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.128

 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.025

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.476

 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.935

 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.137

 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.166

 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.166

 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.059

 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.570

 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.967

coco eval results saved to ./output/sniper_res101_bn/results/detections_test-dev2015_results.pkl

All done!

### scale [640, 400] range [32,180]

Accumulating evaluation results...

DONE (t=0.05s).

~~~ Mean and per-category AP @ IoU=0.50,0.95] ~~~~

all              12.1

> self.classes ['__background__', u'car', u'van', u'bus', u'truck', u'forklift', u'person', u'person-sitting', u'bicycle', u'motor', 
u'open-tricycle', u'close-tricycle', u'water-block', u'cone-block', u'other-block', u'crash-block', u'triangle-block', u'warning-block', 
u'small-block', u'large-block', u'bicycle-group', u'person-group', u'motor-group', u'parked-bicycle', u'parked-motor', u'cross-bar']

car              20.9

van               0.4

bus               0.0

truck            35.0

/home/luyujie/.local/lib/python2.7/site-packages/numpy/core/fromnumeric.py:2957: RuntimeWarning: Mean of empty slice.
  out=out, **kwargs)
/home/luyujie/.local/lib/python2.7/site-packages/numpy/core/_methods.py:80: RuntimeWarning: invalid value encountered in double_scalars
  ret = ret.dtype.type(ret / rcount)

forklift          nan

person            0.0

person-sitting    nan

bicycle           nan

motor            16.2

open-tricycle     nan

close-tricycle    nan

water-block       nan

cone-block        nan

other-block       nan

crash-block       nan

triangle-block    nan

warning-block     nan

small-block       nan

large-block       nan

bicycle-group     nan

person-group      nan

motor-group       nan

parked-bicycle    nan

parked-motor      nan

cross-bar         nan

~~~ Summary metrics ~~~~

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121

 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.173

 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.130

 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.020

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.500

 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.809

 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.130

 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.156

 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.156

 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.041

 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.585

 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.900

coco eval results saved to ./output/sniper_res101_bn/results/detections_test-dev2015_results.pkl

All done!

### scale [640, 400] range [-1,90]

Accumulating evaluation results...

DONE (t=0.05s).

~~~ Mean and per-category AP @ IoU=0.50,0.95] ~~~~

all              11.3

> self.classes ['__background__', u'car', u'van', u'bus', u'truck', u'forklift', u'person', u'person-sitting', u'bicycle', u'motor', u'open-tricycle', u'close-tricycle', u'water-block', u'cone-block', u'other-block', u'crash-block', u'triangle-block', u'warning-block', u'small-block', u'large-block', u'bicycle-group', u'person-group', u'motor-group', u'parked-bicycle', u'parked-motor', u'cross-bar']

car              20.0

van               0.4

bus               0.0

truck            30.0

/home/luyujie/.local/lib/python2.7/site-packages/numpy/core/fromnumeric.py:2957: RuntimeWarning: Mean of empty slice.
  out=out, **kwargs)
/home/luyujie/.local/lib/python2.7/site-packages/numpy/core/_methods.py:80: RuntimeWarning: invalid value encountered in double_scalars
  ret = ret.dtype.type(ret / rcount)

forklift          nan

person            0.0

person-sitting    nan

bicycle           nan

motor            17.3

open-tricycle     nan

close-tricycle    nan

water-block       nan

cone-block        nan

other-block       nan

crash-block       nan

triangle-block    nan

warning-block     nan

small-block       nan

large-block       nan

bicycle-group     nan

person-group      nan

motor-group       nan

parked-bicycle    nan

parked-motor      nan

cross-bar         nan

~~~ Summary metrics ~~~~

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.113

 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.172

 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.117

 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.026

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.515

 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.344

 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.125

 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.157

 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.157

 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.064

 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.601

 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.346

coco eval results saved to ./output/sniper_res101_bn/results/detections_test-dev2015_results.pkl

All done!

### scale [1920, 1200] range [-1,90]

Accumulating evaluation results...

DONE (t=0.09s).

~~~ Mean and per-category AP @ IoU=0.50,0.95] ~~~~
all              44.2

> self.classes ['__background__', u'car', u'van', u'bus', u'truck', u'forklift', u'person', u'person-sitting', u'bicycle', u'motor', u'open-tricycle', u'close-tricycle', u'water-block', u'cone-block', u'other-block', u'crash-block', u'triangle-block', u'warning-block', u'small-block', u'large-block', u'bicycle-group', u'person-group', u'motor-group', u'parked-bicycle', u'parked-motor', u'cross-bar']

car              57.3

van              22.3

bus              45.5

truck            65.8

/home/luyujie/.local/lib/python2.7/site-packages/numpy/core/fromnumeric.py:2957: RuntimeWarning: Mean of empty slice.
  out=out, **kwargs)
/home/luyujie/.local/lib/python2.7/site-packages/numpy/core/_methods.py:80: RuntimeWarning: invalid value encountered in double_scalars
  ret = ret.dtype.type(ret / rcount)

forklift          nan

person            0.1

person-sitting    nan

bicycle           nan

motor            74.2

open-tricycle     nan

close-tricycle    nan

water-block       nan

cone-block        nan

other-block       nan

crash-block       nan

triangle-block    nan

warning-block     nan

small-block       nan

large-block       nan

bicycle-group     nan

person-group      nan

motor-group       nan

parked-bicycle    nan

parked-motor      nan

cross-bar         nan

~~~ Summary metrics ~~~~

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.442

 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.661

 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.510

 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.386

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.831

 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.151

 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.458

 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.587

 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.590

 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.565

 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.881

 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.150

coco eval results saved to ./output/sniper_res101_bn/results/detections_test-dev2015_results.pkl

All done!

### scale [1280, 800] range [32,180]

Accumulating evaluation results...

DONE (t=0.07s).

~~~ Mean and per-category AP @ IoU=0.50,0.95] ~~~~

all              33.3

> self.classes ['__background__', u'car', u'van', u'bus', u'truck', u'forklift', u'person', u'person-sitting', u'bicycle', u'motor', u'open-tricycle', u'close-tricycle', u'water-block', u'cone-block', u'other-block', u'crash-block', u'triangle-block', u'warning-block', u'small-block', u'large-block', u'bicycle-group', u'person-group', u'motor-group', u'parked-bicycle', u'parked-motor', u'cross-bar']

car              31.5

van              13.8

bus              21.1

truck            65.0

/home/luyujie/.local/lib/python2.7/site-packages/numpy/core/fromnumeric.py:2957: RuntimeWarning: Mean of empty slice.
  out=out, **kwargs)
/home/luyujie/.local/lib/python2.7/site-packages/numpy/core/_methods.py:80: RuntimeWarning: invalid value encountered in double_scalars
  ret = ret.dtype.type(ret / rcount)

forklift          nan

person            0.0

person-sitting    nan

bicycle           nan

motor            68.0

open-tricycle     nan

close-tricycle    nan

water-block       nan

cone-block        nan

other-block       nan

crash-block       nan

triangle-block    nan

warning-block     nan

small-block       nan

large-block       nan

bicycle-group     nan

person-group      nan

motor-group       nan

parked-bicycle    nan

parked-motor      nan

cross-bar         nan

~~~ Summary metrics ~~~~

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.333

 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.503

 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.379

 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.238

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.726

 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.928

 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.357

 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.457

 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.458

 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.378

 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.792

 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.942

coco eval results saved to ./output/sniper_res101_bn/results/detections_test-dev2015_results.pkl

All done!

### scale [1920, 1200] [1280, 800] [640, 400] range [-1, 90] [32, 180] [75, -1]

Accumulating evaluation results...

DONE (t=0.11s).

~~~ Mean and per-category AP @ IoU=0.50,0.95] ~~~~

all              45.1

> self.classes ['__background__', u'car', u'van', u'bus', u'truck', u'forklift', u'person', u'person-sitting', u'bicycle', u'motor', u'open-tricycle', u'close-tricycle', u'water-block', u'cone-block', u'other-block', u'crash-block', u'triangle-block', u'warning-block', u'small-block', u'large-block', u'bicycle-group', u'person-group', u'motor-group', u'parked-bicycle', u'parked-motor', u'cross-bar']

car              58.6

van              18.8

bus              44.5

truck            74.4

/home/luyujie/.local/lib/python2.7/site-packages/numpy/core/fromnumeric.py:2957: RuntimeWarning: Mean of empty slice.
  out=out, **kwargs)
/home/luyujie/.local/lib/python2.7/site-packages/numpy/core/_methods.py:80: RuntimeWarning: invalid value encountered in double_scalars
  ret = ret.dtype.type(ret / rcount)

forklift          nan

person            0.1

person-sitting    nan

bicycle           nan

motor            74.0

open-tricycle     nan

close-tricycle    nan

water-block       nan

cone-block        nan

other-block       nan

crash-block       nan

triangle-block    nan

warning-block     nan

small-block       nan

large-block       nan

bicycle-group     nan

person-group      nan

motor-group       nan

parked-bicycle    nan

parked-motor      nan

cross-bar         nan

~~~ Summary metrics ~~~~

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.451

 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.668

 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.527

 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.375

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.810

 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.943

 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.453

 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.594

 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.599

 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.551

 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.879

 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.971

coco eval results saved to ./output/sniper_res101_bn/results/detections_test-dev2015_results.pkl

All done!

### scale [1400, 2000] [800, 1280] [480, 512] range [-1, 90] [32, 180] [75, -1]

Accumulating evaluation results...

DONE (t=0.12s).

~~~ Mean and per-category AP @ IoU=0.50,0.95] ~~~~

all              56.0

> self.classes ['__background__', u'car', u'van', u'bus', u'truck', u'forklift', u'person', u'person-sitting', u'bicycle', u'motor', u'open-tricycle', 
u'close-tricycle', u'water-block', u'cone-block', u'other-block', u'crash-block', u'triangle-block', u'warning-block', u'small-block', u'large-block', u'bicycle-group', u'person-group', u'motor-group', u'parked-bicycle', u'parked-motor', u'cross-bar']

car              67.7

van              42.2

bus              63.6

truck            83.5

/home/luyujie/.local/lib/python2.7/site-packages/numpy/core/fromnumeric.py:2957: RuntimeWarning: Mean of empty slice.
  out=out, **kwargs)
/home/luyujie/.local/lib/python2.7/site-packages/numpy/core/_methods.py:80: RuntimeWarning: invalid value encountered in double_scalars
  ret = ret.dtype.type(ret / rcount)

forklift          nan

person            2.6

person-sitting    nan

bicycle           nan

motor            76.3

open-tricycle     nan

close-tricycle    nan

water-block       nan

cone-block        nan

other-block       nan

crash-block       nan

triangle-block    nan

warning-block     nan

small-block       nan

large-block       nan

bicycle-group     nan

person-group      nan

motor-group       nan

parked-bicycle    nan

parked-motor      nan


cross-bar         nan

~~~ Summary metrics ~~~~

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.560

 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.781

 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.641

 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.508

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.844

 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.954

 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.556

 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.707

 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.721

 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.698

 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.888

 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.988

coco eval results saved to ./output/sniper_res101_bn/results/detections_test-dev2015_results.pkl

All done!

