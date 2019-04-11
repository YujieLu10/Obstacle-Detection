train1030
val1010
no occlusion = 2

Loading and preparing results...
DONE (t=16.35s)
creating index...
index created!
lib/dataset/pycocotools/cocoeval.py:466: DeprecationWarning: object of type <type 'numpy.float64'> cannot be safely interpreted as an integer.
  self.iouThrs = np.linspace(.5, 0.95, np.round((0.95-.5)/.05)+1, endpoint=True)
  lib/dataset/pycocotools/cocoeval.py:467: DeprecationWarning: object of type <type 'numpy.float64'> cannot be safely interpreted as an integer.
    self.recThrs = np.linspace(.0, 1.00, np.round((1.00-.0)/.01)+1, endpoint=True)
    Running per image evaluation...
    DONE (t=85.37s).
    Accumulating evaluation results...
    DONE (t=25.19s).
    ~~~ Mean and per-category AP @ IoU=0.50,0.50] ~~~~
    all              65.4
    >>>> self.classes ['__background__', u'car', u'bus', u'truck', u'person', u'bicycle', u'tricycle', u'block']
    car              81.9
    bus              73.2
    truck            70.0
    person           46.7
    bicycle          79.9
    tricycle         73.9
    block            32.3
    ~~~ Mean and per-category AR @ IoU=0.50,0.50] ~~~~
    all              87.2
    >>>> self.classes ['__background__', u'car', u'bus', u'truck', u'person', u'bicycle', u'tricycle', u'block']
    car              93.2
    bus              94.5
    truck            91.2
    person           78.1
    bicycle          89.8
    tricycle         92.5
    block            70.9
    ~~~ Summary metrics ~~~~
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.451
      Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.654
       Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.519
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.211
         Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.529
          Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.673
           Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.378
            Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.594
             Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.623
              Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.352
               Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.721
                Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.817
                coco eval results saved to ./output/pvalite_b5_bn/results/detections_test-dev2015_results.pkl
                All done!
