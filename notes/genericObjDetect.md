2.Background
Accuracy related challenges
1) the vast range of intraclass variations
intrinsic factors : different instances in one catefory may vary in color, texture, material, shape, and size
imaging conditions : different instances, times, locations, weather conditions, cameras, backgrounds, illuminations, viewpoints, and viewing distances. -> illumination, pose, scale, occlusion, background clutter, shding, blur and motion. further challenges : digitization artifacts, noise corruption, poor resolution, and filtering distortions

2) the huge number of object categories : demands great discriminain power of the detector to distinguish between subtly different interclass variations -> interclass similarities

Efficiency related challenges
limited computational capabilities and storage space
large number of object categories
large number of possible locations and scales
scalability
unseen objects, unknown situations, and rapidly increasing image data

3.Frameworks
Milestone Detectors Category
A. Two stage detection framework : includes a pre-processing step for region proposal
B. One stage detection framework : region proposal free framework
RCNN:Region-based Convolutional Neural Networks
Fast RCNN : significantly sped up the detection process; still relies on external region proposals -> bottleneck region proposal computation

CNNs have a remarkable ability to localize objects in CONV layers -> weakened in the FC layers

Faster RCNN
RPN : generating region proposals
Fast RCNN : region classification
share large number of convolutional layers

RPN : a kind of Fully Convolutional Network(FCN)
Faster RCNN : a purely CNN based framework without using handcrafted features
