import sys
import json
import cv2
import os
import shutil

path = '/home/luyujie/imid_anno'
boxcnt = 0
for filename in os.listdir(r"/home/luyujie/imid_anno"):
    filepath = os.path.join(path, filename)
    with open(filepath) as f:
        for line in f:
            boxcnt += 1

print boxcnt