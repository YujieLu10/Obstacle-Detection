import os
import cv2

cnt = 0
for filename in os.listdir(r"/home/luyujie/SNIPER/data/coco/images/val2014"):
    cnt = cnt + 1
    #print filename
    filepath = os.path.join('/home/luyujie/SNIPER/data/coco/images/val2014', filename)
    if cnt % 1000 == 0:
        print cnt
    image = cv2.imread(filepath)
    res = cv2.resize(image, (640, 400), interpolation=cv2.INTER_CUBIC)
    newpath = os.path.join('/home/luyujie/SNIPER/data/coco/images/resval', filename)
    cv2.imwrite(newpath, res)
    
