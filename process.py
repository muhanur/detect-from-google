import numpy as np
import time
import pandas

from keras_retinanet.models import load_model
from keras_retinanet.utils.image import preprocess_image, resize_image

from osgeo import gdal

from helpers import sliding_window
from helpers import pixel2coord
from helpers import non_max_suppression_fast

(winW, winH, stepSize) = (500, 500, 400)
scorethreshold = 0.5
iouthreshold = 0.5
file = "D:/Muaro Jambi/GOOGLE/Resize"
model_name = "GOOGLE05-101-all2"
model = load_model('export/infer-' + model_name + '-model.h5', backbone_name='resnet101')

ds = gdal.Open(file + ".tif")
width = ds.RasterXSize
height = ds.RasterYSize

bboxes = []
x_list = []
y_list = []

for (x, y) in sliding_window(width, height, stepSize, windowSize=(winW, winH)):
    
    st = time.time()
    
    # Stop sliding windows if widows end
    if x + winH > width or y + winW > height:
        continue
    
    # crop image
    a_image = ds.ReadAsArray(x,y,winW,winH)
    crop = np.dstack((a_image[0],a_image[1],a_image[2]))
    
    # preprocess image for network
    image = preprocess_image(crop)
    image, scale = resize_image(image)
    
    # process image
    boxes, scores, labels = model.predict(np.expand_dims(image, axis=0))
    
    # correct for image scale
    boxes /= scale
    
    # select indices which have a score above the threshold
    indices = np.where(scores[0, :] >= scorethreshold)[0]
    
    # select those scores
    scores = scores[0][indices]
    
    # find the order with which to sort the scores
    scores_sort = np.argsort(-scores)
    
    # select detections
    image_boxes = boxes[0, indices[scores_sort], :]
    
    for i in indices:
        b = np.array(image_boxes[i,:]).astype(int)
        x1 = b[0] + x
        y1 = b[1] + y
        x2 = b[2] + x
        y2 = b[3] + y
        
        bboxes.append([x1, y1, x2, y2])
    
    print('Elapsed time = {}'.format(time.time() - st))
    
bboxes = np.array(bboxes, dtype=np.float32)

print('Non max suppression all detected box')

# non max suppression on overlay bboxes
new_boxes = non_max_suppression_fast(bboxes, iouthreshold)

print('Creating point from bbox')

for jk in range(new_boxes.shape[0]):
    
    b = np.array(new_boxes[jk,:]).astype(int)
    
    x1 = b[0]
    y1 = b[1]
    x2 = b[2]
    y2 = b[3]
    
    # Centroid
    xc  = (x1 + x2) / 2
    yc  = (y1 + y2) / 2
    
    # get geo coordinate
    (coor_x, coor_y)  = pixel2coord(ds, xc, yc)
    
    x_list.append(coor_x)
    y_list.append(coor_y)
    
df = pandas.DataFrame(data={"x": x_list, "y": y_list})
df.to_csv(file + "_" + model_name + "_" + str(scorethreshold) +".csv", sep=',',index=False)