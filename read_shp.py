# import necessary packages
import sys
from osgeo import ogr
from osgeo import gdal
from osgeo import gdalconst
import numpy as np
import time
import pandas
import glob
import cv2

# load keras retinanet function
from keras_retinanet.models import load_model
from keras_retinanet.utils.image import preprocess_image, resize_image

# load tms function
from tms2geotiff import draw_tile

# load helper model function
from helpers import sliding_window
from helpers import pixel2coord
from helpers import non_max_suppression_fast

# configuration for TMS process
TMS = 'http://mt0.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}'
ZOOM = 19
OUTPUT_DIR = 'D:/2022/Riset Sawit Tahun II/Spasial/IDENTIFICATION/GOOGLE/'
OUTPUT_FILE_NAME = 'temp.tiff'
SHP_FILE = r'D:/2022/Riset Sawit Tahun II/Spasial/SHP/GRID/index_rbi_5k.shp'

# configuration for identification model
(winW, winH, stepSize) = (500, 500, 450)
scorethreshold = 0.5
iouthreshold = 0.5
path = "D:/2022/Riset Sawit Tahun II/Spasial/IDENTIFICATION/GOOGLE/"
model_name = "GOOGLE05-101-all2"
backbone_name = "resnet101"

# temporary variable
all_bboxeses = []

#load model
model = load_model('models/infer-' + model_name + '-model.h5', backbone_name=backbone_name)

# load data
ds = ogr.Open(SHP_FILE, 0)
if ds is None:
    sys.exit('Could not open {0}.'.format(fn))

lyr = ds.GetLayer(0)
num_features = lyr.GetFeatureCount()
print("Number of feature: {0}".format(num_features))

class minmax:
    def __init__(self, minimum, maximum):
        self.minimum = minimum
        self.maximum = maximum

def detect_palm(x, y, winW, winH, minmaxlist):
    # crop image
    a_image = rs.ReadAsArray(x, y, winW, winH)
    datatype = rs.GetRasterBand(1).DataType

    normalizedImg = []

    if not datatype == 1:
        for i in range(3):
            band_255 = map_uint16_to_uint8(a_image[i], int(minmaxlist[i].minimum), int(minmaxlist[i].maximum))
            normalizedImg.append(band_255)

        normalizedImg = np.dstack((normalizedImg[2], normalizedImg[1], normalizedImg[0]))

    else:
        normalizedImg = np.dstack(((a_image[2],a_image[1],a_image[0])))

    # preprocess image for network
    image = preprocess_image(normalizedImg)
    image, scale = resize_image(image)

    # process image
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

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

i = 0
for feat in lyr:
    geom = feat.geometry()
    name = feat.GetField('PageName')
    extent = geom.GetEnvelope()
    print("Grid Name: {0}".format(name))
    draw_tile(TMS, extent[3], extent[1], extent[2], extent[0], ZOOM, OUTPUT_DIR + OUTPUT_FILE_NAME)

    rs = gdal.Open(OUTPUT_DIR + OUTPUT_FILE_NAME)
    width = rs.RasterXSize
    height = rs.RasterYSize

    bboxes = []
    x_list = []
    y_list = []
    minmaxlist = []

    for bandId in range(rs.RasterCount):
        bandId = bandId + 1
        band = rs.GetRasterBand(bandId)
        band_arr_tmp = band.ReadAsArray()
        min, max = np.percentile(band_arr_tmp, (2, 98))
        minmaxlist.append( minmax(min, max) )

    for (x, y) in sliding_window(width, height, stepSize):

        # Stop sliding windows if widows end
        if x + winW > width and y + winH > height:
            xl = width - winW
            yl = height - winH
            detect_palm(xl, yl, winW, winH, minmaxlist)
            continue

        if x + winW > width:
            xl = width - winW
            detect_palm(xl, y, winW, winH, minmaxlist)
            continue

        if y + winH > height:
            yl = height - winH
            detect_palm(x, yl, winW, winH, minmaxlist)
            continue

        detect_palm(x, y, winW, winH, minmaxlist)

    bboxes = np.array(bboxes, dtype=np.float64)
    print("Detected Bounding Box: {}".format(len(bboxes)))

    new_boxes = non_max_suppression_fast(bboxes, iouthreshold)
    print("Total Bounding Box after NMS: {}".format(len(new_boxes)))

    if len(new_boxes) > 0:
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
            (coor_x, coor_y)  = pixel2coord(rs, xc, yc)

            x_list.append(coor_x)
            y_list.append(coor_y)

        df = pandas.DataFrame(data={"x": x_list, "y": y_list})
        df.to_csv(path + name + "_" + model_name + "_" + str(scorethreshold) +".csv", sep=',',index=False)
    else:
        print('No object detected on feature name: {}.'.format(name))

    del rs

del ds
