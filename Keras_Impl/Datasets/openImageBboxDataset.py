import Keras_Impl.Datasets.abstractDataset as ds
import numpy as np
# Open Images Dataset V5
#
# Dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images.
import pandas as pd
import os
from tqdm import tqdm

class OpenImageBboxDataset(ds.Dataset):

    def __init__(self, path_to_bounding_boxes_sheet="D:/Users/T0227964-A/Documents/BBox_Dataset/train-annotations-bbox.csv", path_to_image_files="D:/Users/T0227964-A/Documents/BBox_Dataset/train_0.tar/train_0"):
        super().__init__(self)
        self.isFlatten = False
        print("\tChosen Dataset: OpenImageBboxDataset")
        print("Warning: This dataset does not load directly the data. instead, it provides a list of string links to the data that will need to be loaded after.")
        print("Therefore, it suits nicely any kind of generator (with yielded outputs)")
        
        self.path_to_bounding_boxes_sheet = path_to_bounding_boxes_sheet
        self.path_to_image_files = path_to_image_files

        self.x_train = []
        self.x_test = []
        self.y_train = {}
        self.y_test = []

        print("Opening bbox csv file...")
        csvfile = pd.read_csv(path_to_bounding_boxes_sheet)
        print("File successfully opened !")

        imageIds = np.array(list(map(lambda x:x.split(".jpg")[0],os.listdir(path_to_image_files))))
        print('Linking images with corresponding bounding boxes..')
        size = len(imageIds)
        for i in tqdm(range(size)):
            imageId = imageIds[i]
            rows = csvfile[(csvfile.ImageID == imageId)][['LabelName', 'XMin', 'XMax', 'YMin', 'YMax']]
            if (len(rows) != 0):
                self.x_train.append(imageId)
                category, gt_boxes = self.rowToBoxData(rows.iterrows())
                self.y_train[imageId] = gt_boxes

    
    def rowToBoxData(self, rows):
        category = []
        
        xmin=[]
        ymin=[]
        xmax=[]
        ymax=[]

        for row in rows:
            category.append(row[1][0])
            xmin.append(row[1][1])
            ymin.append(row[1][2])
            xmax.append(row[1][3])
            ymax.append(row[1][4])

        gt_boxes=[list(box) for box in zip(xmin,ymin,xmax,ymax)]

        return category, np.asarray(gt_boxes, np.float)

    def getTrainingData(self):
        return (self.x_train, self.y_train)

    def getTestingData(self):
        return (self.x_test, self.y_test)
    
    def getInputShape(self):
        return (None, None, 1536)

    def getNumClasses(self):
        return 1