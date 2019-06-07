import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.initializers as KI
import keras.engine as KE
import keras.models as KM
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from Keras_Impl.Networks import abstractModel as AM 
import numpy as np
from Keras_Impl.Networks.RPN.utils import *

BATCH_SIZE=512

NUM_OF_EPOCHS = 100

VALIDATION_DATA = 0.2

NUM_OF_ANCHORS = 9 # number of anchors, it is linked with the scales of anchors so please modify both at the same time

SCALE_OF_ANCHORS = [3, 6, 12] # defines the scale of the 3 set of anchors : it corresponds to the number of tiles in the horizontal side of the 1:1 anchor

BG_FG_RATIO = 2 # Constant value that defines the maximum ratio between bg (0) and fg (1) training samples


class RPNModel(AM.Model):
    """
    """
    def __init__(self):
        super().__init__(self)
        self.model = None
        # uncomment if you don't have a proxy
        # self.pretrained_model = InceptionResNetV2(include_top=False, weights='imagenet')
        self.pretrained_model = InceptionResNetV2(include_top=False, weights='D:/Users/T0227964-A/.keras/models/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5')

    def createModel(self, input_shape = (None,None,1536), num_classes = None):

        k = NUM_OF_ANCHORS

        feature_map_tile = KL.Input(shape=input_shape)


        convolution_3x3 = KL.Conv2D(
            filters=512,
            kernel_size=(3, 3),
            name="3x3"
        )(feature_map_tile)

        # First output used to refine the borders of the k anchors of each given tile. It is stored as a 4-sized vector.
        output_deltas = KL.Conv2D(
            filters= 4 * k,
            kernel_size=(1, 1),
            activation="linear",
            kernel_initializer="uniform",
            name="deltas1"
        )(convolution_3x3)

        # Second output used to say if the anchor of each given tile contains a bg object or a foreground object. It is stored as a Integer value
        output_scores = KL.Conv2D(
            filters=1 * k,
            kernel_size=(1, 1),
            activation="sigmoid",
            kernel_initializer="uniform",
            name="scores1"
        )(convolution_3x3)

        self.model = KM.Model(inputs=[feature_map_tile], outputs=[output_scores, output_deltas])

        # TODO: maybe adapt loss function to the actual formula of the paper
        self.model.compile(optimizer='adam', loss={'scores1':'binary_crossentropy', 'deltas1':'mse'})

    def tiling_input_image(self, img, gt_boxes):
        img_width=np.shape(img)[1]
        img_height=np.shape(img)[0]
        # TODO: initialiser gt_boxes (format (k, 4) avec k le nombres de boxes dans l'image)

        ## First objective: We need to get all the possible anchors to feed our rpn network with the correct data for each of them

        # We get the feature map from the pre-trained model to feed it into the RPN model
        feature_map = self.pretrained_model.predict(x)

        height = np.shape(feature_map)[1]
        width = np.shape(feature_map)[2]
        num_feature_map = width*height

        # We initialize the size of the stride based on the size of the original image and the size of the feature_map 
        w_stride = img_width / width
        h_stride = img_height / height

        # we split the original_image into a grid of height*width tiles of individual sizes of (h_stride, w_stride)
        shift_x = np.arange(0, width) * w_stride
        shift_y = np.arange(0, height) * h_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(),shift_y.ravel())).transpose()
        
        # we generate all the possible anchors from the tiles created earlier
        base_anchors=generate_anchors(w_stride,h_stride,scale=np.asrray(SCALE_OF_ANCHORS))
        all_anchors = (base_anchors.reshape((1, 9, 4)) +
            shifts.reshape((1, num_feature_map, 4)).transpose((1, 0, 2)))

        # we delete all the anchors with pixels outside of the image 
        border=0
        inds_inside = np.where(
                (all_anchors[:, 0] >= -border) &
                (all_anchors[:, 1] >= -border) &
                (all_anchors[:, 2] < img_width+border ) &  # width
                (all_anchors[:, 3] < img_height+border)    # height
        )[0]
        anchors=all_anchors[inds_inside]

        ## Second objective : now that we have all our potential anchors candidates, we will need to generate their suposed predicted output values
        ## In the original paper, an anchor is said to have a positive output if : 
        ## "(i) the anchor/anchors with the highest Intersection-overUnion (IoU) overlap with a ground-truth box, or 
        ##  (ii) an anchor that has an IoU overlap higher than 0.7 with any gt boxes"

        overlaps = bbox_overlaps(anchors, gt_boxes)
        # for each anchors, find the gt box with biggest overlap
        # and the overlap ratio. therefore, result has shape (len(anchors),)
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        # for each gt boxes, find the anchor with biggest overlap,
        # and the overlap ratio. therefore, result has shape (len(gt_boxes),)
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                    np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
        #labels, 1=fg/0=bg/-1=ignore
        labels = np.empty((len(inds_inside), ), dtype=np.float32)
        # we set the default value of labels at -1 and change the value of all ground_truth maximum overlaping anchors
        labels.fill(-1)
        labels[gt_argmax_overlaps] = 1
        # we set every anchor with an overlaping ratio over 0.7 (as specified in the paper) at 1, those lower than .3 at 0. those in between are ignored
        labels[max_overlaps >= .7] = 1
        labels[max_overlaps <= .3] = 0
        # to be sure to respect the desired batch size of 256 elements, we sample the positive (0 or 1) labels that are potentialy more numerosous than 256.

        fg_inds = np.where(labels == 1)[0]
        num_max_bg = int(len(fg_inds) * BG_FG_RATIO)
        bg_inds = np.where(labels == 0)[0]
        # If there is more bg samples than specified by- the bg_fg_ratio, then we subsample it randomly by ignoring (setting their label to -1) a certain amount of them
        if len(bg_inds) > num_max_bg:
            disable_inds = npr.choice(
                bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1

        # we prepare the batch of data by assigning to each feature map point an index of the form of integer. This index specifies to which tiles the anchor is supposed to be working on
        # For instance, if we get an index equals to 20 and our feature maps is a 14*9, then it means it corresponds to the tiles of index (1, 6) (2nd row, 7th column) and therefore to the point of the' feature maps of same index
        # /!\ This process is HIGHLY dependant on the supposed order the anchors in the original algorithm of anchor generation. The anchors need to be created in order, 9 at a time, for each pixel from to top-left one to the bottom-right one.
        batch_inds=inds_inside[labels!=-1]
        batch_inds=(batch_inds / k).astype(np.int)

        # generate batch feature map 3x3 tile from batch_inds
        padded_fcmap=np.pad(feature_map,((0,0),(1,1),(1,1),(0,0)),mode='constant')
        padded_fcmap=np.squeeze(padded_fcmap)
        batch_tiles=[]
        for ind in batch_inds:
            x = ind % width
            y = int(ind/width)
            fc_3x3=padded_fcmap[y:y+3,x:x+3,:]
            batch_tiles.append(fc_3x3)

        # We create the variables that will hold the the target label batch of shape (batch_size, 1, 1, 1*9) for k = 9
        full_labels = unmap(labels, total_anchors, inds_inside, fill=-1)
        batch_label_targets=full_labels.reshape(-1,1,1,1*k)[batch_inds]

                    # We create the variables that will hold the the target bbox batch of shape (batch_size, 1, 1, 4*9) for k = 9
        bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
        pos_anchors=all_anchors[inds_inside[labels==1]]
        bbox_targets = bbox_transform(pos_anchors, gt_boxes[argmax_overlaps, :][labels==1])
        bbox_targets = unmap(bbox_targets, total_anchors, inds_inside[labels==1], fill=0)
        batch_bbox_targets = bbox_targets.reshape(-1,1,1,4*k)[batch_inds]

        return np.asarray(batch_tiles), batch_label_targets.tolist(), batch_bbox_targets.tolist()


    def trainModel(self, train_dataset):

        k = NUM_OF_ANCHORS

        (x_tmp, y_tmp) = train_dataset

        def input_generator():

            batch_tiles=[]
            batch_labels=[]
            batch_bboxes=[]

            count=0

            while 1:
                for x in x_tmp:
                    img = load_img(x)
                    gt_boxes = y_tmp[img]

                    tiles, labels, bboxes = self.tiling_input_image(img, gt_boxes)
                    for i in range(len(tiles)):

                        batch_tiles.append(tiles[i])
                        batch_labels.append(labels[i])
                        batch_bboxes.append(bboxes[i])

                        if(len(batch_tiles)==BATCH_SIZE):
                            a=np.asarray(batch_tiles)
                            b=np.asarray(batch_labels)
                            c=np.asarray(batch_bboxes)
                            if not a.any() or not b.any() or not c.any():
                                print("empty array found.")

                            yield a, [b, c]
                            batch_tiles=[]
                            batch_labels=[]
                            batch_bboxes=[]
                    
                    
        model.fit_generator(input_generator(), epochs=NUM_OF_EPOCHS)
    
    def evaluateModel(self, test_dataset):
        #TODO
        return []

    def getName(self):
        return 'Region Proposal Network'