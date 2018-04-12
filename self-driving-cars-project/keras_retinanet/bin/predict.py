from keras_retinanet.models.resnet import custom_objects
from glob import glob
import sys
import keras
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
import numpy as np
from keras_retinanet import layers

import json

path_to_model = sys.argv[1]


model = keras.models.load_model(path_to_model, custom_objects=custom_objects)

#transfer to predict model
classification   = model.outputs[1]
detections       = model.outputs[2]
boxes            = keras.layers.Lambda(lambda x: x[:, :, :4], name='lambda_before_output')(detections)
detections       = layers.NonMaximumSuppression(name='nms')([boxes, classification, detections])
prediction_model = keras.models.Model(inputs=model.inputs, outputs=model.outputs[:2] + [detections])

model = prediction_model

test_files = glob('deploy/test/*/*_image.jpg')

result_dict = {}

#01bb984a-d3f1-4ba5-b0c0-5242533efa4d/0000

count = 0
all_num = len(test_files)

for test_file in test_files:
    
    print(test_file)
    name_parts = test_file.split('/')

    folder_name = name_parts[2]
    test_sample_name = name_parts[3].split('_')[0]

    sample_key = folder_name+'/'+test_sample_name
    
    raw_input_img = read_image_bgr(test_file)
    #print(raw_input_img.shape)

    (image, scale) = resize_image(preprocess_image(read_image_bgr(test_file)))
    #print(image.shape)
    image_batch = np.zeros((1,) + image.shape, dtype=keras.backend.floatx())
    image_batch[0, :image.shape[0], :image.shape[1], :image.shape[2]] = image

    _, _, detections = model.predict_on_batch(image_batch)

    box_num = detections.shape[1]

    possible_result = []

    possible_idx = np.argwhere(detections[0,:,4]>0.1)
    for i in possible_idx:
        current_result = []
        for j in range(4):
            current_result.append(np.float64(detections[0,i[0],j]/scale))
        current_result.append(np.float64(detections[0,i[0],4]))
        possible_result.append(current_result)

    print(possible_result)

    result_dict[sample_key] = possible_result
    
    count += 1
    print("%d/%d finished."%(count, all_num))

with open('test_result.txt', 'w') as f:
    f.write(json.dumps(result_dict))
