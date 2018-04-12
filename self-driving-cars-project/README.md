# Open Source Repo

We developed the network based on Keras RetinaNet

https://github.com/fizyr/keras-retinanet

# Installation

Clone this repository.

In the repository, execute `python setup.py install --user`.

# Preprocessing

Put all data into the repo (deploy/trainval deploy/test)

Execute `python lidar.py` to generate lidar image for all lidar clouds.

Execute `python result.py` to generate labeling file for training data. (as [retinanet_label.csv])

Create a file named [retinanet_class.csv] with a single line: `car,0`

# Training

Execute `python keras-retinanet/bin/train.py --epochs {# of epoch} --steps {# of steps in one epoch} csv retinanet_label.csv retinanet_class.csv`

All pretrain data will be automatically downloaded as in keras-retinanet

Model will be saved as `original_RGB_model.h5`

# Predict

Execute `python keras-retinanet/bin/predict.py original_RGB_model.h5` to load trained model and predict the original output as `test_result.txt`

Then execute `python test_output_handle_T1.py test_result.txt 0.4` to get the standard result for kaggle submission

## Team 13: DRIVERUNKNOWN'S RACINGGROUNDS