## visual object detection
The task is to find a location of a phone dropped on the floor from a single RGB camera image.

## Getting Started 
Use the following instructions to make your system ready to run the code.

## Dependencies
1. Windows 10
2. Python 3.6.12
3. TensorFlow 1.13.1
4. Keras 2.2.4

## Installation
- A requirements.txt is added to the package which can be used to install the dependencies using the following code in your virtual environment.

```
pip install -r requirements.txt
```

## Inside the package
- The repository contains python scripts to train and test the model. This repo contains weights of the model created using the jupyter notebook along with visualizations of the results.

## Training Script
- train_phone_finder.py : takes a single command line argument which is a path to the folder with labeled images and labels.txt

```
python train_phone_finder.py find_phone
```

- Before testing the model, change the path of the weights in find_phone.py.  

## Testing Script
- find_phone.py : takes a single command line argument which is a path to the jpeg image to be tested.

```
 python find_phone.py find_phone/108.jpg
```
## Preprocessing of the training data for data augmentation
1. Resizing image to 128 x 128 
2. Incrase the brightness and contrast of the image
3. Use Gaussian Filter
4. Use Median Filter
5. Flipping of the images
6. Normalization of the images

## Steps to improve the performance
1. Gather more data
2. The model can be improved with a better architecture.
3. This problem can also be solved using Saliency detection algorithm which would locate the salient part of the image from the background. In this case the salient part of the image would be the phone.