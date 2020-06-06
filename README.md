# :dress: Fashion Mnist  [![License](https://img.shields.io/badge/licence-MIT-blue)](https://choosealicense.com/licenses/mit/)

## Introduction

### Goal
Classify images of clothing from dataset of Zalando's articles ([source](https://github.com/zalandoresearch/fashion-mnist))

### Dataset
Fashion-MNIST dataset of Zalando's article images - replacement for the original MNIST dataset for benchmarking machine learning algorithms.
- training set (60,000 examples).
- test set (10,000 examples). 

Each example is a 28x28 grayscale image, associated with a label from 10 classes. 

![dataset_example](/app/knn/results/models/example__rand_img_bw.png)
(You can generate your plots by using function: `app.utils.data_utils.plot_rand_images`)

### Content

Each image has 784 pixels in total (28 pixels in height * 28 pixels in width).
Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel (higher numbers meaning darker).
Pixel-value is an integer between 0 and 255.
Each training and test example is assigned to one of the following labels:

| Number |  Label |
| - | ----------  |
| 0 | T-shirt/top |
| 1 | Trouser     |
| 2 | Pullover 	  |
| 3 | Dress       |
| 4 | Coat        |
| 5 | Sandal      |
| 6 | Shirt       |
| 7 | Sneaker     |
| 8 | Bag         |
| 9 | Ankle boot  |



## Used methods

Repository contains two machine learning algorithms, used to build image classification tool:

### KNN (K-Nearest Neighbor)

KNN is a non-parametric, classification algorithm. The basic idea behind KNN is simple. Given a (test) vector or image to classify label, find k vectors or images in training set that are “closest” to the (test) vector or image. With the k closest vectors or images, there are k labels. Assign the most frequent label of k labels to the (test) vector or image.

For KNN all algorithm methods, utilities for plotting images and manipulate of data, was written from scratch by myself (TensorFlow library is used here only to downloa training data)

**Data preprocessing**:
- splitting training data to "training" and "validation" set
- flatting training data from size (60000,(28,28)) to (60000,784)   
- data normalization (approximately normalizing data dimensions scale - dividing by 255)

**Examples**:

![normalisation_example](/app/knn/results/models/example__rand_img_bw.png)

**Used metrics to measure “closeness”**:

| Attempt_No | Algorithm | Max Accuracy |
| ---------- | --------- | ------------ | 
| 1 | Hamming Distance | 45% |
| 2 | Euclidean Distance (L2) | 84,77% |

KNN is an exception to general workflow for building/testing supervised machine learning models. There is no model build and becouse of that we don't have a training and validating set. All what we can do is selecting best k parameter and "closeness" parameter. To find it we can split our data to sth like "training" and "validation" set, which will be used in distance calculating method.

- <ins> Splitting proportion </ins>: 25%
- <ins> "Train" images qty <ins>: 45000
- <ins> "Validation" images qty <ins>: 15000

![k_search_log](/app/knn/results/models/example__rand_img_bw.png)

Finally we found best parameter **k=7**

Both to test our algorithm on test data and searching best k value, first we need to split data to batches (in our case each of size=2000/2500 images). KNN is very space consuming and without splitting, we would have to need sth about 15-25GB of free RAM memory to evaluate matrices calculations.    

### CNN (Convolutional Neural Networks)

CNN image classifications takes an input image, process it and classify it under certain categories. Deep learning CNN models to train and test, each input image will pass it through a series of convolution layers with filters (kernals), pooling, fully connected layers (FC) and apply Softmax function to classify an object with probabilistic values between 0 and 1

**Data preprocessing**:
- splitting training data to training and validation set
- resizing training data to 3D: from size (60000,(28,28)) to (60000,(28,28,1))  
- data normalization (normalizing the data dimensions, by dividing by 255, so that they are of considered approximately the same scale)
- data augmentation

**Data augmentation**:

Encompasses a wide range of techniques used to generate “new” training samples from the original ones by transforming data in various ways.

Used augmentations techniques:

| Attempt_No |  Transformations |
| ---------- | ---------------- |
| 1 | rotation_range=90, horizontal_flip, vertical_flip |
| 2 | rotation_range=5, horizontal_flip, vertical_flip, zoom_range=0.1 |

Example augmentated data:

![augmentated_data](/app/knn/results/models/example__rand_img_bw.png)
(You can generate your augmentiated images plots by using function: `app.utils.data_utils.plot_rand_images_from_gen`)

Validation set to training set proportions: 1/4

**Used models**: </br>

<ins>Model 1:</ins>

![model_1](/app/knn/results/models/example__rand_img_bw.png)

| Layers | Description |
| ------ | ----------- |
| Conv2D | 2D convolutional layer |
| MaxPooling2D | Max pooling operation for spatial data |
| Flatten | Flattens the input |
| Dense | Regural, fully-connected NN layer|

<ins>Model 2:</ins>

![model_2](/app/knn/results/models/example__rand_img_bw.png)

| Layers | Description |
| ------ | ----------- |
| Conv2D | 2D convolutional layer |
| MaxPooling2D | Max pooling operation for spatial data |
| Flatten | Flattens the input |
| Dropout | Reducing overfitting |
| Dense | Regural, fully-connected NN layer|

**Training attempts**:

| Attempt_no | Model_no | Batch size | Epochs | Augmentation_no | Time |
| ------ | ----------- | ----------- | ----------- | ----------- | ----------- |
| 1 | 1 | 64 | 150 | 1 | 3:28:06 |
| 2 | 2 | 64 | 120 | 2 | x |


## Results 

### KNN

<ins> Best result: </ins>

-**Accuracy: 84.77%**
-<ins> k: 7 </ins>
-Distance calculation method: euclidean distance (L2)
-Train images qty: 45000
-Total calculation time= 0:05:45
-Total k searching time= 0:18:51

![Benchmark](/app/knn/results/models/example__rand_img_bw.png)

As we can see, compared to benchmark our result is quite good, wheras relatively short training time

### CNN

**First attempt:**

<ins> Test name: </ins>  1_epoch150_batch64

- **Prediction accuracy: 84.61%** 
- Batch size: 64
- Epochs: 150
- Started data size qty: 45000
- Prediction loss: 0.44
- Total calculation time: 3:28:06

![Losses1](/app/knn/results/models/example__rand_img_bw.png)
![Accuracy1](/app/knn/results/models/example__rand_img_bw.png)

As we can see, futher increasing epochs value doesn't have sense, because our accuracy and losses are becaming more and more flat. We change our model and (knowing that test data are positioned straight) we reduce rotation value to 5 and add negligible zoom to augmentation) 

___
**Second attempt**:

<ins> Test name: </ins> 2_epoch120_batch64

- **Prediction accuracy: 88,6%** 
- Batch size: 64
- Epochs: 120
- Started data size qty: 45000
- Prediction loss: 0.33
- Total calculation time: 2:25:54

![Losses2](/app/knn/results/models/example__rand_img_bw.png)
![Accuracy2](/app/knn/results/models/example__rand_img_bw.png)

We achieve better results. Check if we can gain even more from this model by increasing batch size and slightly epochs too

___
**Third attempt:**

<ins> Test name: </ins> X 

- **Prediction accuracy: X%** 
- Batch size: x
- Epochs: x
- Started data size qty: 45000
- Prediction loss: X
- Total calculation time: X

![Losses3](/app/knn/results/models/example__rand_img_bw.png)
![Accuracy3](https://raw.githubusercontent.com/Ukasz09/Clothing-recognition/master/app/knn/results/models/example__rand_img_bw.png)

<results summary>

## Usage

### Software requirements
`python 3.+, tensorFlow, keras, numpy, matplotlib, IPython, skit-learn`

### How to use it?
You just need to simply run `cnn_main.py` or `knn_main.py` according to algorithm method which you want to test. You don't need to download Fashion_MNIST data - it is done automatically by TensorFlow library. 

```bash
python3 cnn_main.py
```
or
```bash
python3 knn_main.py
```
### Expected output steps for main files

**CNN_main**:
- making, compiling, fitting and measuring accuracy for model with default parameters, model and data 
  (you can change epochs, batch size or model as you want)
- plotting graph with history of training accuracy and losses
- plotting example training images
- plotting example images with predictions

**KNN_main:**
- searching best k value
- making predictions for test data and calculating accuracy for k which had been found step before 
- plotting example training images
- plotting example images with predictions

Both in `cnn` and `knn` folder:
- All plots are saved to `.png` files inside `results\models`.
- Logs are written to `.txt` files inside `results\logs` directory. 

___
## 📫 Contact 
Created by <br/>
<a href="https://github.com/Ukasz09" target="_blank"><img src="https://avatars0.githubusercontent.com/u/44710226?s=460&v=4"  width="100px;"></a>
<br/> gajerski.lukasz@gmail.com - feel free to contact me! ✊
