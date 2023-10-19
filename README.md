# Chest X-Ray Pneumonia Classification using Convolutional Neural Networks

## Table of Contents
  1. [Introduction](#introduction)<br>
  2. [Datasets](#datasets)<br>
  3. [Preprocess Data](#preprocess)<br>
  4. [Visualize Data](#visualize)<br>
  5. [Neural Network](#neural-network)<br>
  6. [Logistic Regression](#logistic-regression)<br>
  7. [Support Vector Machines (SVM)](#svm)<br>
  8. [Testing](#testing)<br>
  9. [Summary](#summary)

## <a name="introduction"> Introduction</a>
This project aims to develop a Convolutional Neural Network (CNN) for accurately classifying chest X-ray images for the presence of pneumonia. The goal is to build a model capable of distinguishing between normal and pneumonia-affected X-ray images. The project is divided into various key sections, including data loading, data preprocessing, neural network architecture, model training, evaluation, and result visualization.

## <a name="datasets"> Datasets</a>
The project begins by loading the dataset using the "keremberke/chest-xray-classification" dataset provided through the "datasets" module from Huggingface Community. The dataset comprises training, validation, and test sets. The dataset structure is stored in the variables X and y, where X represents the training data, and y represents the validation data.

## <a name="preprocess"> Preprocess Data</a>
Data preprocessing is a critical step before training the CNN model. The following steps are performed to preprocess the X-ray images:
1. Open and resize the image to 28x28 pixels.
2. Convert the image to grayscale.
3. Convert the image to a NumPy array and reshape it to (28, 28, 1).
4. Normalize pixel values to the range [0, 1].
The preprocessed data is then divided into three sets: training, validation, and test sets. Images and their corresponding labels are stored in separate arrays.



## <a name="visualize"> Visualize Data</a>
A function named visualize_random_data is defined to visualize a random subset of the training data. This function displays a specified number of images along with their labels.

## <a name="neural-network"> Neural Network</a>
A CNN model is defined using TensorFlow's Keras API. The model architecture includes convolutional layers, max-pooling layers, and fully connected (dense) layers. The model consists of several layers:

- Convolutional layers with ReLU activation.
- Max-pooling layers.
- Fully connected (dense) layers with ReLU activation.
- The output layer with a sigmoid activation function.

The model is compiled with a loss function, an optimizer, and accuracy as a monitoring metric.

## <a name="logistic-regression"> Logistic Regression
The Logistic Regression model is employed with optimized parameters obtained through grid search. The model is trained using the training set and subsequently tested on the test set. The model's performance is assessed based on accuracy, F1 score, and Jaccard score, providing insights into its effectiveness in phishing URL detection.
```python
logreg = LogisticRegression(C= 0.01, class_weight= 'balanced', max_iter= 500, penalty= 'l2', random_state=42)
lr_lbfgs = logreg.fit(x_train,y_train)

y_pred_lr=logreg.predict(x_test)
```
The accuracy, scores, and confusion matrix of the prediction are as follows:
- Accuracy: 0.9843
- F1 score: 0.9842
- Jaccard score: 0.9691

![EDA Image](imgs/confusion-matrix-lr.png)

## <a name="svm"> Support Vector Machines (SVM)</a>
In addition to Logistic Regression, Support Vector Machines (SVM) are used as an alternative machine learning model for phishing URL detection. The SVM model is trained and evaluated similarly to Logistic Regression, allowing for a comparison of their performance.
```python
clf = svm.SVC()
clf_md = clf.fit(x_train, y_train)

y_pred_svm = clf.predict(x_test)
```
The accuracy, scores, and confusion matrix of the prediction are as follows:
- Accuracy: 0.9968
- F1 score: 0.9968
- Jaccard score: 0.9936

![EDA Image](imgs/confusion-matrix-svm.png)

## <a name="testing"> Testing</a>
The trained models are put to the test using an external dataset containing URLs with their corresponding labels. Feature extraction is applied to this dataset, and both the Logistic Regression and SVM models are used to make predictions. The performance of the models on this test dataset is evaluated, providing an assessment of their real-world applicability.

The TEST accuracy, scores, and confusion matrix of the prediction by <strong>Logistic Regression</strong> model are as follows:
- Jaccard score: 0.05
- Accuracy: 0.49
- F1 score: 0.65

![EDA Image](imgs/confusion-matrix-lr-test.png)

The TEST accuracy, scores, and confusion matrix of the prediction by <strong>SVM</strong> model are as follows:
- Jaccard score: 0.05
- Accuracy: 0.49
- F1 score: 0.65

![EDA Image](imgs/confusion-matrix-svm-test.png)

## <a name="testing"> Summary</a>
In summary, this project combines data preprocessing, feature extraction, exploratory data analysis, and machine learning techniques to create an effective phishing URL detection model. The utilization of Logistic Regression and SVM, along with careful feature selection, ensures a robust and reliable solution for identifying potential phishing URLs.
