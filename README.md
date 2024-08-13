# Chest X-Ray Pneumonia Classification using Convolutional Neural Networks

## Table of Contents
  1. [Introduction](#introduction)<br>
  2. [Datasets](#datasets)<br>
  3. [Exploratory Data Analysis](#eda)<br>
  4. [Transform Data](#transform)<br>
  5. [Model](#neural-network)<br>
  6. [Training](#training)<br>
  7. [Evaluation](#model-evaluation)<br>
  8. [Prediction and Visualization](#predict-visualize)<br>
  9. [Summary](#summary)<br>

## <a name="introduction"> Introduction</a>
This project focuses on fine-tuning a pre-trained ResNet-18 model to accurately classify chest X-ray images for the presence of pneumonia. The objective is to leverage the power of transfer learning to develop a model capable of distinguishing between normal and pneumonia-affected X-ray images. The project is organized into several key sections: data loading, data preprocessing, model architecture modification, model training, evaluation, and result visualization.

## <a name="datasets"> Datasets</a>
The project begins by loading the dataset using the "keremberke/chest-xray-classification" dataset provided through the "datasets" module from Huggingface Community. The dataset comprises training, validation, and test sets of X-ray images with a size of 640x640 pixels. The dataset structure is stored in the variables X and y, where X represents the training data, and y represents the validation data.


## <a name="eda"> Exploratory Data Analysis</a>
Dataset Statistics:

- Training Set Size: 4077 samples
- Validation Set Size: 1165 samples
- Test Set Size: 582 samples

![EDA Image 1](imgs/label-distribution.png)

It's worth noting that the class distributions in the dataset are not perfectly balanced. However, through testing and monitoring, it has been observed that this class imbalance may not significantly impact the model's performance and final output.


## <a name="transform"> Transform Data</a>
Before training the ResNet-18 model, the X-ray images need to be appropriately transformed to fit the model's requirements. The following transformations are applied to the dataset:

1. Custom Dataset: A custom dataset class is utilized to load and transform the images, ensuring they meet the input requirements of the model.
2. Grayscale to RGB Conversion: Since ResNet-18 expects RGB images, the grayscale X-ray images are converted to RGB by repeating the single grayscale channel.
3. Resizing: All images are resized to 224x224 pixels, the standard input size for ResNet-18.
4. Normalization: The images are normalized using ImageNet mean and standard deviation values to stabilize and improve the training process.
5. DataLoader Setup: The dataset is split into training, validation, and test sets, with DataLoader objects created to handle batch processing during model training and evaluation.

These are some samples of the dataset (training set) after transformation:
![EDA Image 2](imgs/visualize-data.png)


## <a name="model"> Model (Resnet18 Pre-trained)</a>
In this project, we fine-tune a pre-trained ResNet-18 model to classify chest X-ray images as either normal or pneumonia-affected. ResNet-18 is a popular deep convolutional neural network architecture known for its residual blocks, which help in training very deep networks by mitigating the vanishing gradient problem.

<b>ResNet-18 Architecture</b>
Below is a visual representation of the ResNet-18 architecture:
![Resnet18](imgs/resnet-18.png)
The model consists of several convolutional layers, each followed by a batch normalization layer and a ReLU activation function. The layers are organized into blocks, with shortcut connections that add the input of each block to its output, aiding in the flow of gradients during training.

<b>Model Customization</b>

1. Loading the pre-trained ResNet-18 Model: We start by loading a ResNet-18 model pre-trained on ImageNet.
```python
resnet18 = models.resnet18(pretrained=True)
```

2. Adjusting the Output Layer: The final fully connected layer is modified to output two classes, suitable for the binary classification task.
```python
num_features = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_features, 2)
```

3. Freezing Layers
In the ResNet-18 architecture, we freeze the initial layers up to the layer3 block to use the model as a feature extractor. Only the deeper layers (layer4 and fc) are left unfrozen to allow fine-tuning on the pneumonia dataset.
```python
# Freeze the initial layers to use ResNet-18 purely as a feature extractor
for param in resnet18.parameters():
    param.requires_grad = False

# Unfreeze the last few layers
for param in resnet18.layer4.parameters():
    param.requires_grad = True
```

These adjustments allow the ResNet-18 model to effectively learn from the pneumonia dataset while leveraging its pre-trained weights for optimal performance.


## <a name="training"> Training</a>



## <a name="model-evaluation"> Model Evaluation</a>
During model training, the following performance metrics were achieved:

- Training Loss: 0.0728
- Training Accuracy: 0.9738
- Validation Loss: 0.1189
- Validation Accuracy: 0.9614

![evaluate image](imgs/train-val-loss-acc.png)

These metrics indicate the model's performance on both the training and validation datasets. The training loss and accuracy reflect how well the model learned from the training data, while the validation loss and accuracy provide insights into its generalization capabilities.


## <a name="predict-visualize"> Predictions on Test Data</a>
The model's performance on the test data is summarized as follows:
- Test loss: 0.14
- Test accuracy: 0.95

![testing image 1](imgs/confusion-matrix.png)

The confusion matrix provides a detailed breakdown of the model's predictions, highlighting true positives, true negatives, false positives, and false negatives, which is crucial for assessing the model's performance in binary classification tasks.

These are some visualized predictions of the model:

![testing image 2](imgs/test-predictions.png)


## <a name="summary"> Summary</a>
In this project, we have successfully developed a Convolutional Neural Network (CNN) for the classification of chest X-ray images to detect pneumonia. The model demonstrated strong performance, with a high accuracy of 95% on the test data. This achievement underscores the potential of CNNs in aiding medical professionals with early pneumonia diagnosis based on chest X-ray images. The combination of effective data preprocessing, model training, and evaluation techniques has resulted in a powerful tool for medical image analysis and diagnosis.

