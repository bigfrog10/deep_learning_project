dataset from: https://cocodataset.org/#download

cv selective search
segmentation -> pick out masked part of image
supercategories -> ID for subcategory
- drawbox gives label, box coordinates, and prediction probability
  - connects into image segmentation
category distribution of all the classes, can create a pie chart as well
item segmentation with a key 

- creating a binary mask of key item + background
- create rgb mask to separate different items in different classes
  - can overlay the masks over

### evaluation indicators in target detection
- IoU (intersection over union) ratio
  - IoU = (A∩B) / (A∪B)
  - there is generally a threshold for the determination of detection frame

refer to [confusion matrix](task1/confusion_matrix.png) to understand how to calculate TP, TN, FP and FN
- accuracy
  - most common evaluation indicator
  - correctly classified samples/total number of samples
  - accuracy = (TP+TN)/(TP+TN+FP+FN)
- precision
  - proportion of correctly found positives
  - precision = TP/( TP+FP)
- recall rate
  - recall rate = TPR (true positive rate) = number of positive samples retrieved among total positive samples
  - TPF = TP/(TP+FN)
- FPR (false positive rate)
  - proportion of actual negative examples incorrectly judged as possible
    - smaller the better
    - FPR = FP/(FP+TN)
- f1-score
  - harmonic mean of precision and recall with a max of 1 and min of 0
  - F1 = 2TP/(2TP+FP+FN)
  - Fβ = ((1+β*β)*precision*recall) / (β*β*precision + recall)
  - F2 and F0.5 are the second most commonly used 
- PR
  - precision/recall curve, the higher precision + recall rate, the better performance
  - area under the curve is the AP (average precision), the larger the higher accuracy
- ROC
  - receiver operating characteristic curve
  - when the TPR is larger and FPR is smaller, the classification result is better
- AP value
- AUC value
- mAP
  - mean average precision refers to entire dataset 