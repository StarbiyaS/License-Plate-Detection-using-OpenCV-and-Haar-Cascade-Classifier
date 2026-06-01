# License Plate Detection using OpenCV and Haar Cascade Classifier
### Name: Starbiya S
### Register Number: 212223040208
## Aim
To implement a License Plate Detection system using OpenCV and Haar Cascade Classifier, draw bounding boxes, crop the detected region, and blur the license plate to improve privacy. The detection accuracy is improved by tuning Haar Cascade parameters.

## Software Used
1.Python 3.7 or above  
2.OpenCV (opencv-python)  
3.NumPy  
4.Matplotlib  
5.Jupyter Notebook (Anaconda)  
6.Haar Cascade File: haarcascade_russian_plate_number.xml

## Algorithm
1.Import necessary libraries such as OpenCV and Matplotlib  
2.Read the input vehicle image  
3.Convert the original image to grayscale for faster computation  
4.Load the Haar Cascade classifier for license plate detection  
5.Detect license plate using detectMultiScale function  
6.Draw rectangle around detected area  
7.Crop the detected region using numpy slicing with (x, y, w, h) values  
8.Apply median blurring on the cropped region  
9.Replace the original region with blurred version  
10.Display final result using Matplotlib

## Program
```python
import cv2
import matplotlib.pyplot as plt
img = cv2.imread("car_plate.jpg")
def display(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10,6))
    plt.imshow(img_rgb)
    plt.axis('off')
display(img)
import os

plate_cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_russian_plate_number.xml"))
def detect_plate(img):
    img_copy = img.copy()
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    plates = plate_cascade.detectMultiScale(gray,
                                            scaleFactor=1.1,
                                            minNeighbors=4)

    for (x, y, w, h) in plates:
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0,255,0), 3)

    return img_copy

result = detect_plate(img)
def detect_and_blur_plate(img):
    img_copy = img.copy()
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    plates = plate_cascade.detectMultiScale(gray,
                                            scaleFactor=1.1,
                                            minNeighbors=4)

    for (x, y, w, h) in plates:
        roi = img_copy[y:y+h, x:x+w]     # Grab license plate
        blurred_roi = cv2.medianBlur(roi, 15) # Blur using kernel size 15
        img_copy[y:y+h, x:x+w] = blurred_roi  # Paste back

    return img_copy

result = detect_and_blur_plate(img)
display(result)
```

## Output
<img width="794" height="457" alt="download" src="https://github.com/user-attachments/assets/8121ff6d-83ce-4f2a-992a-6d5f454e2dce" />



<img width="794" height="457" alt="download" src="https://github.com/user-attachments/assets/32256292-a335-47e5-9e2a-a7890d901f5a" />



<img width="794" height="457" alt="download" src="https://github.com/user-attachments/assets/fb21aa84-5307-4e88-8d02-696fd6a9aa71" />


## Modification Done

Parameter tuning was performed by adjusting scaleFactor and minNeighbors values in detectMultiScale to improve accuracy and reduce false detections. Median blur was applied to protect license plate information.

## Result

The License Plate Detection system was successfully implemented using OpenCV and Haar Cascade. The detected license plate region was blurred using median filtering. The modified values improved overall detection performance and output quality.

## Conclusion

This workshop demonstrates how classical computer vision methods like Haar Cascades can be used for real-time applications such as automated toll systems, smart parking, and traffic surveillance. Proper preprocessing and parameter tuning significantly improve detection results.
