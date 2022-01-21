import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#Python Imaging Library (PIL) - external library adds support for image processing capabilities
from PIL import Image
import PIL.ImageOps


#Fetching the data
X = np.load('image.npz')['arr_0']
y = pd.read_csv("labels.csv")["labels"]
print(pd.Series(y).value_counts())
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U'', 'V', 'W', 'X', 'Y', 'Z']

nclasses  = len(classes)

#To only consider the area inside the box for detecting the digit

   #ro1 = Region of Internet

   roi = gray[upper_left[1]:bottom_right[1],
upper_left[0]:bottom_right[0]]

# convert to grayscale image - 'L' format means each pixel is
# represented by a single value from 0 to 255
image_bw = im_pill.convert('L')
image_bw_resized= image_bw.resize((28,28), Image.ANTIALIAS)
#invert the image
image_bw_resized_inverted = PIL.ImageOps. invert(image_bw_resized)
pixel_filter = 20
#convert to scalar  quantity
min_pixel = np.percentile(image_bw_resized_inverted, pixel_filter)
#using clip to limit the values between 0,255
image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted-min_pixel, 0, 255)
max_pixel = np.max(image_bw_resized_inverted)
#converting into and array
iamge_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
#creating a test sample and making a prediction
test_sample = np.arrray(image_bw_resized_inverted_scaled).reshape(1,784)
test_pred = clf.predict(test_sample)

print("Predicted class is: ", test_pred)


clf = LogisticRegression(solver= "saga", multi_class= 'multinomial').fit(xtrainscaled, ytrain)
ypred = clf.predict(xtestscaled)
acc = accuracy_score(ytest, ypred)
print(acc)

#Starting the camera
cap = cv2.VideoCapture(0)
while(True):
    try:
        ret,frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #Drawing a box in the center of the video
        height,width = gray.shape
        upperleft = (int(width / 2 - 56), int(height / 2 - 56))
        bottomright = (int(width / 2 + 56), int(height / 2 + 56))
        cv2.rectangle(gray, upperleft, bottomright, (0,255,0), 2)
        roi = gray[upperleft[1]:bottomright[1], upperleft[0]: bottomright[0]]
        #Converting cv2 image to pil format so that the interpreter understands
        impil = Image.fromarray(roi)
        imagebw = impil.convert('L')
        iamgebwresize  = imagebw.resize( (28,28), Image.ANTIALIAS)
        imgInverted = PIL.ImageOps.invert(iamgebwresize)
        pixelfilter = 20
        #percentile() converts the values in scalar quantity
        minpixel = np.percentile(imgInverted, pixelfilter)
        
        #using clip to limit the values betwn 0-255
        imgInverted_scaled = np.clip(imgInverted - minpixel, 0, 255)
        maxpixel = np.max(imgInverted)
        imgInverted_scaled = np.asarray(imgInverted_scaled)/maxpixel
          #converting into an array() to be used in model for prediction
        testsample = np.array(imgInverted_scaled).reshape(1,784)
        testpred = clf.predict(testsample)
        
        print("Predicted class is: ", testpred)
        # Display the resulting frame
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break
    except Exception as e:
        pass
cap.release()
cv2.destroyAllWindows()