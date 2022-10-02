from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from cv2 import VideoCapture,imwrite,imread,waitKey
import streamlit as st
import pandas as pd
import webbrowser

# Load the model
model = load_model('keras_model.h5')

cam = VideoCapture(0)
result, image = cam.read()
if result:
    
    # show the image
    # imshow("GeeksForGeeks", image)
    
    # save the image
    imwrite("scan.jpg", image)

    waitKey(0)

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
# Replace this with the path to your image
image = Image.open('scan.jpg')
#resize the image to a 224x224 with the same strategy as in TM2:
#resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.ANTIALIAS)

#turn the image into a numpy array
image_array = np.asarray(image)
# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
# Load the image into the array
data[0] = normalized_image_array

# run the inference
prediction = model.predict(data)
mylist=prediction.tolist()
l=max(mylist)
# print(l)


d={}
d[l[0]]='Pringles'
d[l[1]]='Lays'

print(d[max(l)])


if d[max(l)] == 'Pringles':
    st.image('Pitchdeck.jpg')

else:
    st.image('Pitchdeck2.jpg')


# st.write("Here's the nutritional information for Pringles:")
# st.write(pd.DataFrame({
#     'first column': ['sugar', 2, 3, 4],
#     'second column': ['Calories', 20, 30, 40]
# }))
# st.write()
# st.write("Similar recommended products for you")
# st.write("Lay's Sour  Cream and Onion Chips")
# st.write("No milk and corn allergens")