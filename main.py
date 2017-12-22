from user_img_proc import face_detection
from imageio import imread
import numpy as n
fd = face_detection()
fd.get_faces(imread("img/f3.jpg"),True,True)
# x,y = fd.load_data("test")
# yy = range(len(y))
# fr = face_recognition()

# fd.get_faces(imread("img/f1.jpg"),True,True)