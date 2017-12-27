from user_img_proc import face_detection
from imageio import imread
import numpy as n
import matplotlib.pyplot as plt
fd = face_detection()
# fd.read_img("img/f1.jpg")
fd.train("test/")
fd.predict("test/trump/1.jpg")
# res,_ = fd.load_data_paths()
# res = fd.read_img(res)
# print(n.shape(res))
# for i in range(14):
# 	plt.figure()
# 	plt.imshow(res[i,:,:,:])
# plt.show()


# fd.get_faces(imread("test/f3.jpg"),True,True)
# x,y = fd.load_data("test")
# yy = range(len(y))
# fr = face_recognition()

# fd.get_faces(imread("img/f1.jpg"),True,True)