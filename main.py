from user_img_proc import face_detection, face_embs
from imageio import imread
import numpy as n
import matplotlib.pyplot as plt
import sys
from user_sklearn import Classification

fd = face_detection()
emb = face_embs()
datax,datay,datal = fd.data_dict_to_list(fd.load_data_paths())
datax = emb.get_embs(fd.read_img(datax))
print(datay)
train = Classification(datax,datay)
clf = train.train_model(model="sgd",with_plot=True,fixed_params={"loss":"log"})
print("pred",train.fitted_model.predict(emb.get_embs(fd.read_img("img/f5.jpg"))))
# print(n.shape(trainx))
# print(n.shape(trainx+testx))

# fd.predict("img/f7.jpg")
# a = fd.get_faces(imread("img/f6.jpg"))
# fe = face_embs()
# print(fe.get_embs(a))
# fd.read_img("img/f1.jpg")
# fd.train("test/")
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