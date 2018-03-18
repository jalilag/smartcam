import sys
sys.path.insert(0,'../facenet/')
import tensorflow as tf
import os
import scipy
from src.align import detect_face
from src import facenet
from imageio import imread
import cv2
import matplotlib.pyplot as plt
import re
import numpy as n
import datetime
import math
import pickle
from sklearn.svm import SVC
import random

class face_embs():
	sess = None
	embeddings = None
	g = None
	images_placeholder = None
	phase_train_placeholder = None
	model_dir = "models/"
	pre_trained_model_name = "20170512-110547"
	pre_trained_model = model_dir+"pre_trained/"+pre_trained_model_name+"/"+pre_trained_model_name+".pb"
	def __init__(self):
		self.g = tf.Graph()
		with self.g.as_default():
			self.sess = tf.Session()		
			# Load the model
			print('Loading feature extraction model')
			facenet.load_model(self.pre_trained_model)            
			# Get input and output tensors
			self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
			self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
			self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
			# embedding_size = self.embeddings.get_shape()[1]            
		# Run forward pass to calculate embeddings
	def get_embs(self,imgs):
		print('Calculating features for images')
		feed_dict = {self.images_placeholder:imgs, self.phase_train_placeholder:False }
		return self.sess.run(self.embeddings, feed_dict=feed_dict)

class face_detection():
	#   setup facenet parameters
	gpu_memory_fraction = 1.0
	minsize = 50 # minimum size of face
	threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
	factor = 0.709 # scale factor
	image_size = 160
	margin = 32
	random_state = 42
	model_dir = "models/"
	pre_trained_model_name = "20170512-110547"
	pre_trained_model = model_dir+"pre_trained/"+pre_trained_model_name+"/"+pre_trained_model_name+".pb"
	batch_size = 1000
	pnet = None
	rnet = None
	onet = None
	def __init__(self):
		with tf.Graph().as_default():
			gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory_fraction)
			sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,log_device_placement=False))
			with sess.as_default():
				self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(sess, None)

		
	def get_faces(self,img,cropped_img=True,with_plot=False):
		bounding_boxes, _ = detect_face.detect_face(img,self.minsize,self.pnet,self.rnet,self.onet,self.threshold,self.factor)
		res = list()
		img_size = n.asarray(img.shape)[0:2]
		for (x1, y1, x2, y2, acc) in bounding_boxes:
			w = x2-x1
			h = y2-y1
			bb = n.zeros(4, dtype=n.int32)
			bb[0] = n.maximum(x1-self.margin/2, 0)
			bb[1] = n.maximum(y1-self.margin/2, 0)
			bb[2] = n.minimum(x2+self.margin/2, img_size[1])
			bb[3] = n.minimum(y2+self.margin/2, img_size[0])
			print ('Accuracy score', acc)
			if cropped_img: res.append(img[bb[1]:bb[3],bb[0]:bb[2],:])
			else:
				res = img
				cv2.rectangle(res,(int(x1),int(y1)),(int(x1+w),int(y1+h)),(255,0,0),2)
			if with_plot and cropped_img:
				plt.figure()
				plt.imshow(res[-1])
		if with_plot and cropped_img == False:
			plt.figure()
			plt.imshow(res)	
		if with_plot: plt.show()
		return res


	def load_data_paths(self,img_path="test/"):
		data = dict()
		for fid in os.listdir(img_path):
			l= list()
			for i in os.listdir(img_path+fid):
				l.append(img_path+fid+"/"+i)
			data[fid] = l
		return data

	def data_split(self,data,test_fraction=0.3):	
		train = dict()
		test = dict()
		for i,j in data.items():
			random.shuffle(j)
			N = int(test_fraction*len(j))
			train[i] = j[N:]
			test[i] = j[0:N]
		return train,test

	def data_dict_to_list(self,data):
		datax = list()
		datay = list()
		datalabs = list()
		ct = -1
		for i,j in data.items():
			datalabs.append(i)
			ct += 1
			for ii in j:
				datax.append(ii)
				datay.append(ct)
		return datax,datay,datalabs


	def read_img(self,fpath,multiple=False):
		res = list()
		if isinstance(fpath,str): fpath = [fpath]
		for i in fpath:
			print(i)
			img=cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2GRAY)
			if img.ndim == 2: img = self.to_rgb(img)
			img = self.get_faces(img)
			print(n.shape(img))
			if len(img) > 1: return None
			res.append(cv2.resize(img[0],(self.image_size,self.image_size), interpolation=cv2.INTER_CUBIC))
		return n.array(res)


	def split_args(self,sample_size,coef=0.7):
		a = n.random.permutation(n.arange(sample_size))
		x = int(coef*sample_size)
		return a[:x],a[x:]


	def to_rgb(self,img):
		w, h = img.shape
		ret = n.empty((w, h, 3), dtype=n.uint8)
		ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
		return ret
