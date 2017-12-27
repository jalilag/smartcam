import sys
sys.path.insert(0,'../facenet/')
import tensorflow as tf
# import nn4 as network
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
		
	def get_faces(self,img,cropped_img=True,with_plot=False):
		with tf.Graph().as_default():
			gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory_fraction)
			sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,log_device_placement=False))
			with sess.as_default():
				pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
		bounding_boxes, _ = detect_face.detect_face(img,self.minsize,pnet,rnet,onet,self.threshold,self.factor)
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

	def load_data(self,fpath):
		data = dict()
		datax = list()
		datay = list()

		for fid in os.listdir(fpath):
			res = list()
			for i in os.listdir(fpath+"/"+fid):
				if re.search('.+.jpg',i):
					img=cv2.cvtColor(scipy.misc.imread(fpath+"/"+fid+"/"+i), cv2.COLOR_BGR2GRAY)
					if img.ndim == 2: img = self.to_rgb(img)
					img = self.get_faces(img)
					if len(img) > 1: break
					img = cv2.resize(img[0],(self.image_size,self.image_size), interpolation=cv2.INTER_CUBIC)
					datax.append(img)
					datay.append(fid)
		return datax,datay

	def load_data_paths(self,img_path="test/"):
		datax = list()
		datay = list()
		datalabs = list()
		ct = -1
		for fid in os.listdir(img_path):
			ct += 1
			res = list()
			print(img_path)
			datalabs.append(fid)
			for i in os.listdir(img_path+fid):
				if re.search('.+.jpg',i):
					datax.append(img_path+fid+"/"+i)
					datay.append(ct)
		return datax,datay,datalabs


	def read_img(self,fpath,reshape=True):
		res = list()
		for i in fpath:
			print(i)
			img=cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2GRAY)
			if img.ndim == 2: img = self.to_rgb(img)
			img = self.get_faces(img)
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

	def get_embs(self,imgs):
		with tf.Graph().as_default():
			with tf.Session() as sess:
				n.random.seed(seed=self.random_state)
				# Load the model
				print('Loading feature extraction model')
				facenet.load_model(self.pre_trained_model)            
				# Get input and output tensors
				images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
				embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
				phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
				embedding_size = embeddings.get_shape()[1]            
				# Run forward pass to calculate embeddings
				print('Calculating features for images')
				feed_dict = { images_placeholder:imgs, phase_train_placeholder:False }
				emb_array = sess.run(embeddings, feed_dict=feed_dict)
		return emb_array

	def train(self,img_path="img/"):
		datax,datay,datalabs = self.load_data_paths(img_path)
		embs = self.get_embs(self.read_img(datax))
		print('Training classifier')
		model = SVC(kernel='linear', probability=True)

		model.fit(embs, datay)
		model_name = "models/classifier/"+datetime.datetime.now().strftime("%Y%d%m")+".model"
		with open(model_name, 'wb') as outfile:
			pickle.dump((model, datalabs), outfile)
		print('Saved classifier model to file "%s"' % model_name)
					
	def predict(self,fpath,model_path="models/classifier/"):
		emb = self.get_embs(self.read_img([fpath]))
		m = n.sort(os.listdir(model_path))[-1]
		with open(model_path+m, 'rb') as infile:
			(model, class_names) = pickle.load(infile)
		predictions = model.predict_proba(emb)
		print(predictions)
		best_class_indices = n.argmax(predictions, axis=1)
		print(best_class_indices)
		best_class_probabilities = predictions[n.arange(len(best_class_indices)), best_class_indices]
		print(best_class_probabilities)
		print(class_names)
		
		for i in range(len(best_class_indices)):
			print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
			
		# accuracy = n.mean(n.equal(best_class_indices, labels))
		# print('Accuracy: %.3f' % accuracy)
		# print(predictions)
		# return predictions


