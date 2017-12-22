import sys
sys.path.insert(0,'../facenet/')
import tensorflow as tf
# import nn4 as network
import os
from src.align import detect_face
from src import facenet
from imageio import imread
import cv2
import matplotlib.pyplot as plt
import re
import numpy as n
import datetime
import math 

class face_detection():
	#   setup facenet parameters
	gpu_memory_fraction = 1.0
	minsize = 50 # minimum size of face
	threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
	factor = 0.709 # scale factor
	image_size = 96
	pnet = None
	rnet = None
	onet = None
	random_state = 42
	model_dir = "models/"
	pre_trained_model_name = "20170511-185253"
	pre_trained_model = model_dir+"pre_trained/"+pre_trained_model_name+"/"+pre_trained_model_name+".pb"
	batch_size = 1000

	def __init__(self,image_size=96):
		self.image_size = image_size
		with tf.Graph().as_default():
			gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory_fraction)
			sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,log_device_placement=False))
			with sess.as_default():
				self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(sess, None)
	
	def get_faces(self,img,cropped_img=True,with_plot=False):
		bounding_boxes, _ = detect_face.detect_face(img,self.minsize,self.pnet,self.rnet,self.onet,self.threshold,self.factor)
		res = list()
		for (x1, y1, x2, y2, acc) in bounding_boxes:
			w = x2-x1
			h = y2-y1
			if cropped_img:
				res.append(img[int(y1):int(y2),int(x1):int(x2),:])
			else:
				res = img
				cv2.rectangle(res,(int(x1),int(y1)),(int(x1+w),int(y1+h)),(255,0,0),2)
			if with_plot and cropped_img:
				plt.figure()
				plt.imshow(res[-1])
			print ('Accuracy score', acc)
		if with_plot and cropped_img == False:
			print("ok")
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
					img=cv2.cvtColor(cv2.imread(fpath+"/"+fid+"/"+i), cv2.COLOR_BGR2GRAY)
					if img.ndim == 2: img = self.to_rgb(img)
					img = self.get_faces(img)
					if len(img) > 1: break
					img = cv2.resize(img[0],(self.image_size,self.image_size), interpolation=cv2.INTER_CUBIC)
					# res.append(img.reshape(-1,self.image_size,self.image_size,3))
					if reshape: img =img.reshape(-1,self.image_size,self.image_size,3) 
					datax.append(img)
					datay.append(fid)
		return datax,datay

	def load_data_paths(self,img_path="test/"):
		datax = list()
		datay = list()
		for fid in os.listdir(img_path):
			res = list()
			print(img_path)
			for i in os.listdir(img_path+fid):
				if re.search('.+.jpg',i):
					datax.append(img_path+fid+"/"+i)
					datay.append(fid)
		return datax,datay


	def read_img(self,fpath,reshape=False):
		res = list()
		for i in fpath:
			print(i)
			img=cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2GRAY)
			if img.ndim == 2: img = self.to_rgb(img)
			img = self.get_faces(img)
			if len(img) > 1: return None
			img = cv2.resize(img[0],(self.image_size,self.image_size), interpolation=cv2.INTER_CUBIC)
			if reshape: img =img.reshape(-1,self.image_size,self.image_size,3) 
			res.append(img)
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

	def train(self,img_path="img/"):
		datax,datay = self.load_data_paths(img_path)
		imgs = self.read_img(datax,False)
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
				nrof_images = len(datax)
				# nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / self.batch_size))
				# emb_array = n.zeros((nrof_images, embedding_size))
				# for i in range(nrof_batches_per_epoch):
				# 	start_index = i*self.batch_size
				# 	end_index = min((i+1)*self.batch_size, nrof_images)
				# 	# paths_batch = paths[start_index:end_index]
				# 	# images = facenet.load_data(paths_batch, False, False, args.image_size)
				feed_dict = { images_placeholder:imgs, phase_train_placeholder:False }
				emb_array = sess.run(embeddings, feed_dict=feed_dict)
				
				# classifier_filename_exp = os.path.expanduser(args.classifier_filename)

				print('Training classifier')
				model = SVC(kernel='linear', probability=True)
				model.fit(emb_array, n.arange(len(datay)))

				# Saving classifier model
				model_name = datetime.datetime.now().strftime("%d%m%Y")+".model"
				with open(model_name, 'wb') as outfile:
					pickle.dump((model, datay), outfile)
				print('Saved classifier model to file "%s"' % classifier_filename_exp)
					
	def predict(self,fpath):
		img = self.read_img([fpath])
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
				nrof_images = len(datax)
				nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / self.batch_size))
				emb_array = n.zeros((nrof_images, embedding_size))
				for i in range(nrof_batches_per_epoch):
					start_index = i*self.batch_size
					end_index = min((i+1)*self.batch_size, nrof_images)
					feed_dict = { images_placeholder:img, phase_train_placeholder:False }
					emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
					# Classify images
					print('Testing classifier')
				with open(model_dir+model_name, 'rb') as infile:
					(model, class_names) = pickle.load(infile)

				predictions = model.predict_proba(emb_array)
				print(predictions)
				# best_class_indices = np.argmax(predictions, axis=1)
				# best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
				
				# for i in range(len(best_class_indices)):
				#     print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
					
				# accuracy = np.mean(np.equal(best_class_indices, labels))
				# print('Accuracy: %.3f' % accuracy)



