import cv2
import dlib
import argparse
import imutils
from imutils import face_utils
import pickle
import numpy as np
import math

# Criation of classifier that allows detecting faces with opencv
face_cascade = cv2.CascadeClassifier("venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

model = pickle.load(open('LinearSVC.sav', 'rb'))
#model1 = pickle.load(open('train/Multiple_networks_Image/LinearSVC1.sav', 'rb'))
#model2 = pickle.load(open('train/Multiple_networks_Image/LinearSVC2.sav', 'rb'))
#model3 = pickle.load(open('train/Multiple_networks_Image/LinearSVC3.sav', 'rb'))
#model4 = pickle.load(open('train/Multiple_networks_Image/LinearSVC4.sav', 'rb'))
#model5 = pickle.load(open('train/Multiple_networks_Image/LinearSVC5.sav', 'rb'))
#model6 = pickle.load(open('train/Multiple_networks_Image/LinearSVC6.sav', 'rb'))
#model7 = pickle.load(open('train/Multiple_networks_Image/LinearSVC7.sav', 'rb'))
#model8 = pickle.load(open('train/Multiple_networks_Image/LinearSVC8.sav', 'rb'))
scaler= pickle.load(open('Scaler.sav', 'rb'))

class Video:
	def __init__(self):
		pass

	# Opens the connection to the camera
	def open(self,url):
		self.url=url
		self.cam = cv2.VideoCapture(url)
		self.imagAtual=None
		#self.cam.open(self.url)

	# Closes the connection to the camera
	def close(self):
		if self.cam.isOpened():
			self.cam.release()

	# Capture a new frame
	def capture(self):
		if not self.cam.isOpened():
			self.cam.open(self.url)
		ret, image = self.cam.read()
		return image

	# Detection of faces in an image with dlib
	def detectFaces_dlib(self, image):
		image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		#Detect faces
		rects = detector(image_gray, 1)

		# Draw rectangle around the faces
		for(i,rect) in enumerate(rects):
			# Convert dlib's rectangle to a OpenCV-style bounding box
			(x, y, w, h) = face_utils.rect_to_bb(rect)
			cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
		return image,image_gray,rects

	# Find landmarks in the face
	def getLandmaks(self,image):
		image, image_gray, rects = self.detectFaces_dlib(image)
		shape=None
		distances = np.zeros((68*68-68), dtype=float)  # 68,2)
		# Cycles through the detected faces
		for (i, rect) in enumerate(rects):
			# determine the facial landmarks for the face region
			shape = predictor(image_gray, rect)
			#convert the facial landmark (x, y)-coordinates to a NumPy array
			shape = imutils.face_utils.shape_to_np(shape)

			#Cycles through and draw them on the image
			j=0
			for (x, y) in shape:
				cv2.putText(image,str(j),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.3,255)
				cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
				j=j+1


			dist_nose = math.sqrt((shape[35, 0] - shape[31, 0]) ** 2 + (shape[35, 1] - shape[31, 1]) ** 2)

			for i in range(0,68):
				for j in range(0, 68):
					if (j < i):
						distance = math.sqrt((shape[i, 0] - shape[j, 0]) ** 2 + (shape[i, 1] - shape[j, 1]) ** 2)
						if distance == 0:
							distances[(i * 68) + j - (i)] = 0
						else:
							distances[(i * 68) + j - (i)] = dist_nose / distance
					elif (j > i):
						distance = math.sqrt((shape[i, 0] - shape[j, 0]) ** 2 + (shape[i, 1] - shape[j, 1]) ** 2)
						if distance == 0:
							distances[(i * 68) + j - (i + 1)] = 0
						else:
							distances[(i * 68) + j - (i + 1)] = dist_nose / distance
		return image, distances, rects


	def classify(self,image):

		image, shape, rects = self.getLandmaks(image.copy())

		if len(rects) != 0:
			(x, y, w, h) = face_utils.rect_to_bb(rects[0])
			normalize = scaler.transform(np.array(shape).reshape(1, -1))
			predict = model.predict(normalize)
			probability = model._predict_proba_lr(normalize)
			max = probability.max()
			probability = np.delete(probability, probability.argmax())
			if max > 0.25 and max-probability.max()>0.02:
				cv2.putText(image, "Grupo " + str(predict), (x, y - 50), 0, 1, (0, 255, 0))
				cv2.putText(image, str(max), (x, y - 10), 0, 1, (0, 255, 0))
			else:
				cv2.putText(image, "Desconhecido", (x, y - 10), 0, 1, (0, 0, 255))
		self.imagAtual=image
		return image

	def training(self,image):
		image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		# Detect faces
		rects = detector(image_gray, 1)
		# Cycles through the detected faces
		for (i, rect) in enumerate(rects):
			(x, y, w, h) = face_utils.rect_to_bb(rect)
			# determine the facial landmarks for the face region
			shape = predictor(image_gray, rect)
			print(x,y,w,h)
			#shape = shape-rect
			# convert the facial landmark (x, y)-coordinates to a NumPy array
			shape = imutils.face_utils.shape_to_np(shape)

			for (x, y) in shape:
				cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
			cv2.imshow(str(i),image)
			#cv2.waitKey(0)

			#Distância do fundo do nariz
			#Usado como referencia para as restantes distancias
			dist4 = math.sqrt((shape[35, 0] - shape[31, 0]) ** 2 + (shape[35, 1] - shape[31, 1]) ** 2)

			#Cáluclo relação de das distancias de ponto a ponto consecutivo em relação nariz
			distances = np.zeros((68*68-68), dtype=float)  # 68,2)
			for i in range(0,68):
				#distances[i]=dist4/math.sqrt((shape[i+1, 0] - shape[i, 0]) ** 2 + (shape[i+1, 1] - shape[i, 1]) ** 2)
				for j in range(0,68):
					if(j<i):
						distance=math.sqrt((shape[i, 0] - shape[j, 0]) ** 2 + (shape[i, 1] - shape[j, 1]) ** 2)
						if distance==0:
							distances[(i * 68) + j - (i)] = 0
						else:
							distances[(i * 68) + j - (i)] = dist4 / distance
					elif(j>i):
						distance=math.sqrt((shape[i, 0] - shape[j, 0]) ** 2 + (shape[i, 1] - shape[j, 1]) ** 2)
						if distance==0:
							distances[(i * 68) + j - (i + 1)]=0
						else:
							distances[(i*68)+j-(i+1)] = dist4 / distance

			distances = distances.reshape(-1, 1)
		return distances

	def screenshot(self):
		if type(self.imagAtual)!=type(None):
			cv2.imwrite("screenshot.png",self.imagAtual)



	'''	def classifyMultiple(self,image):
			image, shape, rects = self.getLandmaks(image.copy())
			result=[]
			if len(rects) != 0:
				(x, y, w, h) = face_utils.rect_to_bb(rects[0])
				normalize = scaler.transform(np.array(shape).reshape(1, -1))
				max=result[0]
				max[1]=0

				for i in range(1,len(result)):
					print(max[1])
					if result[i][0]== 1:
						if result[i][1]>max[1]:
							max[0]=i+1
							max[1]=result[i][1]

				print("Máximo", max)

			return image
		'''