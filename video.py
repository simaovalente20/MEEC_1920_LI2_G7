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
scaler= pickle.load(open('Scaler.sav', 'rb'))

class Video:
	def __init__(self):
		pass

	# Opens the connection to the camera
	def open(self,url):
		self.url=url
		self.cam = cv2.VideoCapture(url)
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

	# Detection of faces in an image with opencv
	def detectFaces_opencv(self, image):
		self.image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		# Detect faces
		self.faces = face_cascade.detectMultiScale(self.image_gray, 1.1, 4)
		# Draw rectangle around the faces
		for (x, y, w, h) in self.faces:
			cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
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
			if probability.max() > 0.25:
				cv2.putText(image,"Grupo " + str(predict), (x, y - 10), 0, 1, (0,255,0))
			else:
				cv2.putText(image, "Desconhecido", (x, y - 10), 0, 1, (0,0,255))
		return image


	def classify1(self, shape):
		x = np.zeros((68, 2), dtype=float)
		xx = np.zeros((136), dtype=float)
		x = shape
		nx, ny = x.shape
		xx = x.reshape(nx * ny)
		xx = xx.reshape(1, -1)
		model = pickle.load(open('KNeighborsClassifier.sav', 'rb'))
		print(x)
		predict = model.predict(np.array(xx))
		print(predict)

	def getImage(self,url):
		image = cv2.imread(url)
		return image

	def getImageFace100x100(self, image, rects):
		image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		# Draw rectangle around the faces
		for (i, rect) in enumerate(rects):
			print("number",i)
			# Convert dlib's rectangle to a OpenCV-style bounding box
			(x, y, w, h) = face_utils.rect_to_bb(rect)
			im_face = image_gray[y:y + h, x:x + w]
			im_face = cv2.resize(im_face, (100, 100))
			gray_face=im_face.reshape(-1)
		return np.array(gray_face.reshape(-1))



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