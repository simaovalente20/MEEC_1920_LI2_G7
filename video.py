import cv2
import dlib
import argparse
import imutils
from imutils import face_utils

# Criation of classifier that allows detecting faces with opencv
face_cascade = cv2.CascadeClassifier("venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

class Video:
	def __init__(self,url):
		self.url = url
		self.cam=cv2.VideoCapture("http://192.168.1.66:8080")

	# Opens the connection to the camera
	def open(self):
		self.cam.open(self.url)

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
	def getLandmaks(self,image,image_gray,rects):
		# Cycles through the detected faces
		for (i, rect) in enumerate(rects):
			# determine the facial landmarks for the face region
			shape = predictor(image_gray, rect)
			#convert the facial landmark (x, y)-coordinates to a NumPy array
			shape = imutils.face_utils.shape_to_np(shape)
			#Cycles through and draw them on the image
			for (x, y) in shape:
				cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

		return image

	def getImage(self,url):
		image = cv2.imread(url)
		return image
