import cv2

face_cascade = cv2.CascadeClassifier("venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")

class Video:
    def __init__(self,url):
        self.url = url
        self.cam=cv2.VideoCapture()

    def open(self):
        self.cam.open(self.url)

    def close(self):
        if self.cam.isOpened():
            self.cam.release()

    def capture(self):
        if not self.cam.isOpened():
            self.cam.open(self.url)
        ret, image = self.cam.read()
        return image

    def detectFaces(self, image):
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(image_gray, 1.1, 4)
        # Draw rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        return image
