import cv2
import numpy as np
from .models import Picture
import os
from django.conf import settings
import shutil

file_path = os.path.join(settings.BASE_DIR, 'haarcascade_frontalface_default.xml')
print (settings.BASE_DIR)

subjects = Picture.objects.values_list('name', flat=True).order_by('name').distinct()

def detect_face(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	face_cascade = cv2.CascadeClassifier(file_path)
 
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

	if (len(faces) == 0):
		return None, None
 
	(x, y, w, h) = faces[0]

	if(w>h):
		temp=gray[y:y+w, x:x+w]
	else:
		temp=gray[y:y+h, x:x+h]
	
	temp=cv2.resize(temp,(200,200),interpolation=cv2.INTER_CUBIC)
	
	return temp, faces[0]


def prepare_training_data():

	faces = []
	labels = []
	i=0
	for x in subjects:
		f = Picture.objects.filter(name=x)
		for fac in f:
			label = i
			image = cv2.imread(fac.photo.path)
			face, rect = detect_face(image)
			if face is not None:
				
				faces.append(face)
				labels.append(label)
		
		i = i+1
 
	return faces, labels

def predict(test_img):
	faces, labels = prepare_training_data()
	face_recognizer = cv2.face.FisherFaceRecognizer_create()
	face_recognizer.train(faces, np.array(labels))
	
	#cv2.cvtcolor(test_img,img)
	img = test_img.copy()
	face, rect = detect_face(img)
	
	if(face is None):
		return img,""
	
	label,predicted_confidence= face_recognizer.predict(face)
	label_text = subjects[label]
	
	shutil.rmtree(os.path.join(settings.BASE_DIR,'uploaded')) 
	
	#if(predicted_confidence<100):
	return img,subjects[label]
	
	#return img,"umm! try again with another photo"