from django.shortcuts import render
from .models import Picture,Upload
import cv2
import numpy as np
from .forms import UploadImage
from .face_recog import *

# Create your views here.
def submit(request):
	if request.method == 'POST':
		form = UploadImage(request.POST, request.FILES)
		if form.is_valid():
			instance = Upload(image = request.FILES['file'])
			instance.save()
			
			images=Upload.objects.all()[:1]
			for uploaded in images:
				path = uploaded.image.path
				img = cv2.imread(path,1)
				image, nam = predict(img)
			Upload.objects.all().delete()
			return render(request, 'human_face/result.html', {'match': nam})
	else:
		form = UploadImage()
    
	return render(request, 'human_face/upload.html', {'form': form})

