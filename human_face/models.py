from django.db import models

class Picture(models.Model):
	name=models.CharField(max_length=200)
	photo=models.ImageField(upload_to='pictures')

class Upload(models.Model):
	image=models.ImageField(upload_to='uploaded')