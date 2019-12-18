#A class to handle image objects. The data is set up so that the name of the image file contains vital information about the image
import os
import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd

class ImageHandler():
	#NUM=1_SLEEVES=N_GLOVES=N_BACK=IN_PERSON=FENTON_ORIENT=BACK_HAND=LEFT
	#The path is the path to the image file
	#The width and height refer to the dimensions of the image.
	#All the images should be the same size
	def __init__(self, path, width, height):

		self.path = path
		#First let's extract information from the path (String)
		self.fullName = os.path.basename(path)

		numIndex = self.fullName.find("NUM") + 4
		self.num = self.fullName[NumIndex]

		#The sleeves index is either Y (has sleeves) or N (No sleeves)
		sleeveIndex = self.fullName.find("SLEEVES") + 8
		self.sleeves = False
		if self.fullName[sleeveIndex] == 'Y':
			self.sleeves = True

		#The gloves index is either Y (Wearing gloves) or N (No gloves)
		gloveIndex = self.fullName.find("GLOVES") + 7
		self.gloves = False
		if self.fullName[gloveIndex] == 'Y':
			self.gloves = True

		#For the background, the image is either (IN) inside or (OUT) outside
		backIndex = self.fullName.find("BACK") + 5
		self.background = 'IN'
		if self.fullName[backIndex] == 'O':
			self.background = 'OUT'

		#Here we find the person
		personIndex = self.fullName.find("PERSON") + 7
		self.person = ""
		#We have the first character, let's add the rest of the name
		while(self.fullName[personIndex] != "_"):
			self.person += self.fullName[personIndex]
			personIndex += 1

		#The orientation is either BACK or FRONT (Sides of the hand)
		orientIndex = self.fullName.find("ORIENT") + 7
		self.orientation = "BACK"
		if self.fullName[orientIndex] == 'F':
			self.orientation = "FRONT"

		#The hand is either the LEFT or the RIGHT.
		handIndex = self.fullName.find("HAND") + 5
		self.hand = "LEFT"
		if self.fullName[handIndex] == "R":
			self.hand = "RIGHT"

		#Now let's load the image in  
		self.image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

		self.resize(width, height)
	#Display an image
	def displayImage(self):
		plt.imshow(self.path)
		plt.title(self.fullName)
		plt.xticks([])
		plt.yticks([])
		plt.show()


	def resize(self, width, height):
		dim = (width, height)

		#Make this more independent later
		self.res_img = [] #res_img for resized image

		for i in range(len(self.image)):
			res = cv2.resize(self.image[i], dim, intepolation=cv2.INTER_LINEAR)
			res_img.append(res)

		#Displaying the image
		displayImage(res+img[1])

	'''
	Applies Guassian smoothing to reduce the image noise.
	Involves perforiming a 2-dimensional Weierstrass transform.
	Lowers image high-frequency parts. 

	'''
	def gaussian(self):
		if self.noiseless == None:

			self.noiseless = []
			for i in range(0, len(self.res_img)):
				blur = cv2.GaussianBlur(self.res_img[i])
				self.noiseless.append(blur)
		else:
			#Another filter was already applied. Let's add this
			copy_noise = []
			for i in range(0, len(self.noiseless)):
				blur = cv2.medianBlur(self.noiseless[i])
				copy_noise.append(blur)

			self.noiseless = copy_noise

	def median(self):
		if self.noiseless == None:
			#Here we apply the median filter to the resized image
			self.noiseless = []
			#Should error check if the resizing has been done
			for i in range(0, len(self.res_img)):
				#ksize = 5? 
				blur = cv2.medianBlur(self.res_img[i], 5)
				self.noiseless.append(blur)

		else:
			copy_noise = self.noiseless
			#Another filter was applied. Let's apply this filter over it
			for i in range(0, len(self.noiseless)):
				blur = cv2.medianBlur(self.res_img[i], 5)
				copy_noise.append(blur)

			self.noiseless = copy_noise

	#See http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MANDUCHI1/Bilateral_Filtering.html
	#Good for preserving edges
	def bilateral(self):
		if self.noiseless == None:
			self.noiseless = []

			for i in range(0, len(self.res_img)):
				#The arguments are image, diameter of pixel neighborhood, Sigmas of color space
				blur = cv2.bilateralFilter(self.res_img[i], 5, 75,75)
				self.noiseless.append(blur)

		else:
			copy_noise = self.noiseless

			for i in range(0, len(self.noiseless)):
				blur = cv2.bilateralFilter(self.noiseless[i], 5, 75, 75)
				copy_noise.append(blur)
			self.noiseless = copy_noise


	def colorSegment(self):

		#