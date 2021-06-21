import sys
import numpy as np
from PIL import Image
from pathlib import Path
import os
import cv2

def change(path):
    path_obj = Path(path) 
    idx = 1
    for item in path_obj.iterdir():
        person = cv2.imread(str(item))
        if person is None:
        	break
        person = cv2.resize(person, dsize=(32, 32))
        person = cv2.cvtColor(person,  cv2.COLOR_BGR2GRAY)
        cv2.imwrite('./images/'+str(idx)+'.png', person)
        # cv2.imshow("person", person)
        # cv2.waitKey()
        # cv2.waitKey()
        idx += 1

def readImages(folder):
    images = []
    idx = 1
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, dsize=(32, 32))
       
        if idx > 3000: 
            break  # 모든 하위 항목을 순회하며
        
        if img is not None:
            images.append(img)
        idx += 1

    return images

def createDataMatrix(images):
	print("Creating data matrix",end=" ... ")
	''' 
	Allocate space for all images in one data matrix. 
        The size of the data matrix is
        ( w  * h  * 3, numImages )
       
        where,
        
        w = width of an image in the dataset.
        h = height of an image in the dataset.
        3 is for the 3 color channels.
        '''
  
	numImages = len(images)
	sz = images[0].shape
	data = np.zeros((numImages, sz[0] * sz[1]), dtype=np.float32)
	for i in range(0, numImages):
		image = images[i].flatten()
		data[i,:] = image
	
	print("DONE")
	return data

def createNewFace(*args):
	# Start with the mean image
	output = averageFace
	
	# Add the eigen faces with the weights
	for i in range(0, NUM_EIGEN_FACES):
		'''
		OpenCV does not allow slider values to be negative. 
		So we use weight = sliderValue - MAX_SLIDER_VALUE / 2
		''' 
		sliderValues[i] = cv2.getTrackbarPos("Weight" + str(i), "Trackbars");
		weight = sliderValues[i] - MAX_SLIDER_VALUE/2
		output = np.add(output, eigenFaces[i] * weight)

	# Display Result at 2x size
	output = cv2.resize(output, (0,0), fx=2, fy=2)
	cv2.imshow("Result", output)

def findCoefficents(test_data):
	global NUM_EIGEN_FACES
	coef = np.zeros((len(test_data), 1024))
	
	for i in range(len(test_data)):
		prac = test_data[i] - mean
		
		for j in range(len(eigenVectors)):
			coef[i][j] = np.dot(prac, eigenVectors[j])

	return coef

def checkSimilarity(coef1, coef2, same_flag=False):
	dist = [[0 for x in range(5)] for y in range(5)]
	for i in range(5):
		for j in range(5):
			dist[i][j] = np.linalg.norm((coef1[i])-(coef2[j]))
	
	length = 0
	if same_flag:
		for diagonal in range(5):
			for index in range(diagonal+1, 5):
				length += dist[diagonal][index]
		length /= 10
	else :
		for diagonal in range(5):
			for index in range(diagonal, 5):
				length += dist[diagonal][index]
		length /= 15
	print(length)
	return length

	'''
	for i in range(5):
		for j in range(5):
			dist[i][j] = np.dot(coef1[i], coef2[j]) \
			/(np.linalg.norm(coef1[i])*np.linalg.norm(coef2[j]))

	length = 0
	if same_flag:
		for diagonal in range(5):
			for index in range(diagonal+1, 5):
				length += dist[diagonal][index]
		length /= 10
	else :
		for diagonal in range(5):
			for index in range(diagonal, 5):
				length += dist[diagonal][index]
		length /= 15
	print(length)
	'''
def generateFaceImage(test_data, coef, direc):
	idx = 1
	for i in range(len(test_data)):
		print("image {}".format(i))
		recreate = np.zeros((1, 1024))

		for j in range(len(eigenVectors)):
			recreate += coef[i][j] * eigenVectors[j]
		
		recreate += mean
		rec_img = recreate.reshape(sz)
		rec_img = rec_img.astype(np.uint8)
		output = cv2.resize(rec_img, (0, 0), fx=10, fy=10)
		cv2.imshow("Result", output)
		cv2.waitKey()
		cv2.imwrite('./eigenfaces/'+str(idx)+'.png', output)
		idx += 1

def testImage(direc):
	test_imgs = readImages(direc)
	test_data = createDataMatrix(test_imgs)

	coef = findCoefficents(test_data)
	#heckSimilarity(coef, coef, True)
	#generateFaceImage(test_data, coef, direc)

	return coef
	
if __name__ == '__main__':
	# 프로젝트 경로의 모든 하위 항목 출력
	#print("Create sample data ", end="...")
	#change('lfwcrop_grey/faces/')
	#print("Done") 

	# Directory containing images
	dirName = "images"

	# Read images
	images = readImages(dirName)
	print("Number of Sample Data : {}".format(len(images)))
	# Size of images
	sz = images[0].shape
	
	# Create data matrix for PCA.
	data = createDataMatrix(images)

	# Number of EigenFaces
	global NUM_EIGEN_FACES
	NUM_EIGEN_FACES = 30
	# Maximum weight
	MAX_SLIDER_VALUE = 255

	# Compute the eigenvectors from the stack of images created
	print("Calculating PCA ", end="...")
	mean, eigenVectors = cv2.PCACompute(data, mean=None, 
		maxComponents=NUM_EIGEN_FACES)
	print ("DONE")
	averageFace = mean.reshape(sz)
	
	'''	
	eigenFaces = []; 
	
	idx = 1
	for eigenVector in eigenVectors:
		print(np.linalg.norm(eigenVector))
		eigenFace = eigenVector.reshape(sz)
		eigenFace = eigenFace * 750 + 128
		eigenFace = eigenFace.astype(np.uint8)
		eigenFaces.append(eigenFace)
		output = cv2.resize(eigenFace, (0, 0), fx=10, fy=10)
		cv2.imshow("Result", output)
		cv2.waitKey()
		cv2.imwrite('./eigenfaces/'+str(idx)+'.png', output)
		idx += 1
	
	# Create window for displaying Mean Face
	cv2.namedWindow("Result", cv2.WINDOW_AUTOSIZE)
	averageFace = averageFace.astype(np.uint8)
	# Display result at 2x size
	output = cv2.resize(averageFace, (0, 0), fx=10, fy=10)
	cv2.imshow("Result", output)
	cv2.waitKey()
	'''
	

	coef0 = testImage("test/sample0")
	coef1 = testImage("test/sample1")
	coef2 = testImage("test/sample2")
	coef3 = testImage("test/sample3")
	coef4 = testImage("test/sample4")
	coef5 = testImage("test/sample5")
	coef6 = testImage("test/sample6")
	coef7 = testImage("test/sample7")
	coef8 = testImage("test/sample8")
	coef9 = testImage("test/sample9")
	
	error = checkSimilarity(coef9, coef0)
	error += checkSimilarity(coef9, coef1)
	error += checkSimilarity(coef9, coef2)
	error += checkSimilarity(coef9, coef3)
	error += checkSimilarity(coef9, coef4)
	error += checkSimilarity(coef9, coef5)
	error += checkSimilarity(coef9, coef6)
	error += checkSimilarity(coef9, coef7)
	error += checkSimilarity(coef9, coef8)
	print(error/9)
	'''
	# Create Window for trackbars
	cv2.namedWindow("Trackbars", cv2.WINDOW_AUTOSIZE)

	sliderValues = []
	
	# Create Trackbars
	for i in range(0, NUM_EIGEN_FACES):
		sliderValues.append(MAX_SLIDER_VALUE/2)
		cv2.createTrackbar( "Weight" + str(i), "Trackbars", MAX_SLIDER_VALUE/2, MAX_SLIDER_VALUE, createNewFace)
	
	# You can reset the sliders by clicking on the mean image.
	cv2.setMouseCallback("Result", resetSliderValues);
	'''
	#print('''Usage:
	#Change the weights using the sliders
	#Click on the result window to reset sliders
	#t ESC to terminate program.''')
	'''
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	'''