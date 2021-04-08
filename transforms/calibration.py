#Jeremiah Hsieh Camera Calibration Chessboard Corner Detection
#program gets images of chessboard at different angles from images folder and calculates camera distortion and matrix
#apparently you are not supposed to look for the whole board just for the fully conncted squares inside? os on an 8x8 board you only look for 7x7?
#this program calculates the camera pinhole matrices parameters used to help in undistort images
#this is done but utilizing known parameters (in this case, chessboard square real size/shape) and checking for any inconsistencies in the image taken
#inconsistencies are compensated for and undistorted to the known chessboard size


import numpy as np 
import cv2 as cv
import os
import pickle



#chessboard corners dimesions

#store image points
#3d points
objpoints = [] 
#2d points
imgpoints = []

#3d points storage
#REMEMBER IT'S ENCLOSED CORNERS NOT SQUARES
#hence why dimensions are 4x6 not 5x7
#stores corresponding 3d points
objset = np.zeros((4 * 6, 3), np.float32)
#write coordinates of 6x4 chessboard corners
objset[:, :2] =  np.mgrid[0:6, 0:4].T.reshape(-1,2) 

#relative path folder to search for images
# images = glob.glob('/images/*.jpg')
directory = os.path.dirname(__file__)
images_path = os.path.join(directory, 'images')
# images_path = r'C:\Users\Nekaz\Desktop\masters project other copy\Camera callibration\images'
#loop through all images in folder
count = 0
for fname in os.listdir(images_path):
    #check if file is jpg
        if fname.endswith("jpg"):
            #read image
            img = cv.imread(os.path.join(images_path, fname))
            #convert to grayscale
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            #get corners of known size chessboard
            ret, corners = cv.findChessboardCorners(gray, (6,4), None)
            #store point values if corners are found
            if ret == True:
                #stores 2d points
                imgpoints.append(corners)
                #stores 3d points
                objpoints.append(objset)  
            count +=1
            
            # # output result
            # if count == 2:
            #     #draw detected chessboard corners
            #     img = cv.drawChessboardCorners(img, (6,4), corners, ret)
            #     cv.imshow('corners', img)
            #     if cv.waitKey(30) & 0xFF == ord('q'):
            #         break
cv.destroyAllWindows()


#save example of found corners
cv.imwrite( "foundcorners.jpg", img)
shape = (img.shape[1], img.shape[0])
#estimate camera parameters with chessboard points
ret, matrix, distortion, rvectors, tvectors = cv.calibrateCamera(objpoints, imgpoints, shape, None, None)
#save calibration data as pickle file for later usage
data = {}
data["matrix"] = matrix
data["distortion"] = distortion
destination = 'pickle.p'
pickle.dump(data, open( destination, "wb" ) )
