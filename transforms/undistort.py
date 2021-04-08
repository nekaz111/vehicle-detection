#Jeremiah Hsieh Camera Calibration Chessboard image undistort on images and video
#program reads camera parameters calculated by calibration.py


import cv2 as cv
import pickle


#open pickle file
with open('pickle.p', 'rb') as file:
    data = pickle.load(file)
    matrix = data['matrix']       # calibration matrix
    distortion = data['distortion']     # distortion coefficients

#get image
img = cv.imread('distort2.jpg')

#call opencv undistort
undistorted = cv.undistort(img, matrix, distortion, None, matrix)

#show image
cv.imshow('original', img)
cv.imshow('undistorted', undistorted)
cv.waitKey(0)

#write image
cv.imwrite('undistorted2.jpg', undistorted) 




##using it on video seems to be quite slow
# # output_video = []

# cap = cv.VideoCapture('calibration_video.mp4')
# # ret, frame = cap.read()
# # height, width, channels = frame.shape
# # video = cv.VideoWriter('corners.mp4', cv.VideoWriter_fourcc('M','J','P','G'), 20, (width, height))

# #read image data from cap stream
# while(cap.isOpened()):
#   ret, frame = cap.read()
#   if ret == True:
#     #show a frame
#     # cv.imshow('original',frame)
      
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     temp, corners = cv.findChessboardCorners(gray, (6,4), None)
#     if temp == True:
#         img = cv.drawChessboardCorners(frame, (6,4), corners, temp)
        
#         cv.imshow('detected', img)
#         # output_video.append(img)
#     # undistorted = cv.undistort(frame, mtx, dist, None, mtx)
#     # cv.imshow('undistorted', undistorted)
#     #exit
#     else:
#         cv.imshow('detected', frame)
#         # output_video.append(frame)
#     if cv.waitKey(25) & 0xFF == ord('q'):
#             # for i in range(len(output_video)):
#             #     video.write(output_video[i])
#             #     video.release()  
#             break
#   else: 
#     break
# cap.release()
    
    
    
cv.destroyAllWindows()