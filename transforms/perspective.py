#Jeremiah Hsieh Perspective Warping
#this program uses camera information gathered by calibration.py to use relative transform on an image and calculate equivalent top down image 
#overall project methodology is from the udacity self driving car nanodegree course
#i think maybe another issue is camera quality? cuz getting transformed lines further away from the  camera is really blurry even for a transform



import cv2 as cv
import pickle
import numpy as np


#store x/y coordinates from mouseclick
right_clicks = list()

#temporary manual eyeballing tools until i figure out how to automate road intersection and vanishing point
#get coordinates at mouse click on window
def mouseclick(event, x, y, flags, params):

    #right-click event value is 2
    if event == 2:
        global right_clicks

        #store the coordinates of the right-click event
        right_clicks.append([x, y])

        #this just verifies that the mouse data is being collected
        #you probably want to remove this later
        print(right_clicks)

#given original x and y coordinates in image, make relative transform using warp  matrix
def transformCoordinate(x, y, transform_matrix):
    tx = (transform_matrix[0][0] * x + transform_matrix[0][1] * y + transform_matrix[0][2]) / ((transform_matrix[2][0] * x + transform_matrix[2][1] * y + transform_matrix[2][2]))
    ty = (transform_matrix[1][0] * x + transform_matrix[1][1] * y + transform_matrix[1][2]) / ((transform_matrix[2][0] * x + transform_matrix[2][1] * y + transform_matrix[2][2]))
    return tx, ty
    

def findVanishingPoint(img):
    height = img.shape[0]
    width = img.shape[1] 
    left = np.zeros((2,2), dtype= np.float32)
    right = np.zeros((2,1), dtype= np.float32)  
    roi_points = np.array([[0, height - 50],[width, height-50], [width // 2, height // 2 + 50]], dtype=np.int32)
    roi = np.zeros((height, width), dtype=np.uint8)
    cv.fillPoly(roi, [roi_points], 1)
    hls = cv.cvtColor(img, cv.COLOR_RGB2HLS)
    
    #color ranges
    white_lower = np.array([np.round(  0 / 2), np.round(0.75 * 255), np.round(0.00 * 255)])
    white_upper = np.array([np.round(360 / 2), np.round(1.00 * 255), np.round(0.30 * 255)])
    white_mask = cv.inRange(hls, white_lower, white_upper)
    # #incorrect range?
    # yellow_lower = np.array([np.round( 40 / 2), np.round(0.00 * 255), np.round(0.35 * 255)])
    # yellow_upper = np.array([np.round( 60 / 2), np.round(1.00 * 255), np.round(1.00 * 255)])
    # yellow_mask = cv.inRange(hls, yellow_lower, yellow_upper)
    
    # #generate mask of each colors
    # mask = cv.bitwise_or(yellow_mask, white_mask)
    # masked = cv.bitwise_and(img, img, mask = mask)
    # cv.imshow("white", white_mask)
    # # cv.imshow("yellow", yellow_mask)
    # cv.imshow("final", masked)
    # cv.imshow('original', img)1

    #canny edge detection
    edges = cv.Canny(white_mask, 200, 100)
    #extrapolate lines in image
    lines = cv.HoughLinesP(edges * roi, 0.5, np.pi/180, 10, None, 100, 150) 
    
    #combine and calculate lines based on slope (assume negative slope is right lane and positive slope is left lane)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                normal = np.array([[-(y2-y1)], [x2-x1]], dtype = np.float32)
                normal /= np.linalg.norm(normal)
                point = np.array([[x1],[y1]], dtype = np.float32)
                outer = np.matmul(normal, normal.T)
                left += outer
                right += np.matmul(outer, point)
                cv.line(img, (x1,y1), (x2, y2),(255, 0, 0), thickness = 2)
        # calculate the vanishing point
    vanishing_point = np.matmul(np.linalg.inv(left), right)

    # cv.imshow('edges', edges)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    if lines is not None:
        return vanishing_point[0][0], vanishing_point[1][0], lines
    else:
        return None, None, None



#takes image and detected object bounding box coordinates and performs transforms to approximate distance from camera 
def transformImage(img = None, vx = None, vy = None, lines = None):
    trigger = 0
    # #read image
    if img is None:
        img = cv.imread("footage01_trim_shot.jpg")
        trigger = 1

    height = img.shape[0]
    width = img.shape[1] 

    warpheight = 800
    warpwidth = 600
    #open pickle file containing camera matrix/coefficients found from calibration program
    with open('pickle.p', 'rb') as file:
        data = pickle.load(file)
        #camera calibration matrix
        matrix = data['matrix']   
        # camera distortion coefficients    
        distortion = data['distortion']     
    
    # #use camera matrix to calculate relative position in the y direction
    # print('camera matrix')
    # print(matrix)
    # print('distortion matrix')
    # print(distortion)
        
    

    #calculate vanishing point
    if trigger == 1:
        vx, vy, lines = findVanishingPoint(img)
        #draw vanishing point
        cv.circle(img,(vx, vy), 5, (255,0,0), -1) 
    
    # #original point on image
    # originaltrapazoid = [(800, 770), (480, 930), (1180, 930), (930, 770)]
    
    # #manually set points
    # #figure out how to do this automatically?
    # #top left
    # cv.circle(img,(800, 770), 5, (0,0,255), -1)
    # #bottom left
    # cv.circle(img,(480, 930), 5, (0,0,255), -1)
    # #bottom right
    # cv.circle(img,(1180, 930), 5, (0,0,255), -1)
    # #top right
    # cv.circle(img,(930, 770), 5, (0,0,255), -1)
    
    top = vy + 20
    # top = vy + 40
    bottom = height-100
    
    #top left and right points of trapzeoid
    p1 = (int(vx - 100), int(top))
    p2 = (int(vx + 100), int(top))
    #i also seme to have a chicken or the egg problem where i would like to use lane width to calculate optimal trapezoid values to use (ideally encompassing 3 lane widths wide) but in order to get the best top down transform values to measure the lane dimensions i need to calculate the trapezoid first
    # p1 = (int(vx - 200), int(top))
    # p2 = (int(vx + 200), int(top))
    #bottem left and right points of trapezoid calculated by finding the intersection of the bottom horizontal, p1/p2, and the vanishing point
    p3 = (int(p2[0] + (vx-p2[0]) / float(vy - p2[1]) * (bottom - p2[1])), int(bottom))
    # p3 = on_line(p2, vanishingpoint, bottom)
    # p4 = on_line(p1, vanishingpoint, bottom)
    p4 = (int(p1[0] + (vx - p1[0]) / float(vy - p1[1]) * (bottom - p1[1])), int(bottom))


    
    # originaltrapazoid = np.array([p1,p2,p3,p4], dtype=np.float32)
    # print(originaltrapazoid)
    # cv.polylines(img, [originaltrapazoid.astype(np.int32)],True, (0,0,255), thickness = 3)
    #original point on image
    # originaltrapazoid = [(610, 670), (-10, 850), (1670, 850), (1020, 670)]
    # originaltrapazoid = [originaltrapazoid[0], originaltrapazoid[3], originaltrapazoid[2], originaltrapazoid[1]]
    originaltrapazoid = [p1, p2, p3, p4]
    
    # #manually set points
    # #figure out how to do this automatically?
    # #vanishing point and 
    # #top left
    # cv.circle(img, (610, 670), 5, (0,0,255), -1)
    # #bottom left
    # cv.circle(img,(-10, 850), 5, (0,0,255), -1)
    # #bottom right
    # cv.circle(img,(1670, 850), 5, (0,0,255), -1)
    # #top right
    # cv.circle(img,(1020, 670), 5, (0,0,255), -1)
    

    
    #top left, top right, bottom right, bottom left
    final = np.array([[0, 0], [600, 0],[600, 800], [0, 800]], dtype=np.float32)
    
    #corresponding transformed birds eye equivalent of image
    #car lane is assumed standard 12 feet wide
    #dashes are 10 feet long?
    #top left, bottom left, bottom right, top right
    # distortedtrapezoid = [(450, 0), (450, 950), (1250, 950), (1250, 0)]
    
    
    # Topdown = Topdown(originaltrapazoid, distortedtrapezoid, matrix, distortion)
    
    # undistorted = Topdown.undistort(img)
    
    #undistort image using known camera parameters
    undistorted = cv.undistort(img, matrix, distortion, None, matrix)
    #image size
    shape = (undistorted.shape[1], undistorted.shape[0])
    #convert to array for transformation
    initial = np.array(originaltrapazoid, np.float32)
    # final = np.array(distortedtrapezoid, np.float32)
    
    #map corresponding points to each other
    #change it from lane lines to include adjacent lanes using a larger trapezoid?
    warp_matrix = cv.getPerspectiveTransform(initial, final)
    #convert to top down view
    # topdown = cv.warpPerspective(undistorted, warp_matrix, shape, flags = cv.INTER_LINEAR)
    topdown = cv.warpPerspective(undistorted, warp_matrix, (warpwidth, warpheight), flags = cv.INTER_LINEAR)
    
       
        
    
    #revert perspective transform from top down back to original vie
    revert_matrix = cv.getPerspectiveTransform(final, initial)
    revert = cv.warpPerspective(topdown, revert_matrix, shape, flags = cv.INTER_LINEAR)
    
    

    #calculate the width of the road lane in pixels
    closestleft = 0
    closestright = warpwidth
    for line in lines:
        for x1, y1, x2, y2 in line:
            # print(closestleft)
            # print(closestright)
            transformed_line = transformCoordinate(x1, y1, warp_matrix)
            # print("original")
            # print(line)
            # print('transformed')
            # print(transformed_line)
            # cv.circle(topdown,(int(transformed_line[0]), int(transformed_line[1])), 5, (0,0,255), -1)
            if transformed_line[0] < warpwidth/2 and transformed_line[0] > closestleft:
                closestleft = int(transformed_line[0])
            if transformed_line[0] > warpwidth/2 and transformed_line[0] < closestright:
                closestright = int(transformed_line[0])
    #number of pixels equivalent to 12 feet
    pixelwidth = closestright - closestleft
    #use camera and warp matrix to find relative length in the y direction
    relative_matrix = np.linalg.inv(np.matmul(warp_matrix, matrix))
    pixelheight = pixelwidth * np.linalg.norm(relative_matrix[:, 0]) / np.linalg.norm(relative_matrix[:, 1])

    feetperpixelx = 12 / pixelwidth
    feetperpixely = 12 / pixelheight
        
    
    
    
    
    if trigger == 1:
        #draw pixel distance lines
        cv.line(topdown, (closestleft, int(warpheight-pixelheight)), (closestright, int(warpheight-pixelheight)),(0, 0, 255), thickness = 2)
        cv.line(topdown, (closestleft, 800), (closestleft, int(warpheight-pixelheight)),(0, 255, 0), thickness = 2)
      
        print('pixelwidth: ' + str(pixelwidth))
        print('pixelheight: ' + str(pixelheight))

        #show trapezoid that is transformed
        cv.polylines(img, [initial.astype(np.int32)],True, (0,0,255), thickness = 3)

        
        cv.imshow('reverted', revert)
        # cv.imshow('edges', edges)
        # cv.imshow('gray', blur)
        cv.imshow("img", img)
        # cv.imshow('gray', topblur)
        # cv.imshow('topedges', topedges)
        # cv.imshow('grayscale', gray)
        # cv.imshow('edge', edges)
        # cv.imshow("undistorted", undistorted)
        cv.imshow("topdown", topdown)
        #set mouse callback function for window
        cv.setMouseCallback('img', mouseclick)
        # cv.imshow("Perspective transformation", result)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    return topdown, warp_matrix, feetperpixelx, feetperpixely
   
    


#main function call
if __name__ == "__main__":   
    transformImage()