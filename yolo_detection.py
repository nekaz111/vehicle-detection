#Jeremiah Hsieh YOLO vehicle detection on video file 
#outputs result as video file detected.mp4
#if have time figure output how to run yolo on gpu?
#not sure if i should do that since setup seems more involved than just installing some libraries and idk how final project is to be showcased
#however runtime is significantly slower on CPU only
#if I have time maybe try and train my own neuralnet? idk if i could gather enough of my own data for this tho and if i just download dataset from internet then it's not THAT much different from just using pretrained weights
#in order to calculate relative speed i would need more reliable vehicle tracking, calculate distance change / time, and a way to measure camera recording vehicle speed

import cv2 as cv
import numpy as np
from transforms.perspective import transformImage, findVanishingPoint, transformCoordinate
import math
import copy

#calculate diagonal distance between camera origin and detected object using basic trigonomatry a^2 + b^2 = c^2
#for now assume middle of camera is origin point, later on maybe refine with lane relative position if enough time?
def approximateDistance(center, dot):
    diagonal = math.sqrt((center[0]-dot[0])**2 + (center[1]- dot[1])**2)
    return diagonal



def detectObjects(filename = 'footage01.mp4'):

    #load yolo pretrained weights/parameters from files
    neuralnet = cv.dnn.readNet("yolo-coco/yolov3.weights", "yolo-coco/yolov3.cfg")
    classes = []
    #loadclass names file
    with open("yolo-coco/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = neuralnet.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in neuralnet.getUnconnectedOutLayers()]
    
    
    #video length to parse in seconds
    #currently manually setting this since conversion takes a while
    videolength = 10
    
    #video image array storage
    output_video = []
    topdown_video = []
    #open video file
    cap = cv.VideoCapture(filename)
    #get current frame
    ret, current = cap.read()
    #video dimensions
    height, width, channels = current.shape
    #video writer that converts image array to video
    detected = cv.VideoWriter('detected.mp4',cv.VideoWriter_fourcc(*'DIVX'), 30, (width, height))
    warpwidth = 600
    warpheight = 800
    framecounter = 0
    center = (warpwidth / 2, warpheight)
    #stores specific identified objects for comparison, 
    # identified = {}
    currentidentified = {}
    previousidentified = {}
    currentdistance = {}
    previousdistance = {}
    currentspeed = {}
    #numerical label for identified car
    tagcounter = 1
    
    #calculate vanishing point
    # vpimg = cv.imread("transforms/footage01_trim_shot.jpg")
    # vx, vy, lines = findVanishingPoint(vpimg)
    vx, vy, lines = findVanishingPoint(current)
    #not sure if updatingthis constantly is possible considering there are many shots where the road lines are not visible
    #mph speed parameter that could be linked to either speed estimation value from camera using street lines or from gps value
    #however i did not manage to implement this so currently just hard coded to 40mp cuz that is about the speed limit of the road near my house where the footage was taken.
    cameraspeed = 40
    
    #video read loop
    while True:
        currentidentified = {}
        # #check if finding new vanishing point is possible
        # newvx, newvy, newlines = findVanishingPoint(current)
        # if newlines is not None:
        #     vx = newvx
        #     vy = newvy
        #     lines = newlines
        #get current frame
        ret, current = cap.read()
    
        #opencv blobfromimage parsing for neural network input
        blob = cv.dnn.blobFromImage(current, 1.0/255.0, (416, 416), (0, 0, 0), True, crop = False)
        neuralnet.setInput(blob)
        #run through neural network layers
        layers = neuralnet.forward(output_layers)
        #prediction outputs
        class_ids = []
        confidences = []
        boxes = []
        for results in layers:
            for detection in results:
                #confidence level
                scores = detection[5:]
                #type of object
                class_id = np.argmax(scores)
                #confidence levels of specific objects
                confidence = scores[class_id]
                #confidence threshold 
                # if confidence > 0.2:
                if confidence > 0.3:
                    #detected object coordinates
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    #add detected bounding boxes to array
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)
        
        
        
        #perform perspective transform on image
        #return warp matrix that is used to calculate distance in transformed image
        transform_view, transform_matrix, feetperpixelx, feetperpixely = transformImage(current, vx, vy, lines)
        
        #loop through detected objects and draw bounding boxes
        for i in range(0, len(boxes)):
            if i in indexes:
                tag = ''
                #identified object id
                label = str(classes[class_ids[i]])
                #confidences percentage
                confidence = confidences[i]
                #bounding box coordinates
                x, y, w, h = boxes[i]
                #ghetto way of removing objects we don't care about
                if label == 'car' or label == 'bus' or label == 'truck' or label == 'motorbike':
                    
                    #ghetto method of ignoring bounding box on car dash itself since the yolo object recognition see that is also a vehicle
                    if y < height * (3/4):
                        #add to list of found objects if a nearby box does not already exist
                        #main issue with this simple method is that occoluding objects will probably get mixed up
                        #also due to the instability of YOLO object detection it keeps bumping the counter up as the bounding boxes pop in and out
                        #if time permits maybe implement some kind of color histogram relative comparison although that still may not help with occolusion
                        for i in previousidentified:
                            
                            px, py, pw, ph = previousidentified[i]
                            #check if current object is within 5 pixel values of already found object
                            if (px - 20 < x < px + 20) and (py - 20 < y < py + 20) and (pw - 80 < w < pw + 80) and (ph - 80 < h < ph + 80):
                                currentidentified[i] = x, y, w, h
                                tag = str(i)

                        # identified[label] = (x, y, w, h)
                        if tag == '':
                            currentidentified[tagcounter] = x, y, w, h
                            tag = str(tagcounter)
                            tagcounter += 1
                            
                            
                        #draw object bounding box on image
                        cv.rectangle(current, (x, y), (x + w, y + h), (0, 255, 0, 3), 2)
                    
                        #bottom center of detected object bounding box coordinate
                        coord = (x + (w / 2), y + h)
                        
                    
                        #transform normal x/y coordinate to top down equivalent
                        tx, ty = transformCoordinate(coord[0], coord[1], transform_matrix)
                        
                        #calculate distance from bottom center of bounding box to center of camera
                        
                        #convert pixels to equivalent distance
                        # estimated = diagonal / pixelwidth 
                        #find verticaldiff distance from object to camera
                        verticaldiff = warpheight - int(ty)
                        #multiply distance by number of feet per pixel
                        actualdistance = verticaldiff * feetperpixely
                        
                        #divide by 3.281 to convert feet to meters
                        actualdistance /= 3.281
                        # estimated = verticaldiff / pixelheight
                        
                        currentdistance[tag] = actualdistance
                        # calculate speed as distance / time
                        # increase update time to decrease fluctuation?
                        #update once every second?
                        #turns out the jittering in the topdown conversion is what is causing my distance and speed estimations to be all crap
                        
                        if len(previousdistance) > 0 and framecounter % 10 == 0:
                            if tag in previousdistance:
                                #meters per second
                                speed = (currentdistance[tag] - previousdistance[tag]) / (0.3)
                                #convert to miles per hour
                                speed *= 2.23694
                                currentspeed[tag] = speed
                         
                        #         # cv.putText(current, "vehicle " + tag + ": " + "{0:.1f}".format(speed) + " mph" ,  (x , y) , cv.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
                        #         # cv.putText(current, "vehicle " + tag + ": " + str(int(speed)) + " mph" ,  (x , y) , cv.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
                        #         if speed > 0:
                        #             cv.putText(current, "vehicle " + tag + ": speeding" ,  (x , y) , cv.FONT_HERSHEY_PLAIN, 1, (0,255,255), 2)
                        #         else:
                        #             cv.putText(current, "vehicle " + tag + ": not speeding" ,  (x , y) , cv.FONT_HERSHEY_PLAIN, 1, (255,0,255), 2)
                        # else:
                        #     if tag in currentspeed:
                        #         # cv.putText(current, "vehicle " + tag + ": " + "{0:.1f}".format(speed) + " mph" ,  (x , y) , cv.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
                        #         # cv.putText(current, "vehicle " + tag + ": " + str(int(speed)) + " mph" ,  (x , y) , cv.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
                        #         if speed > 0:
                        #             cv.putText(current, "vehicle " + tag + ": speeding" ,  (x , y) , cv.FONT_HERSHEY_PLAIN, 1, (0,255,255), 2)
                        #         else:
                        #             cv.putText(current, "vehicle " + tag + ": not speeding" ,  (x , y) , cv.FONT_HERSHEY_PLAIN, 1, (255,0,255), 2)
                        #     else:
                        #         # cv.putText(current, "vehicle " + tag + ": " + "{0:.1f}".format(actualdistance) + " meters" ,  (x , y) , cv.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
                        #         cv.putText(current, "vehicle " + tag + ": " + str(int(actualdistance)) + " meters" ,  (x , y) , cv.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
                        #         # cv.putText(current, "vehicle " + tag + ": not speeding" ,  (x , y) , cv.FONT_HERSHEY_PLAIN, 1, (255,0,255), 2)
                        
                        cv.putText(current, "vehicle " + tag + ": " + str(int(actualdistance)) + " meters" ,  (x , y) , cv.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
                        cv.circle(transform_view,(int(tx), int(ty)), 5, (255,0,0), -1) 
                        
                        #measure change in distance over time to get speed
        
        
        # cv.putText(current, "camera speed: " + str(cameraspeed) + " mph" ,  (50 , 50) , cv.FONT_HERSHEY_PLAIN, 2, (0,255,255), 2)            
        #save values
        previousidentified = copy.deepcopy(currentidentified)
        if framecounter % 10 == 0 or framecounter == 0:
            previousdistance = copy.deepcopy(currentdistance)
        
        #add images to array for video conversion
        topdown_video.append(transform_view)   
        output_video.append(current)
        #display image frame
        # cv.imshow("Image", current)
        #add modified image to array for video detected

        framecounter +=1
        
        # key = cv.waitKey(1)
        # if key == 27:
        #     break
        cv.waitKey(1)
        #press q to quit video or wait until 300 frames are read and write to detected
        if framecounter == 30 * videolength: 
            
        # if cv.waitKey(30) & 0xFF == ord('q'):
            #write video to file
            for i in range(len(output_video)):
                detected.write(output_video[i])
            detected.release()
                        
            #checking if output is correct, 
            topdown = cv.VideoWriter('topdown.mp4',cv.VideoWriter_fourcc(*'DIVX'), 30, (600, 800))
            for i in range(len(topdown_video)):
                topdown.write(topdown_video[i])
            topdown.release()
            break
        
    
    cap.release()
    cv.destroyAllWindows()


#main function call
if __name__ == "__main__":    
    detectObjects()