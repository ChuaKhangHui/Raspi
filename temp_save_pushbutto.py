import time

import sys
from imutils.object_detection import non_max_suppression
from imutils import paths
import imutils

import argparse
import cv2

import numpy as np



# Dectection phase
def detector(video_capture,rot_angle, ROI_1, ROI_2, ppl_width):
    
    for i in range (frame_skip):
        video_capture.read()

    #read video
    ret, image = video_capture.read()    
    [height, width, layer] = image.shape;
    
    #rotate
    if (rot_angle != 0):
        (hh, ww) = image.shape[:2]
        center = (ww / 2, hh / 2)
        M = cv2.getRotationMatrix2D(center, rot_angle, 1.0)    
        image = cv2.warpAffine(image, M, (ww, hh))
    
    #mask before resize
    #image = image[ROI_1[1]:ROI_2[1],ROI_1[0]:ROI_2[0]]    
    
    image = imutils.resize(image, width=min(400, image.shape[1]))
    
    ##mask after resize
    resize_ratio = image.shape[1] / float(width) 
        
    max_ppl_size = np.ceil(ppl_width * resize_ratio * 1.4)
    min_ppl_size = np.ceil(ppl_width * resize_ratio * 0.8)
    #print max_ppl_size
    
    ROI_1 = np.int_(np.dot(ROI_1,resize_ratio))   
    ROI_2 = np.int_(np.dot(ROI_2,resize_ratio))
    #print ROI_1
    
    image = image[ROI_1[1]:ROI_2[1],ROI_1[0]:ROI_2[0]] 
    
    #Display - origin 
    orig = image.copy()
    
    #detect
    (rects, weights) = hog.detectMultiScale(image, winStride=(8, 8),
     padding=(8, 8), scale=1.05)
    
    ##delete large rect
    i = 0
    while (i < len(rects)):
        print ("ppp")
        print (float(rects[i][2]))
        if (rects[i][2] > max_ppl_size or rects[i][2] < min_ppl_size):
            #[x,y,w,h] = rects[i]
            #cv2.rectangle(orig, (x, y), (x + w, y + h), (255, 0, 0), 2)
            rects = np.delete(rects,i,0)
        else:
            #[x,y,w,h] = rects[i]
            #cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
            i += 1
     
    #Display - origin 
#    for (x, y, w, h) in rects:
#        #box size validation
#        print ('w = ' + str(w))
#        if (w < 100):
#            cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    #Display - origin 
    cv2.imshow('Original', orig)
    
    #combine rectangle
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    
    
    people = 0;
	# draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        #box size validation  
        #if (xB - xA < 100):
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
        people += 1
        
    #Display - result
    cv2.imshow("HOG + NMS", image)
    
#    if(random.random() > 0.8):
#            print ("save")
#            fname = "./save/" + time.strftime("%m_%d_%H_%M_%S")+ ".jpg"
#            cv2.imwrite(fname,image)
#            
    if people > 0:
        return True
    else:
        return False

#Display - light
def light(img):
    color = (255,255,255)
    cv2.circle(img,(30,20),10,color,-1,0)
    cv2.circle(img,(30,50),10,color,-1,0)
    cv2.circle(img,(30,80),10,color,-1,0)
    
def detecton(img):
    color = (255,0,0)
    cv2.circle(img,(30,20),10,color,-1,0)

def detectoff(img):
    color = (255,255,255)
    cv2.circle(img,(30,20),10,color,-1,0)

def holdon(img,sec):
    color = (0,0,255)
    cv2.circle(img,(30,50),10,color,-1,0)
    cv2.putText(img,sec,(30-5,50+5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255))

def holdoff(img):
    color = (255,255,255)
    cv2.circle(img,(30,50),10,color,-1,0)

def waiton(img,sec):
    color = (255,255,255)
    cv2.circle(img,(30,80),10,color,-1,0)
    cv2.putText(img,sec,(30-7,80+5),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0))
    
def pushon(img,sec):
    color = (0,255,0)
    cv2.circle(img,(30,20),10,color,-1,0)
    cv2.circle(img,(30,50),10,color,-1,0)
    cv2.circle(img,(30,80),10,color,-1,0)
    cv2.putText(img,sec,(30-7,80+5),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255))

def loadpara(filename):    
    global loaded
    
    try:
        text_file = open(filename, "r")
        temp = []
        lines = text_file.readlines()
        for line in lines:
            print (line)
            inner_list = [elt.strip() for elt in line.split(':')]
            # in alternative, if you need to use the file content as numbers
            # inner_list = [int(elt.strip()) for elt in line.split(',')]
            para = [int(elt.strip()) for elt in inner_list[1].split(',')]        
            temp.append(para)
        loaded = True
        return [temp[0][0],temp[1],temp[2],temp[3][0]]
    
    except IOError:
        print ("Could not open file!")
        print ("use default parameter")
        
        

#parameter
rot_angle = 180
ROI_1 = [71,133]
ROI_2 = [564,475]
ppl_width = 123

loaded = False

paras = loadpara("para.txt")
if (loaded == True):
    [rot_angle, ROI_1, ROI_2, ppl_width] = paras
    
    
#Display - light
height = 100
width = 60
cv2.namedWindow('Light',2)

#Input : video
video_path = "C:/Users/hui/Desktop/FYP/FYP_input_video/day.avi"
video_capture = cv2.VideoCapture(video_path)

#rain : 1580
#day : 380

real_fps = video_capture.get(5)
print ("fps : " + str(real_fps))
#video_capture.set(5,5)

##assume (webcam)
#real_fps = 15

#parameter
fps = 5
processingtime_per_frame = round (1.0 / fps, 3)
frame_skip = int(np.ceil(real_fps / fps) - 1)

#skip 100 frame
for i in range(380):
    video_capture.read()

# Detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


#parameter    
missing_max = 5
hold_max = 20
crossing_time = 1
waitkey_time = 1

#initial state
start_fc = 0
pushbutton = False
missing_fc = missing_max
crossing_start = 0

while (True):
    ##time taken to process one frame    
    s_time = time.time()    
    #print (s_time)
    
    #Display - light
    img = np.zeros((height,width,3), np.uint8)
    light(img)
    
    #crossing...    
    if pushbutton == True:
        pushon(img,str(round(crossing_time - (time.time() - crossing_start),1)))
        if (time.time() - crossing_start > crossing_time):
            print("done")
            #reset timer
            pushbutton = False
            light(img)
            start_fc = 0
            missing_fc = 0
    #not push        
    else: 
        #detect people
        detected = detector(video_capture,rot_angle, ROI_1, ROI_2, ppl_width)
        #print detect
        #start frame count
        start_fc += 1
        
        #detected: detect on, reset missing_fc, hold on
        if (detected):
            detecton(img)
            missing_fc = 0
            holdon(img,str(missing_max - missing_fc))
        #not detected: detect off, missing_fc ++, hold (on :if still alow missing, off : missing too much)
        else:
            detectoff(img)
            missing_fc += 1
            if ((missing_max - missing_fc) > 0):
                holdon(img,str(missing_max - missing_fc))
            else: #(missing_fc > missing_max)
                #demand cancel : missing too much, reset start_fc 
                holdoff(img)
                start_fc = 0
        
        #wait : count down (if consistently detected)
        waiton(img,str(hold_max - start_fc))  
        
        #count down end: PUSH, GREEN
        if (hold_max - start_fc < missing_max - missing_fc):
            detectoff(img)
            holdoff(img)            
            pushon(img,str(crossing_time))
            print("crossing")
            crossing_start = time.time()
            pushbutton = True

        #print ("miss : " + str(missing_fc), "hold : " + str(start_fc))
        #detected = (detector(video_capture) or False)
    cv2.imshow('Light',img)    
    
    ##time taken to process one frame    
    e_time = time.time()
    #print(e_time)
    total = round((e_time - s_time) , 2)
    #print("total time : " + str(total))
    
    #1 sec = fps * processing time per frame
    #if fps = 5, pt = 0.2
    
    waitkey_time = int(round((processingtime_per_frame - total) * 1000))
    if (waitkey_time < 1):
        print ("slow")
        waitkey_time = 1
    print (waitkey_time)
    
    key = cv2.waitKey(0)    
#    if key == ord('z'):
#        detected = ~detected
    if key == ord('q'):
        break
    

video_capture.release()
cv2.destroyAllWindows()
        
    
    

