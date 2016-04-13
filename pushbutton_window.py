import time
from imutils.object_detection import non_max_suppression
import imutils
import argparse
import cv2
import numpy as np
import random

#pi#import RPi.GPIO as GPIO
#pi#GPIO.setmode(GPIO.BCM)
#pi##GPIO- LED
#pi#LED1 = 16
#pi#LED2 = 20
#pi#LED3 = 21
#pi#GPIO.setup(LED1,GPIO.OUT)
#pi#GPIO.setup(LED2,GPIO.OUT)
#pi#GPIO.setup(LED3,GPIO.OUT)
#pi#
#pi#def lighton(LED):
#pi#    GPIO.output(LED, True)
#pi#
#pi#def lightoff(LED):
#pi#    GPIO.output(LED, False)
#pi#
#pi#def pushon():
#pi#    GPIO.output(LED1, True)
#pi#    GPIO.output(LED2, True)
#pi#    GPIO.output(LED3, True)
#pi#    time.sleep(0.1)
#pi#    GPIO.output(LED1, False)
#pi#    GPIO.output(LED2, False)
#pi#    GPIO.output(LED3, False)
#pi#    time.sleep(0.1)

def initialize(video_capture,rot_angle, pt1, pt2, ppl_width):
    #read image
    ret, image = video_capture.read()    
    
    (hh, ww) = image.shape[:2]
    
    #rotate
    M = None;
    if (rot_angle != 0):
        center = (ww / 2, hh / 2)
        M = cv2.getRotationMatrix2D(center, rot_angle, 1.0)    
    
    image = imutils.resize(image, width=min(400, image.shape[1]))
    
    ##mask after resize
    resize_ratio = image.shape[1] / float(ww) 
    
    #max_min_ppl_size  
    ppl_size=[50,100]
    ppl_size[0] = np.ceil(ppl_width * resize_ratio * 1.4)
    ppl_size[1] = np.ceil(ppl_width * resize_ratio * 0.8)
    #print max_ppl_size
    
    ROI_1 = np.int_(np.dot(pt1,resize_ratio))   
    ROI_2 = np.int_(np.dot(pt2,resize_ratio))
    
    
    return [ww, hh, M, ppl_size, ROI_1, ROI_2]
    
# Dectection phase
def detector(video_capture, ww, hh, M, ppl_size, ROI_1, ROI_2, video_input = False):
    global display, sampling, NMS    
    
    if (sampling == True):
        global image
        
        
    if (video_input == True):
        global frame_skip 
        for i in range (frame_skip):
            video_capture.read()
        
    ret, image = video_capture.read()
    
    #rotate
    if (M != None):
        image = cv2.warpAffine(image, M, (ww, hh))
     
    #resize
    image = imutils.resize(image, width=min(400, image.shape[1]))
    
    ##mask after resize    
    image = image[ROI_1[1]:ROI_2[1],ROI_1[0]:ROI_2[0]] 
    
    #Display - origin 
    orig = image.copy()
    
    #detect
    (rects, weights) = hog.detectMultiScale(image, winStride=(8, 8),
     padding=(8, 8), scale=1.05)
    
    ##delete large rect
    i = 0
    while (i < len(rects)):
        if (rects[i][2] > ppl_size[0] or rects[i][2] < ppl_size[1]):
            if (display == True):
                [x,y,w,h] = rects[i]
                cv2.rectangle(orig, (x, y), (x + w, y + h), (255, 0, 0), 2)
            rects = np.delete(rects,i,0)
        else:
            if (display == True):
                [x,y,w,h] = rects[i]
                cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
            i += 1
     
    #Display - origin 
#    for (x, y, w, h) in rects:
#        #box size validation
#        print ('w = ' + str(w))
#        if (w < 100):
#            cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    #Display - origin 
    if (display == True): cv2.imshow('HOG', orig)
    
    #combine rectangle
    if (NMS == True):
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
        
        # draw the final bounding boxes
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
        
        #Display - result
        cv2.imshow("HOG + NMS", image)
    
    people = len(rects);
	    
    if (sampling == True):
        if(random.random() > 0.9):
                #print ("save")
                fname = "./save/all_" + time.strftime("%m_%d_%H_%M_%S")+ ".jpg"
                cv2.imwrite(fname,image)
                
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
    cv2.putText(img,sec,(30-7,80+5),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0))
    
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

def adjust_fps():
    global fps, processingtime_per_frame
    
    print ("fps" + str(fps) + " --->" + str(fps-1))
    fps = fps - 1
    processingtime_per_frame = round (1.0 / fps, 3)
    if (video_input ==True):
        global  frame_skip, real_fps
        frame_skip = int(np.ceil(real_fps / fps) - 1)
        
    global missing_max, hold_max
    missing_max = 1 * fps
    hold_max = 4 * fps
    
#start
parser = argparse.ArgumentParser(description='Compare openCV Haar Cascade and HOG+SVM')
parser.add_argument('-l','--lpara', help='file path for load parameter',required=False)
parser.add_argument('-v','--video', help='path for video input',required=False)
parser.add_argument('-j','--jump',type=int, help='number of frame to jump when start',required=False)
parser.add_argument('-w','--webcam', action="store_true", default=None, help='use webcam for image input',required=False)
parser.add_argument('-r','--raspi', action="store_true", default=None, help='raspi version',required=False)
parser.add_argument('-s','--sampling', action="store_true", default=None, help='enable sampling image',required=False)
parser.add_argument('-n','--nms', action="store_true", default=None, help='enable non_max_suppression',required=False)
parser.add_argument('-d','--display', action="store_true", default=None, help='enable display (the system will become slower)',required=False)
parser.add_argument('-f','--fps', type=int, help='fps to process',required=False)
args = parser.parse_args()

#Default parameter : preprocessing
rot_angle = 180
pt1 = [71,133]
pt2 = [564,475]
ppl_width = 123
loaded = False
para_path  = "para.txt"

#Pi
pi = False;
display = False
NMS = False
sampling = False

if (args.raspi != None):
    pi = True;    

if (args.display != None):
    display = True;    

if (args.nms != None):
    NMS = True;  
    
if (args.sampling != None):
    sampling = True;  
    
#load para
if (args.lpara != None):
    para_path = args.lpara;
paras = loadpara(para_path)
if (loaded == True):
    [rot_angle, pt1, pt2, ppl_width] = paras

#Default image input: webcam
#video_path = 0
video_input = False
frame_skip = 0
video_path = "C:/Users/hui/Desktop/FYP/FYP_input_video/day.avi"
video_input = True
#rain : 1580
#day : 380 

#parameter : push buttom system
fps = 20
if (args.fps != None):
    fps = args.fps 

processingtime_per_frame = round (1.0 / fps, 3)

#Change image input   
if (args.video != None):
    video_path = args.video
    video_input = True
    
if (args.webcam != None):
    video_path = 0
    video_input = False
 
video_capture = cv2.VideoCapture(video_path)
image = []

if (video_input == True):
    #skip frame: for video input
    if (args.jump != None):
        #skip 100 frame
        for i in range(args.jump):
            video_capture.read()
       
    #get fps : video input   
    real_fps = video_capture.get(5)
    print ("fps : " + str(real_fps))

    #set fps : fail !!    
    #video_capture.set(5,5)
    frame_skip = int(np.ceil(real_fps / fps) - 1)

# Detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

#prepossesing
[ww, hh, M, ppl_size, ROI_1, ROI_2] = initialize(video_capture,rot_angle, pt1, pt2, ppl_width)

#Display - light
if (display == True):
    height = 100
    width = 60
    cv2.namedWindow('Light',2)

#parameter : Psuh button timer    
missing_max = 5
hold_max = 20
crossing_time = 3
waitkey_time = 1

missing_max = 1 * fps
hold_max = 4 * fps

#initial state
start_fc = 0
pushbutton = False
missing_fc = missing_max
crossing_start = 0

if (sampling == True):
    saveimage = 100
    count = 0
    
slow = 0

try: 
    while (True):
        ##time taken to process one frame: start    
        s_time = time.time()
        
        #Display - light        
        if (display == True): 
            img = np.zeros((height,width,3), np.uint8)
            light(img)
        
        #Button Pushed 
        if pushbutton == True:
            if (display == True): pushon(img,str(round(crossing_time - (time.time() - crossing_start),1)))
            if (time.time() - crossing_start > crossing_time):
                print("...Push Button Request End")
                #reset
                if (display == True): light(img)
                pushbutton = False            
                start_fc = 0
                missing_fc = 5
        #not pushed        
        else: 
            #detect people
            detected = detector(video_capture, ww, hh, M, ppl_size, ROI_1, ROI_2, video_input)
            
            #start frame count
            start_fc += 1
            
            #detected: detect on, reset missing_fc, hold on
            if (detected):
                missing_fc = 0
                if (display == True):
                    detecton(img)
                    holdon(img,str(missing_max - missing_fc))
                #pi#lighton(LED1)
                #pi#lighton(LED2)
                    
            #not detected: detect off, missing_fc ++, hold (on :if still alow missing, off : missing too much)
            else:
                #pi#lightoff(LED1)
                if (display == True): detectoff(img)
                missing_fc = min(missing_fc + 1, missing_max)
                if ((missing_max - missing_fc) > 0):
                    if (display == True): holdon(img,str(missing_max - missing_fc))
                    #pi#lighton(LED2)
                else: #(missing_fc > missing_max)
                    #demand cancel : missing too much, reset start_fc 
                    if (display == True): holdoff(img)
                    #pi#lighton(LED2)
                    start_fc = 0
            
            #wait : count down (if consistently detected)
            if (display == True): waiton(img,str(hold_max - start_fc))  
            #pi#lighton(LED3)
            
            #count down end: PUSH, GREEN
            if (hold_max - start_fc < missing_max - missing_fc):
                if (display == True):
                    detectoff(img)
                    holdoff(img)
                    pushon(img,str(crossing_time))
                print("Push Button Request Send...")
                crossing_start = time.time()
                if (sampling == True):
                     #print ("save")
                     if (count < saveimage):
                        fname = "./save/push_" + time.strftime("%m_%d_%H_%M_%S")+ ".jpg"
                        cv2.imwrite(fname,image)
                        count = count + 1;
                pushbutton = True
    
            #print ("miss : " + str(missing_fc) + "/" + str(missing_max), "hold : " + str(start_fc) + "/" + str(hold_max))
            #detected = (detector(video_capture) or False)
        
        #Display - light
        if (display == True): cv2.imshow('Light',img)    
        
        ##time taken to process one frame : end 
        e_time = time.time()
    
        total = round((e_time - s_time) , 2)
        #print("total time : " + str(total))
        
        #1 sec = fps * processing time per frame
        #if fps = 5, process_t = 0.2
        
        waitkey_time = int(round((processingtime_per_frame - total) * 1000))
        if (waitkey_time < 1):
            print ("slow")
            slow += 2
            if (slow > 10):
                adjust_fps()
                slow = 0
            waitkey_time = 1
        #print (waitkey_time)
        
        slow = max(slow-1,0)
        
        key = cv2.waitKey(waitkey_time)    
    
        if key == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()
    print ("Stop")
    #pi#GPIO.cleanup()

except KeyboardInterrupt:
    print ("Stop")
    video_capture.release()
    cv2.destroyAllWindows()
    #pi#GPIO.cleanup()
