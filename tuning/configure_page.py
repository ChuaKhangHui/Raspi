import cv2 
import numpy as np

import argparse

#function
def wirtepara(filename):
    global rot_angle, firstpt, secondpt, people_size    
    with open(filename, 'w') as f:
        f.write('rot_angle :' + str(rot_angle) + '\n')
        f.write('1st point :' + str(firstpt[0]) + ',' + str(firstpt[1]) + '\n')
        f.write('2nd point :' + str(secondpt[0]) + ',' + str(secondpt[1]) + '\n')
        f.write('ppl_size :' + str(people_size))
    print ('parameter saved')                
        
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
    
def nothing(x):
    pass

def draw_ROI(event,x,y,flags,param):
    global draw,firstpt,secondpt,img_draw_rect, img_rot
    
    #draw_rect = False
    if event == cv2.EVENT_LBUTTONDOWN:
        draw = True
        firstpt = [x,y]
#        print (str(x) + ", " + str(y) )
    elif event == cv2.EVENT_MOUSEMOVE:
        if draw == True:
            secondpt = [x,y]
            #draw_rect = True
            
    elif event == cv2.EVENT_LBUTTONUP:
        draw = False
        secondpt = [x,y]
        #draw_rect = True
        #update_mask()
        
    #img_draw_rect = img_rot.copy()
    #if draw_rect == True:
        #cv2.rectangle(img_draw_rect,(firstpt[0],firstpt[1]) ,(secondpt[0],secondpt[1]),(0,255,0),2)
    #cv2.imshow("draw",img_draw_rect)        
    #cv2.waitKey(1)
    
def update_mask():
    validate_secondpt()
    global masking,firstpt,secondpt
    
    masking_temp = np.zeros_like(img_ori)
    masking_temp = masking_temp + 255

    mask = np.zeros((height,width),'uint8')
    #mask = mask + 1;
    
    for i in range(firstpt[0],secondpt[0]):
        for j in range(firstpt[1],secondpt[1]):
            mask[j][i] = 1
    
    for i in range(3):
        masking_temp[:,:,1] = masking_temp[:,:,1] * mask
    masking = masking_temp

def validate_secondpt():
    global secondpt, width, height
    if secondpt[0] >= width:
        secondpt[0] = int(width)
    if secondpt[1] >= height:
        secondpt[1] = int(height)


#parameter
rot_angle = 180
firstpt = [20,20]
secondpt = [200,200]
people_size = 200
# (firstpt[0],firstpt[1]) ,(secondpt[0],secondpt[1])

loaded = False
para_path  = "para.txt"
save_para_path = "para.txt"
draw = False
#start
video_input = False;
pi = False;
 
parser = argparse.ArgumentParser(description='Compare openCV Haar Cascade and HOG+SVM')
parser.add_argument('-l','--lpara', help='file path for load parameter',required=False)
parser.add_argument('-s','--spara', help='file path for save parameter',required=False)
parser.add_argument('-v','--video', help='path for video input',required=False)
parser.add_argument('-i','--image', help='path for image input',required=False)
parser.add_argument('-w','--webcam', action="store_true", help='use webcam for image input',required=False)
parser.add_argument('-r','--raspi', action="store_true", default=None, help='raspi version',required=False)
args = parser.parse_args()

if (args.raspi != None):
    pi = True;    

if (args.lpara != None):
    para_path = args.lpara;
 
if (args.spara != None):
    save_para_path = args.spara;   

paras = loadpara(para_path)
if (loaded == True):
    [rot_angle, firstpt, secondpt, people_size] = paras

#read image
#default
img_ori = cv2.imread("test.jpg")
    
if (args.image != None):
    img_path = args.image
    img_ori = cv2.imread(img_path)
    
if (args.video != None):
    video_path = args.video
    video_input = True
    
if (args.webcam != None):
    video_path = 0
    video_input = True
    
if (video_input == True):
    video_capture = cv2.VideoCapture(video_path)
    for i in range(3):
        video_capture.read()
    rect, img_ori = video_capture.read()


height, width, layer = img_ori.shape

#rotate required input
#(hh, ww) = img_ori.shape[:2]
center = (width / 2, height / 2)

#masking
global masking

#people
people_size = 200
img_people = cv2.imread("people.png", -1)
#cv2.imshow('people',img_people)
people_height, people_width, layer = img_people.shape

#cv2.imshow('ppl',img_people_resize)
people_x = int(center[0])
people_y = int(center[1])

cv2.namedWindow('ori')
cv2.imshow("ori",img_ori)

cv2.namedWindow('result')
#cv2.namedWindow('draw')

cv2.createTrackbar('Angle' , 'ori', rot_angle, 360, nothing)
min_ratio = np.min((float(width)/people_width, float(height)/people_height))
cv2.createTrackbar('ppl_size' , 'ori', people_size,int(people_width * min_ratio), nothing)
cv2.createTrackbar('ppl_x' , 'ori', people_x, int(width), nothing)
cv2.createTrackbar('ppl_y' , 'ori', people_y, int(height), nothing)

cv2.setMouseCallback("result",draw_ROI)

#cv2.imshow('draw',img_ori)

while (True):
    #get a copy of img_ori
    img = img_ori.copy()
    
    #rotate    
    rot_angle = cv2.getTrackbarPos('Angle' , 'ori')
    M = cv2.getRotationMatrix2D(center, rot_angle, 1.0)    
    img_rot = cv2.warpAffine(img, M, (width, height))
    #cv2.imshow("temp_rot",img_rot)
    
    #img_draw_rect = img_rot.copy()        
    #cv2.imshow("draw",img_rot)     
    
    #ROI rectangle
    cv2.rectangle(img_rot,(firstpt[0],firstpt[1]) ,(secondpt[0],secondpt[1]),(0,255,0),2)
    #cv2.imshow("temp_rot_rect",img_rot)
    
    update_mask()
    ROI = cv2.addWeighted(img_rot,0.6,masking,0.4,0)
    #cv2.imshow("temp_mask",ROI)    
    
    people_size = cv2.getTrackbarPos('ppl_size' , 'ori')
    if (people_size < 1):
        people_size = 1
    resize_ratio = float(people_size) / people_width
    img_people_resize = cv2.resize(img_people,(0,0),fx= resize_ratio,fy =resize_ratio)
    sppl_height, sppl_width, layer = img_people_resize.shape
    
    people_x = cv2.getTrackbarPos('ppl_x' , 'ori')
    people_y = cv2.getTrackbarPos('ppl_y' , 'ori')
    if (people_x < sppl_width/2 + 1):
        people_x = sppl_width/2 + 1
    elif (people_x > width - sppl_width/2):
        people_x = width - sppl_width/2
        
    if (people_y < sppl_height/2 + 1):
        people_y = sppl_height/2 + 1
    elif (people_y > height - sppl_height/2):
        people_y = height - sppl_height/2
    
    x_offset = people_x - sppl_width/2
    y_offset = people_y - sppl_height/2
    x_offset2 = people_x + (sppl_width - sppl_width/2)
    y_offset2 = people_y + (sppl_height - sppl_height/2)
    
    for c in range(0,3):    
        ROI[y_offset - 1:y_offset2 -1 , x_offset - 1:x_offset2 - 1, c] = np.uint8(img_people_resize[:,:,c] * (img_people_resize[:,:,3]/255.0) + ROI[y_offset - 1:y_offset2 -1 , x_offset - 1:x_offset2 - 1, c] * (1.0 - img_people_resize[:,:,3]/255.0))
     
    cv2.imshow("result",ROI)

    if (pi == True): key = cv2.waitKey(0)
    else: key = cv2.waitKey(1)
    
    if key == ord('q'):
        break
    
    if key == ord('r'):
        rect, img_ori = video_capture.read()
        cv2.imshow("ori",img_ori)
    
    if key == ord('s'):
        wirtepara(save_para_path)
    
cv2.destroyAllWindows()
video_capture.release()
