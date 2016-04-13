
compare README
  optional arguments:
  -h, --help            show this help message and exit
  -v VIDEO, --video VIDEO
                        video path
  -r, --raspi           raspi version
  -d, --display         disable display window
  
  default : day.avi
  
  (q) quit
  
  
  configure README
  
  optional arguments:
  -h, --help            show this help message and exit
  -l LPARA, --lpara LPARA
                        file path for load parameter
  -s SPARA, --spara SPARA
                        file path for save parameter
  -v VIDEO, --video VIDEO
                        path for video input
  -i IMAGE, --image IMAGE
                        path for image input
  -w, --webcam          use webcam for image input
  -r, --raspi           raspi version
  
  default:
  LPARA = para.txt
  SPARA = para.txt
  input image: test.jpg
  
  (r) reload image
  (q) quit
  (s) save parameter
  
  
  psuhbutton README
  optional arguments:
  -h, --help            show this help message and exit
  -l LPARA, --lpara LPARA
                        file path for load parameter
  -v VIDEO, --video VIDEO
                        path for video input
  -j JUMP, --jump JUMP  number of frame to jump when start
  -w, --webcam          use webcam for image input
  -r, --raspi           raspi version
  -s, --sampling        enable sampling image
  -n, --nms             enable non_max_suppression
  -d, --display         enable display (the system will become slower)
  -f FPS, --fps FPS     fps to process
  
  default:
  LPARA = para.txt
  input image: webcam
  fps = 25
  
  uncomment #pi# for pi version
  
  (q) quit