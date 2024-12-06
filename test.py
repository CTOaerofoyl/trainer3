from backgroundImageProcessor import BIP
from defisheye import Defisheye
from ReverseDefisheye import ReverseDefisheye
import cv2



bip = BIP(saveCamConfig=True)
defisheye = Defisheye()
refish = ReverseDefisheye()


ip = f'192.168.1.222'
cap = cv2.VideoCapture(f'rtsp://admin:PAG00319@{ip}:554/live')
ret,frame = cap.read()

sqr,_ = bip.make_square(frame,ip)
undistorted_image = bip.defish(sqr,ip,fov=145,pfov=128)
h,w = undistorted_image.shape[:2]

# Define the color (BGR format) and thickness
color = (0, 255, 0)  # Green color
thickness = 8  # Thickness of 2 pixels (-1 to fill the rectangle)
x1 =1000
x2 =1500
y1 =1000
y2 =1500
# Draw the rectangle
cv2.rectangle(undistorted_image, (x1,y1), (x2,y2), color, thickness)

refished = refish.distort(undistorted_image,fov=145,pfov=128)

# Define the color (BGR format) and thickness
color = (255, 0, 0)  # Green color
thickness = 4  # Thickness of 2 pixels (-1 to fill the rectangle)
top = refish.get_distorted_coordinates(x1,y1,fov=145,pfov=128,width=w,height=h)
print(top,(x1,y1))
bot = refish.get_distorted_coordinates(x2,y2,fov=145,pfov=128,width=w,height=h)
print(bot,(x2,y2))
# Draw the rectangle
cv2.rectangle(refished, top,bot, color, thickness)

bip.show_images_sidebyside(frame,sqr,undistorted_image,refished)

