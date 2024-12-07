from backgroundImageProcessor import BIP
from defisheye import Defisheye
from ReverseDefisheye import ReverseDefisheye
import cv2
import numpy as np





bip = BIP(saveCamConfig=True)
defisheye = Defisheye()
refish = ReverseDefisheye()


ip = f'192.168.1.221'
cap = cv2.VideoCapture(f'rtsp://admin:PAG00319@{ip}:554/live')
ret,frame = cap.read()

sqr,_ = bip.make_square(frame,ip)
xs,ys,undistorted_image =defisheye.getMaps(sqr,fov=145,pfov=128)
data = np.load('defisheye_maps.npz')

# Access individual matrices
xs = data['xs']
ys = data['ys']

h,w = undistorted_image.shape[:2]

# Define the color (BGR format) and thickness
color = (0, 255, 0)  # Green color
thickness = 8  # Thickness of 2 pixels (-1 to fill the rectangle)
x1 =1827
x2 =2363
y1 =1343
y2 =1800
# Draw the rectangle
cv2.rectangle(undistorted_image, (x1,y1), (x2,y2), color, thickness)

refished = refish.distort(undistorted_image,fov=145,pfov=128)

# Define the color (BGR format) and thickness
color = (255, 0, 0)  # Green color
thickness = 4  # Thickness of 2 pixels (-1 to fill the rectangle)
# top = refish.get_distorted_coordinates(y1,x1,fov=145,pfov=128,width=w,height=h)
x_tl = int(ys[x1,y1])
x_bl = int(ys[x1,y2])
x_tr = int(ys[x2,y1])
x_br = int(ys[x2,y2])
y_tl = int(xs[x1,y1])
y_bl = int(xs[x1,y2])
y_tr = int(xs[x2,y1])
y_br = int(xs[x2,y2])

top = (min(x_tl,x_bl),min(y_tl,y_tr))
bot = (max(x_tr,x_br),max(y_bl,y_br))


print(top,(x1,y1),(xs[x1,y1],ys[x1,y1]))
# bot = refish.get_distorted_coordinates(y2,x2,fov=145,pfov=128,width=w,height=h)
print(bot,(x2,y2))
# Draw the rectangle
cv2.rectangle(refished, top,bot, color, thickness)

bip.show_images_sidebyside(frame,sqr,undistorted_image,refished)

