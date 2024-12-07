from backgroundImageProcessor import BIP
from roiSelector import RotatedRectangleDrawer
import cv2

bip = BIP()
roi = RotatedRectangleDrawer()

for i in range(6,8):
    ip = f'192.168.1.22{i}'
    cap = cv2.VideoCapture(f'rtsp://admin:PAG00319@{ip}:554/live')
    ret,frame = cap.read()

    sqr_img,_ = bip.make_square(image=frame,ip=ip)
    # bip.show_images_sidebyside(sqr_img)
    undistorted_image = bip.defish(sqr_img,ip,fov=145,pfov=128)
    # img = cv2.imread('Crops/bag1_1.png')
    # bip.calculateScaleFactor(ip,img,undistorted_image)

    rois = roi.draw(undistorted_image)
    bip.saveROIs(ip,rois)
    cv2.imwrite(f'undistortedBeltImages/{ip}.jpg',undistorted_image)


## run for test
# img = cv2.imread(f'undistortedBeltImages/{ip}.jpg')

# im2= bip.refish(img,ip)
# im2 = bip.remove_padding(im2,ip)
# bip.show_images_sidebyside(frame,im2)

