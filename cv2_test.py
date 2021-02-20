import cv2

img = cv2.imread('/home/ncl/ADD_sy/inference/sg_inference/visualize/bbox/bbox_detection_0.jpg')
img2 = cv2.imread('/home/ncl/ADD_sy/inference/sg_inference/visualize/sg_result/0_sg.png')
cv2.imshow('sg', img);cv2.waitKey(1)
cv2.imshow('image',img2)
cv2.waitKey(1)
