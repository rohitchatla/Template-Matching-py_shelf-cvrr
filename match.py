#Template matching


import cv2
import numpy as np
self = cv2.imread("self.jpg") # core image
gray_img = cv2.cvtColor(self, cv2.COLOR_BGR2GRAY)
mylogo = cv2.imread("mylogo.jpg", cv2.IMREAD_GRAYSCALE) # template
w,h = mylogo.shape[::-1]
result = cv2.matchTemplate(gray_img, mylogo, cv2.TM_CCOEFF_NORMED)
loct = np.where(result >= 0.9)

# threshold = 0.8
# loc = np.where( res >= threshold)

for pt in zip(*loct[::-1]):

    cv2.rectangle(self, pt, (pt[0] + w, pt[1] + h), (147, 112, 219), 1) # mediumpurple:(147,112,219)

cv2.imshow("self", self)
cv2.waitKey(0)
cv2.destroyAllWindows()








# Realtime Template matching

# import cv2
# import numpy as np
# cap = cv2.VideoCapture(0)
# template = cv2.imread("self.jpg", cv2.IMREAD_GRAYSCALE)
# w, h = template.shape[::-1]
# while True:
#     _, frame = cap.read()
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
#     loc = np.where(res >= 0.7)
#     for pt in zip(*loc[::-1]):
#         cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 3)
#     cv2.imshow("Frame", frame)
#     key = cv2.waitKey(1)
#     if key == 27:
#         break
# cap.release()
# cv2.destroyAllWindows()















# import cv2
# import numpy as np
#
# img_rgb = cv2.imread('self.jpg')
# img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
#
# template = cv2.imread('mylogo.jpg',0)
# w, h = template.shape[::-1]
#
# res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
# threshold = 0.8
# loc = np.where( res >= threshold)
#
# for pt in zip(*loc[::-1]):
#     cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)
#
# cv2.imshow('Detected',img_rgb)
#
# cv2.waitKey()
# cv2.destroyAllWindows()








