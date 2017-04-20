import cv2, numpy as np

def fill_holes(image_depth, extra=2):
    kernel = np.ones((3,3),'uint8')
    iterations = 0
    while sum(sum(image_depth==0)) > 0:
        image_depth = cv2.dilate(image_depth,kernel,borderType=cv2.BORDER_REPLICATE)
        iterations += 1
    if iterations>0:
        if extra>0:
            image_depth = cv2.dilate(image_depth,kernel,iterations=extra)
        image_depth = cv2.erode(image_depth,kernel,iterations=iterations+extra)
    return image_depth
