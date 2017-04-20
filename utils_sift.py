import cv2, glob, json, numpy as np, os.path
from matplotlib import pyplot as plt

def compute_sift(image_RGB, mask=None, debug=True):
    gray_image = cv2.cvtColor(image_RGB, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT()
    kp, des = sift.detectAndCompute(gray_image, mask)
    if debug:
        print('%d features detected' % len(kp))
        draw_keypoints(image_RGB,kp)
    return (kp, des)

def pack_keypoint(keypoints, descriptors):
    kpts = np.array([[kp.pt[0], kp.pt[1], kp.size,
                  kp.angle, kp.response, kp.octave,
                  kp.class_id]
                 for kp in keypoints])
    desc = np.array(descriptors)
    return kpts, desc

def unpack_keypoint(kpts,desc):
    try:
        #kpts = array[:,:7]
        #desc = array[:,7:]
        keypoints = [cv2.KeyPoint(x, y, _size, _angle, _response, int(_octave), int(_class_id))
                 for x, y, _size, _angle, _response, _octave, _class_id in list(kpts)]
        return keypoints, np.array(desc)
    except(IndexError, ValueError):
        return np.array([]), np.array([])

def calc_matches(des,des_bin):
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des,des_bin,k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    return good

def draw_keypoints(image,kp):
    image_kp = cv2.drawKeypoints(image.copy(),kp,color=(0,255,0))
    plt.imshow(image_kp),plt.axis('off');
    
def textured_mask(image_rgb,kp):
    image = np.zeros(image_rgb.shape,dtype='uint8')
    image = cv2.drawKeypoints(image,kp,color=(255,255,255))
    image = image[:,:,0]
    kernel = np.ones((5,5),np.uint8)
    image = cv2.dilate(image,kernel,iterations = 10)
    image = cv2.erode( image,kernel,iterations =  5)
    tx_mask = cv2.inRange(image,240,255)
    return tx_mask

from utils_io import read_json, read_features_from_file, read_bbox_from_file

def match_items(image_bin, kp_bin, des_bin, items, debug=True):
    params = read_json('parameters.json')
    ITEM_FOLDER = params['item_folder']
    MIN_MATCH_COUNT = params['min_match_count']

    item_d = {}
    recognised_items = []
    image_disp = image_bin.copy()
    mask_items = np.zeros(image_bin.shape[0:2]).astype('uint8')
    for item in items:
        prefix = ITEM_FOLDER + '/' + item + '/' + item
        filename = prefix + '_top_01_sift.npy'
        kp, des = read_features_from_file(filename)
        kp, des = unpack_keypoint(kp, des)
        des = des.astype('float32')
        good = calc_matches(des, des_bin)
        item_d[item] = {'file': filename, 'kp': kp, 'des': des, 'good': good}

        filename = prefix + '_bottom_01_sift.npy'
        kp, des = read_features_from_file(filename)
        kp, des = unpack_keypoint(kp, des)
        des = des.astype('float32')
        good = calc_matches(des, des_bin)
        if len(good) > len(item_d[item]['good']):
            item_d[item] = {'file': filename, 'kp': kp, 'des': des, 'good': good}

        if debug:
            print('Item: "%s" Good features: %d' % (item_d[item]['file'], 
                                                  len(item_d[item]['good'])))
        kp = item_d[item]['kp']
        good = item_d[item]['good']
        if len(good) > MIN_MATCH_COUNT:
            dst_pts = [ kp_bin[m.trainIdx] for m in good ]
            image_disp = cv2.drawKeypoints(image_disp,dst_pts,color=(0,255,0))
            src_pts = np.float32([ kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp_bin[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            x, y, w, h = read_bbox_from_file(item_d[item]['file'][:-9] + '_bbox.json')
            pts = np.float32([ [x,y],[x,y+h-1],[x+w-1,y+h-1],[x+w-1,y] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)
            cv2.polylines(image_disp,[np.int32(dst)],True,(0,255,0),2, cv2.CV_AA)
            cv2.fillConvexPoly(mask_items,np.int32(dst),(255,))
            recognised_items.append( (item, np.int32(dst)) )

    if debug:
        plt.imshow(image_disp), plt.axis('off');
    
    kernel = np.ones((3,3),np.uint8)
    mask_items = 255 - cv2.dilate(mask_items,kernel,iterations = 5)

    return item_d, recognised_items, mask_items

