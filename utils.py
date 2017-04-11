import cv2, glob, json, numpy as np, os.path
from matplotlib import pyplot as plt

def load_items(folder):
    items = sorted(map(os.path.basename, glob.glob(folder + '/[a-z]*')))
    return items

def imread_rgb(filename):
    image_BGR = cv2.imread(filename)
    if image_BGR is None:
        print('Error: image reading failure in file "%s"' % filename)
        return None
    else:
        image_RGB = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)
        return image_RGB
    
def imread_gray(filename):
    image = cv2.imread(filename, 0)
    if image is None:
        print('Error: image reading failure in file "%s"' % filename)
        return None
    else:
        return image

def item_mask(image_RGB, background_threshold):
    mask = 255 - cv2.inRange(image_RGB, np.array((0,0,0)), np.array((background_threshold,)*3))
    kernel = np.ones((3,3),np.uint8)
    mask = cv2.dilate(mask,kernel,iterations = 10)
    mask = cv2.erode( mask,kernel,iterations = 10)
    (cnt,_) = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(cnt,key=lambda c: cv2.contourArea(c),reverse=True)
    mask[:] = 0
    cv2.drawContours(mask,cnt,0,(255,),-1)
    mask = cv2.erode( mask,kernel,iterations = 3)
    return mask

def compute_sift(image_RGB, mask=None):
    gray_image = cv2.cvtColor(image_RGB, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT()
    kp, des = sift.detectAndCompute(gray_image, mask)
    return (kp, des)

def pack_keypoint(keypoints, descriptors):
    kpts = np.array([[kp.pt[0], kp.pt[1], kp.size,
                  kp.angle, kp.response, kp.octave,
                  kp.class_id]
                 for kp in keypoints])
    desc = np.array(descriptors)
    return kpts, desc

def write_features_to_file(filename, locs, desc):
    np.save(filename, np.hstack((locs,desc)))
    
def read_features_from_file(filename):
    """ Read feature properties and return in matrix form. """
    if os.path.getsize(filename) <= 0:
        return np.array([]), np.array([])
    f = np.load(filename)
    if f.size == 0:
        return np.array([]), np.array([])
    f = np.atleast_2d(f)
    return f[:,:7], f[:,7:] # feature locations, descriptors

def read_json(filename):
    with open(filename, 'r') as infile:
        data = json.load(infile)
    return data

def read_bbox_from_file(filename):
    bbox = read_json(filename)
    x = bbox['x']
    y = bbox['y']
    w = bbox['w']
    h = bbox['h']
    return x, y, w, h

def unpack_keypoint(kpts,desc):
    try:
        #kpts = array[:,:7]
        #desc = array[:,7:]
        keypoints = [cv2.KeyPoint(x, y, _size, _angle, _response, int(_octave), int(_class_id))
                 for x, y, _size, _angle, _response, _octave, _class_id in list(kpts)]
        return keypoints, np.array(desc)
    except(IndexError, ValueError):
        return np.array([]), np.array([])
    
def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist

from sklearn.cluster import KMeans, MiniBatchKMeans

def compute_colors(image_RGB, mask):
    image_RGBA = np.dstack((image_RGB, mask))
    pixels = image_RGBA.reshape((image_RGBA.shape[0] * image_RGBA.shape[1], 4))
    filtered_pixels = np.array(filter(lambda x:x[3]==255,pixels))
    n, _ = filtered_pixels.shape
    pixels_LAB = cv2.cvtColor(filtered_pixels[:,0:3].reshape(1,n,3),cv2.COLOR_RGB2LAB)
    pixels_LAB = pixels_LAB.reshape(n,3)
    clt = KMeans(n_clusters = 5)
    #clt = MiniBatchKMeans(n_clusters = 5)
    clt.fit(pixels_LAB)
    hist = centroid_histogram(clt)
    hist, cc = zip(*sorted(zip(hist, clt.cluster_centers_),reverse=True))
    return hist, cc

def plot_colors(hist, cc):
    cc_RGB = cv2.cvtColor(np.array(cc).astype("uint8").reshape(1,5,3), cv2.COLOR_LAB2RGB)
    cc_RGB = cc_RGB[0,:]
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype = "uint8")
    startX = 0
    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, cc_RGB):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
            color.astype('float'), -1)
        startX = endX
    return bar

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