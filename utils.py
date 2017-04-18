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
            recognised_items.append(item)
            src_pts = np.float32([ kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp_bin[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            x, y, w, h = read_bbox_from_file(item_d[item]['file'][:-9] + '_bbox.json')
            pts = np.float32([ [x,y],[x,y+h-1],[x+w-1,y+h-1],[x+w-1,y] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)
            cv2.polylines(image_disp,[np.int32(dst)],True,(0,255,0),2, cv2.CV_AA)
            cv2.fillConvexPoly(mask_items,np.int32(dst),(255,))

    if debug:
        plt.imshow(image_disp), plt.axis('off');
    return item_d, recognised_items, mask_items

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

from sklearn.cluster import KMeans, MiniBatchKMeans

def cluster_colors(image_bin, mask_bin, items, debug=True):

    n_cc = 20
    image_RGBA = np.dstack((image_bin, mask_bin))
    pixels = image_RGBA.reshape((image_RGBA.shape[0] * image_RGBA.shape[1], 4))
    filtered_pixels = np.array(filter(lambda x:x[3]==255,pixels))
    n, _ = filtered_pixels.shape
    pixels_LAB = cv2.cvtColor(filtered_pixels[:,0:3].reshape(1,n,3),cv2.COLOR_RGB2LAB)
    pixels_LAB = pixels_LAB.reshape(n,3)
    #clt = MiniBatchKMeans(n_clusters = n_cc)
    clt = KMeans(n_clusters = n_cc)
    clt.fit(pixels_LAB)

    image = cv2.cvtColor(image_bin, cv2.COLOR_RGB2LAB)
    (h_bin, w_bin) = image.shape[:2]
    pixels = image.reshape((image.shape[0] * image.shape[1], 3))
    labels = clt.predict(pixels)
    quant = clt.cluster_centers_.astype("uint8")[labels]
    quant = quant.reshape((h_bin, w_bin, 3))
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2RGB)
    if debug:
        plt.imshow(cv2.bitwise_and(quant,quant,mask=mask_bin)); plt.title('%d colors' % n_cc); plt.axis('off'); plt.show();
    bin_cc = clt.cluster_centers_

    bin_hist, _ = np.histogram(clt.predict(pixels_LAB),bins=range(n_cc+1))
    if debug:
        plt.bar(range(n_cc), bin_hist); plt.show();
    
    sort_index = np.argsort(bin_hist)[::-1]

    params = read_json('parameters.json')
    ITEM_FOLDER = params['item_folder']
    positions = []
    weights = []
    while len(sort_index)>0:
        obj_label = sort_index[0]
        d_other = [np.linalg.norm(bin_cc[obj_label,1:]-bin_cc[other,1:]) for other in sort_index]
        obj_labels = [sort_index[idx] for idx,val in enumerate(d_other) if val<20]
        obj_hist = np.array([bin_hist[obj_l] for obj_l in obj_labels],dtype='float32')
        obj_hist = obj_hist / np.sum(obj_hist)
        sort_index = np.array([x for x in sort_index if x not in obj_labels])
        mask = np.zeros((h_bin, w_bin)).astype('uint8')
        for val_label in obj_labels:
            mask = cv2.bitwise_or( mask, ((labels==val_label).astype('uint8') * 255).reshape((h_bin, w_bin)) )
        mask = cv2.bitwise_and( mask, mask_bin)
        kernel = np.ones((3,3),np.uint8)
        mask = cv2.erode(mask,kernel,iterations = 3)
        mask = cv2.dilate(mask,kernel,iterations = 3)
        #cnt, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnt, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        cnt = sorted(cnt, key=lambda x:cv2.contourArea(x), reverse=True)
        best_cnt = [c for c in cnt if cv2.contourArea(c)>500]
        positions.append(best_cnt)

        best_item = []
        views = ['top_01','top-side_01','top-side_02','bottom_01','bottom-side_01','bottom-side_02']
        for item in items:
            for view in views:
                try:
                    filename = ITEM_FOLDER + '/' + item + '/' + item + '_' + view + '_dc.json'
                    dc = read_json(filename)
                    hist = dc['hist']
                    obj_cc = dc['cluster_centers']
                    sum_h = 0
                    for i in range(5):
                        d_bin_obj = [np.linalg.norm(obj_cc[i]-bin_cc[obj_l,:]) for obj_l in obj_labels]
                        index_min = np.argmin(d_bin_obj)
                        if d_bin_obj[index_min] < 25:
                            sum_h += hist[i] * obj_hist[index_min]
                            # hist[i] is the number of pixels in the image -> count only in rectangle?
                    #if sum_h > 0.05:
                    if sum_h > 0.1:
                        best_item.append((sum_h,item,view))
                except IOError:
                    pass
        best_item_one = []
        for it in items:
            try:
                w = max([bi[0] for bi in best_item if bi[1]==it])
                best_item_one.append((w,it))
            except ValueError:
                pass
        weights.append(best_item_one)
    return positions, weights