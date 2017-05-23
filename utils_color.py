import cv2, glob, json, numpy as np, os.path
from matplotlib import pyplot as plt

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
    #print((hist,clt.cluster_centers_))
    hist, dummy, cc = zip(*sorted(zip(hist, range(len(hist)), clt.cluster_centers_),reverse=True))
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

from utils_io import read_json

def cluster_colors(image_bin, mask_bin, items, n_cc=20, debug=False):

    image_RGBA = np.dstack((image_bin, mask_bin))
    pixels = image_RGBA.reshape((image_RGBA.shape[0] * image_RGBA.shape[1], 4))
    filtered_pixels = np.array(filter(lambda x:x[3]==255,pixels))
    n, _ = filtered_pixels.shape
    pixels_LAB = cv2.cvtColor(filtered_pixels[:,0:3].reshape(1,n,3),cv2.COLOR_RGB2LAB)
    pixels_LAB = pixels_LAB.reshape(n,3)
    clt = MiniBatchKMeans(n_clusters = n_cc)
    #clt = KMeans(n_clusters = n_cc)
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
        # distance in the ab components of the Lab color space
        d_other = [np.linalg.norm(bin_cc[obj_label,1:]-bin_cc[other,1:]) for other in sort_index]
        # group "similar" colors with distance less than a threshold (20?)
        obj_labels = [sort_index[idx] for idx,val in enumerate(d_other) if val<10]
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
        # select only contours with a minimum area (500?)
        best_cnt = [c for c in cnt if cv2.contourArea(c)>200]
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
                        # distance in the Lab color space
                        d_bin_obj = [np.linalg.norm(obj_cc[i]-bin_cc[obj_l,:]) for obj_l in obj_labels]
                        index_min = np.argmin(d_bin_obj)
                        # distance threshold between object and bin colors (25?)
                        if d_bin_obj[index_min] < 25:
                            sum_h += hist[i] * obj_hist[index_min]
                            # hist[i] is the number of pixels in the image -> count only in rectangle?
                    #if sum_h > 0.05:
                    # weight threshold (0.05?)
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

def calc_EMD2(h_obj,cc_obj,h_ref,cc_ref):
    sig_obj = np.concatenate((np.reshape(np.array(h_obj),(len(h_obj),1)),
                              np.array(cc_obj)[:,0:3]),1)
    a64 = cv2.cv.fromarray(sig_obj)
    a32 = cv2.cv.CreateMat(a64.rows, a64.cols, cv2.cv.CV_32FC1)
    cv2.cv.Convert(a64, a32)
    sig_ref = np.concatenate((np.reshape(np.array(h_ref),(len(h_ref),1)),
                              np.array(cc_ref)[:,0:3]),1)
    b64 = cv2.cv.fromarray(sig_ref)
    b32 = cv2.cv.CreateMat(b64.rows, b64.cols, cv2.cv.CV_32FC1)
    cv2.cv.Convert(b64, b32)
    return cv2.cv.CalcEMD2(a32,b32,cv2.cv.CV_DIST_L2)

from utils_contour import contourClustering

def find_items_by_color(image_bin, image_mask, items, n_cc=20):
    positions, weights = cluster_colors(image_bin, image_mask, items, n_cc)
    
    pos_ok = [(p,w[0][1]) for p,w in zip(positions, weights) if len(w)==1]
    pos_unkw = [(p,w) for p,w in zip(positions, weights) if len(w)>1]
    
    contours = {}
    for item in items:
        contours[item] = []
        it_pos = [p for p,it in pos_ok if it==item]
        for cnt in it_pos:
            contours[item] += cnt

    clusters = {}
    threshold = 30
    for item in items:
        clusters[item] = contourClustering(contours[item], threshold)
        
    recognised_items = [ item for item in clusters.keys() if clusters[item] ]

    bboxes = []
    for item in recognised_items:
        area = 0
        bb = None
        for cl in clusters[item]:
            cc = [c for idx, c in enumerate(contours[item]) if idx in cl]
            # TODO: oriented rectangle
            x,y,w,h = cv2.boundingRect(np.vstack(tuple(cc)))
            if area < w*h:
                area = w*h
                bb = (x,y,w,h)
        bboxes.append(bb)
        
    return recognised_items, bboxes
