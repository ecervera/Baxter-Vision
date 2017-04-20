import cv2, glob, json, numpy as np, os.path

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

def imwrite(filename, image):
    cv2.imwrite(filename, image)
    
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

def write_json(filename, data):
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)
        
def read_bbox_from_file(filename):
    bbox = read_json(filename)
    x = bbox['x']
    y = bbox['y']
    w = bbox['w']
    h = bbox['h']
    return x, y, w, h

