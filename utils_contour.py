import cv2, math, numpy as np

def contourDist(ca,cb):
    dmin = float('inf')
    for pa in ca:
        for pb in cb:
            d = np.sum((pa - pb) * (pa - pb))
            if d < dmin:
                dmin = d
    return math.sqrt(dmin)

def contourInCluster(c, clist):
    found = [idx for idx, cl in enumerate(clist) if c in cl]
    if found:
        return found[0]
    else:
        return None

def contourClustering(c, threshold=30):
    cls = []
    n = len(c)
    if n==1:
        cls.append([0,])
    pairs = []
    for i in range(n):
        for j in range(i+1,n):
            pairs.append( (contourDist(c[i],c[j]), i, j) )
    pairs = sorted(pairs)
    for d, a, b in pairs:
        ca = contourInCluster(a, cls)
        cb = contourInCluster(b, cls)
        if d < threshold:
            if ca is None:
                if cb is None:
                    cls.append( [a, b] )
                else:
                    cls[cb].append(a)
            else:
                if cb is None:
                    cls[ca].append(b)
                else:
                    if ca != cb:
                        cls[ca] += cls[cb]
                        del cls[cb]
        else:
            if ca is None:
                cls.append([a])
            if cb is None:
                cls.append([b])
    return cls
