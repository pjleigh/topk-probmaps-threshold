import torch
import matplotlib.pyplot as plt
import matplotlib.path as path
import numpy as np
import glob
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes, binary_dilation, binary_erosion
from scipy.spatial import ConvexHull
from PIL import Image

# based on modify_test.ipynb amd prediction_visualize_modify.ipynb

# variables
probmapDir ="./probmaps/" # string directory of where input segmentation probability map npy files are
k = 3 # int value to take top k probability map values for thresholding
threshold = 0.3 # float threshold value 0-1 to compare mean of top k for binarization
outDir = "./newmasks/" # string output directory to save new masks

'''
takes segmentation probability maps and binarizes to a segmentation map
based on the mean of the top k results and threshold value
'''
def topkmask(probmaps, thre, k):
    topkmaps, _ = torch.topk(torch.Tensor(probmaps[:,:,1,:]), k, dim=2)
    topkmaps = torch.mean(topkmaps, dim=2)
    return topkmaps > thre

'''
takes single mask array of floats (0-1) [height, width]
puts out single binary mask (0 or 1) [height, width]
depending on threshold value
'''
def thresholdmasks(mask, threshold):
    n, m = mask.shape
    newmask = np.zeros((n, m))
    
    for i in range(n):
        for j in range(m):
            if mask[i, j] >= threshold: # set mask to binary 0 or 1 depending on threshold value
                newmask[i, j] = 1
            else:
                newmask[i, j] = 0
                
    return newmask

'''
modifies a mask to fill holes, convex hull, and remove small portions
'''
def modify(mask):
    mask = binary_dilation(mask, iterations=2)
    mask = binary_fill_holes(mask)
    mask = binary_erosion(mask, iterations=1)

    label_img = label(mask)
    regions = regionprops(label_img)
    for idx in range(len(regions)):
        if regions[idx].area <= 500:
            mask[label_img==idx+1] = 0
    
    label_img = label(mask)
    regions = regionprops(label_img)

    for idx in range(len(regions)):
        hull = ConvexHull(regions[idx].coords)  
        poly_points = np.array([regions[idx].coords[hull.vertices,0], regions[idx].coords[hull.vertices,1]])
        poly_points = np.transpose(poly_points)
        polygon = path.Path(poly_points)
        
        for i in range(min(poly_points[:, 0]), max(poly_points[:, 0])):
            for j in range(min(poly_points[:, 1]), max(poly_points[:, 1])):
                if polygon.contains_point((i, j)):
                    mask[i, j] = 1
        
    mask = binary_erosion(mask, iterations=1)
            
    return np.uint8(mask * 255)

def main():
    probmaps = [f for f in glob.glob(probmapDir+"**.npy", recursive=False)]

    for i in probmaps:
        probmap = np.load(i)
        topkmask = topkmask(probmap, threshold, k)
        modifiedmask = modify(topkmask)
        maskimage = Image.fromarray(modifiedmask)
        maskimage.save(outDir+i.split("/")[1].split(".")[0]+".png")

    print("done.")

if __name__ == "__main__":
    main()
