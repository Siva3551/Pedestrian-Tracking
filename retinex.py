

import numpy as np
import cv2

from PIL import Image

def singleScaleRetinex(img,variance):
    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), variance))
    return retinex

def multiScaleRetinex(img, variance_list):
    retinex = np.zeros_like(img)
    for variance in variance_list:
        retinex += singleScaleRetinex(img, variance)
    retinex = retinex / len(variance_list)
    return retinex

   

def MSR(img, variance_list):
    img = np.float64(img) + 1.0
    img_retinex = multiScaleRetinex(img, variance_list)

    for i in range(img_retinex.shape[2]):
        unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
        for u, c in zip(unique, count):
            if u == 0:
                zero_count = c
                break            
        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0
        for u, c in zip(unique, count):
            if u < 0 and c < zero_count * 0.1:
                low_val = u / 100.0
            if u > 0 and c < zero_count * 0.1:
                high_val = u / 100.0
                break            
        img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)
        
        img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                               (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                               * 255
    img_retinex = np.uint8(img_retinex)        
    return img_retinex



def SSR_start(img, variance):
    img = np.float64(img) + 1.0
    img_retinex = singleScaleRetinex(img, variance)
    for i in range(img_retinex.shape[2]):
        unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
        for u, c in zip(unique, count):
            if u == 0:
                zero_count = c
                break            
        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0
        for u, c in zip(unique, count):
            if u < 0 and c < zero_count * 0.1:
                low_val = u / 100.0
            if u > 0 and c < zero_count * 0.1:
                high_val = u / 100.0
                break            
        img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)
        
        img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                               (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                               * 255
    img_retinex = np.uint8(img_retinex)        
    return img_retinex


#variance_list=[15, 80, 250]
# variance=300
    
# img = cv2.imread('siva2.jpg')
# img_msr=MSR(img,variance_list)
# img_ssr=SSR(img, variance)

# cv2.imshow('Original', img)
# cv2.imshow('MSR', img_msr)
# cv2.imshow('SSR', img_ssr)
# cv2.imwrite('SSR.jpg', img_ssr)
# cv2.imwrite('MSR.jpg',img_msr)


# cv2.waitKey(0)
# cv2.destroyAllWindows()



def SSR(img, variance, box):

    # crop = img[int((box[1]-box[3]/2)):int((box[1]+1.5*box[3])),int((box[0]-1.5*box[2])):int((box[0]+2.5*box[2]))]

    crop = img[int((box[1]-box[3]/4)):int((box[1]+1.25*box[3])),int((box[0]-0.25*box[2])):int((box[0]+1.25*box[2]))]


    # crop = img[int(box[1]):int((box[1]+box[3])), int(box[0]):int((box[0]+box[2]))]

    # imge=np.zeros([img.shape[0],img.shape[1],img.shape[2]])
    
    # imge[int((box[1]-box[3]/2)):int((box[1]+1.5*box[3])),int((box[0]-1.5*box[2])):int((box[0]+2.5*box[2]))] = np.float64(crop) + 1.0
    # imge[int((box[1]-box[3]/2)):int((box[1]+1.5*box[3])),int((box[0]-1.5*box[2])):int((box[0]+2.5*box[2]))] = np.float64(crop) + 1.0
    imge = np.float64(crop) + 1.0
    img_retinex = singleScaleRetinex(imge, variance)

    for i in range(img_retinex.shape[2]):
        unique, count = np.unique(np.int32(img_retinex[:, :, i]*100), return_counts=True)
        for u, c in zip(unique, count):
            if u == 0:
                zero_count = c
                break            
        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0
        for u, c in zip(unique, count):
            if u < 0 and c < zero_count * 0.01:
                low_val = u / 100.0
            if u > 0 and c < zero_count * 0.01:
                high_val = u / 100.0
                break            
        img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)
        
        img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                               (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                               * 255
    img_retinex = np.uint8(img_retinex)  



    # img_ret = np.array(img_retinex) 
    # print(img_ret)

    ori_img = np.array(img) 
    # print(ori_img.shape)

    ori_img[int((box[1]-box[3]/4)):int((box[1]+1.25*box[3])),int((box[0]-0.25*box[2])):int((box[0]+1.25*box[2]))] = img_retinex

    # ori_img[int((box[1]-box[3]/2)):int((box[1]+1.5*box[3])),int((box[0]-1.5*box[2])):int((box[0]+2.5*box[2]))] = img_retinex

   




    # print(ori_img[int((box[1]-box[3]/2)):int((box[1]+1.5*box[3])),int((box[0]-box[2]/2)):int((box[0]+1.5*box[2]))])
    # print(ori_img)
    # img_1 = Image.fromarray(ori_img) 
    # m=cv2.imwrite('img',ori_img)
    # print(img.shape) 
    # print(ori_img.shape)  
    return ori_img






def template(img, variance, box):


    crop = img[int(box[1]):int((box[1]+box[3])),int(box[0]):int((box[0]+box[2]))]

    imge = np.float64(crop) + 1.0
    img_retinex = singleScaleRetinex(imge, variance)

    for i in range(img_retinex.shape[2]):
        unique, count = np.unique(np.int32(img_retinex[:, :, i]*100), return_counts=True)
        for u, c in zip(unique, count):
            if u == 0:
                zero_count = c
                break            
        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0
        for u, c in zip(unique, count):
            if u < 0 and c < zero_count * 0.01:
                low_val = u / 100.0
            if u > 0 and c < zero_count * 0.01:
                high_val = u / 100.0
                break            
        img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)
        
        img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                               (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                               * 255
    img_retinex = np.uint8(img_retinex)  

    return img_retinex