'''
    Author: Ali M Bukar
    
'''
import sys
import cv2
import numpy as np 
from numpy import linalg as LA

import os

def procrustes(X, Y, scaling=False, reflection='best'):
   
    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':
        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0
        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:
        # optimum scaling of Y
        b = traceTA * normX / normY
        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2
        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX
    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX
    
    Zr = np.matmul(Y, T)
    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)
    #transformation values 
    tform = {'rotation':T, 'scale':b, 'translation':c}

    return d, Z, tform, Zr

def creat_transformation_matrix(Tform):
    T = Tform["translation"]
    R = Tform["rotation"]
    S = Tform["scale"]
    
    SR = S*R
    rot_matrix = np.array([[SR[0,0], SR[1,0], 0], [SR[0,1], SR[1,1], 0], [T[0], T[1], 1]])
    return rot_matrix

def creat_transformation_matrix_cv2(Tform):
    T = Tform["translation"]
    R = Tform["rotation"]
    S = Tform["scale"]
    
    SR = S*R
    print(SR)
    rot_matrix = np.array([[SR[0,0], SR[0,1], T[0]], [SR[1,0], SR[1,1],T[1]]])
    print(rot_matrix)
    return rot_matrix

def get_transmat(Tform):
    R = np.eye(3)
    R[0:2,0:2] = Tform['rotation']
    S = np.eye(3) * Tform['scale'] 
    S[2,2] = 1
    t = np.eye(3)
    t[0:2,2] = Tform['translation']
    M = np.dot(np.dot(R,S),t.T).T
    return M,M[0:2,:]

def rotate_coords(coords, Tform):
    T = Tform["translation"]
    R = Tform["rotation"]
    S = Tform["scale"]
    
    a = S*coords
    a = np.matmul(a, R)
    a = a + T
    return a

def normalise_by_norm(data):
    l2_norm = LA.norm(data)
    norm_data = data/l2_norm

    return norm_data

def preprocess_img(img):
    # resized = cv2.resize(img, (128, 128))
    # # center crop image
    # a=int((128-112)/2) # x start
    # b=int((128-112)/2+112) # x end
    # c=int((128-112)/2) # y start
    # d=int((128-112)/2+112) # y end
    # ccropped = resized[a:b, c:d] # center crop the image
    ccropped = cv2.resize(img, (112, 112))
   

    return ccropped

def compare_features(ff, ff1, names, threshd=0.394):
    
    
    data = distance.cdist(ff1, ff, 'cosine')
    res = np.argmin(data)
    score = data[res]
    #print('score', score)
    if(score > threshd):
        conf_val = 0
        name = 'Unknown_Id'
    else:
        name = names[res]
        
    return name

def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()

def crop_eyes(image, coords5):

    #align 5 keypoints to template keypoints
    templ = np.array([[81,  94],[167, 93], [124, 158], [88, 203], [162, 201]])
    _,_,Tform,_ = procrustes(templ, coords5, scaling=True, reflection='best')
    _,transmat = get_transmat(Tform)

    #rotate image and its coordinate
    outImage = cv2.warpAffine(image, transmat, (250, 250))
    outCoord5 = rotate_coords(coords5, Tform)

    #crop eye region based on the rotated image
    eye_region = crop_eye_region(outCoord5, outImage)

    return eye_region

def crop_eye_region(imshape, im):

    inter_occular_d = int(imshape[1][0] - imshape[0][0])
    dx = int(inter_occular_d/3)
    dy = int(inter_occular_d/2)
    region_width = inter_occular_d + 2*dx
    x = int(imshape[0][0]) - dx
    y = int(imshape[0][1]) - dy

    crop_im = im[y:y+region_width, x:x+region_width]
    return crop_im

def softmax2(x):
    x = x- np.max(x)
    return np.exp(x)/sum(np.exp(x))

def crop_face(image, coords5):

    #align 5 keypoints to template keypoints
    templ = np.array([[81,  94],[167, 93], [124, 158], [88, 203], [162, 201]])
    _,_,Tform,_ = procrustes(templ, coords5, scaling=True, reflection='best')
    _,transmat = get_transmat(Tform)

    #rotate image and its coordinate
    outImage = cv2.warpAffine(image, transmat, (250, 250))
    outCoord5 = rotate_coords(coords5, Tform)

    return outImage

def reshape_landmarks(shape):
    default_landmarks = np.array([68, 69, 30, 48, 54])
    
    coords = np.zeros((70, 2))
    coords[0:68,:] = shape
    
    #coordinates 69 and 70
    c69 = coords[36:41,:]
    c69 = np.mean(c69, axis =0)
    coords[68]=c69

    c70 = coords[42:47,:]
    c70 = np.mean(c70, axis =0)
    coords[69]=c70
    
    subset_coords = coords[default_landmarks,:]

    return subset_coords

def crop_face2(image, coords):
    # print('here')
    #given 68 keypoint we will extract five (2 eye centers, nose tip, and two corners of the mouth)
    coords5 = reshape_landmarks(coords)
    #for i, (x, y) in enumerate(coords5):
	    # Draw the circle to mark the keypoint 
            #cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
    #cv2_imshow(image)
    #align 5 keypoints to template keypoints
    templ = np.array([[81,  94],[167, 93], [124, 158], [88, 203], [162, 201]])
    _,_,Tform,_ = procrustes(templ, coords5, scaling=True, reflection='best')
    _,transmat = get_transmat(Tform)

    #rotate image and its coordinate
    #print(image.shape)
    outImage = cv2.warpAffine(image, transmat, (250, 250),borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    outCoord5 = rotate_coords(coords5, Tform)

    return outImage

def softmax2(x):
    x = x- np.max(x)
    return np.exp(x)/sum(np.exp(x))


        



