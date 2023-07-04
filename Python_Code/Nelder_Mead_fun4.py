# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 16:23:18 2023

@author: pli4
"""
from dm.okotech.dm import OkoDM

from camera.ueye_camera import uEyeCamera
from pyueye import ueye
#import glob
import cv2
from zernike import RZern
from scipy.ndimage.measurements import center_of_mass
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import scipy.optimize

walk_num=1
step=1
Camera_Index=1


#2.3 Codes 2 W A VEFRONT SENSORLESS CORRECTION PART I
#print("initial position:",x)
# define a function to calculate the maximum value of intensity from figure
# diiferent function ofr different algorithm

def grabframes(nframes, cameraIndex=0):
    #with uEyeCamera(device_id=1) as cam:
    with uEyeCamera(device_id=1) as cam:
        cam.set_colormode(ueye.IS_CM_MONO8)  # IS_CM_MONO8)
        w = 1280
        h = 1024
        # cam.set_aoi(0,0, w, h)

        cam.alloc(buffer_count=10)
        cam.set_exposure(1.335)
        #cam.set_exposure(0.5)
        cam.capture_video(True)

        imgs = np.zeros((nframes, h, w), dtype=np.uint8)
        acquired = 0
        # For some reason, the IDS cameras seem to be overexposed on the first frames (ignoring exposure time?).
        # So best to discard some frames and then use the last one
        while acquired < nframes:
            frame = cam.grab_frame()
            if frame is not None:
                imgs[acquired] = frame
                acquired += 1
        cam.stop_video()
    return imgs

#def max_intensity(img):
#    intensity=np.amax(img[-1])
#    return intensity

def max_variance(act):
     with OkoDM(dmtype=1) as dm:
         dm.setActuators(act)
         time.sleep(0.1)
         img= grabframes(5, 1)
         img_cal=img[-1]
         var=np.var(img_cal)
#    mean=np.mean(img_cal)
#    N=np.size(img_cal)
#    var=0
#    img_cal.reshape((-1,1))
#    for i in range(N):
#        var=var+(img_cal[i]-mean)**2
#    var=var/N
     return var

#img = grabframes(1, 1)
#         cropped_img = img[0,320:960,256:768]
#         sum_xy = 0
#         weighted_sum_k=0
#         weighted_sum_h=0
#         bright_sum=0
#         
#         for k in range(np.shape(cropped_img)[0]):
#                for h in range(np.shape(cropped_img)[1]):
#                    weighted_sum_k=weighted_sum_k+cropped_img[k,h]*k
#                    weighted_sum_h=weighted_sum_h+cropped_img[k,h]*h
#                    bright_sum=bright_sum+cropped_img[k,h]
#  
#         k_c = weighted_sum_k / bright_sum_k
#         h_c = weighted_sum_h / bright_sum_h
#    
#         weighted_sum_var=0
#         for k in range(np.shape(cropped_img)[0]):
#                for h in range(np.shape(cropped_img)[1]):
#                    d2=(k-k_c)**2+(h-h_c)**2
#                    weighted_sum_var = weighted_sum_var + d2*cropped_img[k,h]
#            
#            
#         var_d =int(weighted_sum_var/bright_sum)
#         
#         return var_d

	# Nelder-Mead Optimization	
def nelder_mead (act): 
#    history = []
    opts = {'maxiter': 10, 'adaptive': True}
    result = scipy.optimize.minimize(max_variance, act, method='Nelder-Mead', tol=1, options = opts)
    print(result)
    print('Status: %s' %result['message'])
    print('Total Evaluations: %d' %result['nfev'])

    solution=result['x']
    evaluation = result['fun']
    print('Solution: f(%s) = %.5f' %(solution, evaluation))
    #plot history
#    plt.plot(solution,eva)
#    plt.show()

# loop of random walk
if __name__ == "__main__":
    from dm.okotech.dm import OkoDM
    with OkoDM(dmtype=1) as dm:
        #act=np.zeros([len(dm)])
        act=np.zeros((19,1))

        nelder_mead(act)   
        #%% screenshoot
if __name__ == "__main__":
    from dm.okotech.dm import OkoDM
    with OkoDM(dmtype=1) as dm:
        V=np.zeros((19,1))
        V[-3]= 0.00025
        #V=[0, 0,0, 0, 0,0,0,0,0,0,0, 0, 0,0, 0, 0, 0.00025,0,0]
        print(np.shape(V))
        dm.setActuators(V)
        img=grabframes(5, 1)
        img_n = img[-1,384:640,480:800]

        plt.imshow(img_n)
        plt.colorbar(label='Intensity', orientation='vertical')
        plt.figure()
        
#        while False:
#            act=np.zeros([len(dm)])
#            dm.setActuators(act)
#            time.sleep(0.1)
    