"""
Example for PC in right corner

Setup right corner:
    Camera 1: Imaging camera
    Camera 2: SH sensor
"""
from dm.okotech.dm import OkoDM

from camera.ueye_camera import uEyeCamera
from pyueye import ueye

import numpy as np
import matplotlib.pyplot as plt
import time
import random


SH_Sensor_Index = 2
Camera_Index = 1


def grabframes(nframes, cameraIndex=0):
    with uEyeCamera(device_id=1) as cam:
        cam.set_colormode(ueye.IS_CM_MONO8)  # IS_CM_MONO8)
        w = 1280
        h = 1024
        # cam.set_aoi(0,0, w, h)

        cam.alloc(buffer_count=10)
        cam.set_exposure(0.067)
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

if __name__ == "__main__":
    #from dm.okotech.dm import OkoDM

    print("started")
    # Use "with" blocks so the hardware doesn't get locked when you press ctrl-c
    with OkoDM(dmtype=1) as dm:
        print(f"Deformable mirror with {len(dm)} actuators")
        # set actuators to 0
        act = np.zeros([len(dm)])
        dm.setActuators(act)

        # Use True/False to turn on code blocks

        # example loop for moving the actutors up and down
        # hint: Check what all actuators do indivudually -> not all mirrors are equal
        # hint: feel free to look up you model online for additional info.

        #move the mirrors individually and ignoring
        num_actuators = len(dm)
        

        # Code for 1-D random walk
        
 
        # statically defining the starting position
        val_act = np.zeros(num_actuators)
        val_act[0] = 0
        val_act[1] = -0.5
        val_act[2] = -0.5
        val_act[3] = -0.5
        val_act[4] = -0.5
        val_act[5] = 1 
        val_act[6] = -0.5
        val_act[7] = -0.5 
        val_act[8] = -0.5
        val_act[9] = 1
        val_act[10] =1
        val_act[11] = 1
        val_act[12] = -0.5
        val_act[13] = -0.5
        val_act[14] = -0.5
        val_act[15] = 1
        val_act[16] = 1
        val_act[17] = 0
        val_act[18] = 0
        
        val_act = np.zeros(num_actuators)
#       val_act[17] = 0.5
        for f in range(len(dm)-2):
            val_act[f] = val_act[f] + random.uniform(-0.5,0.5)
        
        # init RW parameters
        voltage = val_act
        walk_iter = 100
        n = 1
        prev_act_amp = 0
        iteration_progress = 2*10**5
        best_iteration = float('inf')
        
        
        # Probability to move up or down
        prob = [0.575, 0.575]         
        
        # Perform the random walk for the specified number of iterations
        for _ in range(walk_iter):
            for h in range(num_actuators-2):
                down = np.random.random() < prob[0] and voltage[h] > -1
                up = np.random.random() > (1-prob[1]) and voltage[h] < 1
                voltage[h] -= down / (5*n) - up / (5*n)

                
            #print(voltage)
 
        # for loop for making the walking process
#        for idownp, iupp in zip(downp, upp):
#            down = idownp and voltage[-1] > -10
#            up = iupp and voltage[-1] < 10
#            voltage.append(voltage[-1] - down/20 + up/20)
#            
            #print(voltage[-1])
            
            #voltage_val = float(voltage)*0.8/10.0
            
            s_time = 0.01  # sleep time (small amount of time between steps)
            w_time = 0.01  # wait time around focus
            steps = 10
            pre_act_amp = 0
           
            # increase actuator voltage gradually, then reverse, hold at 0
            for i in range(steps):
                #current = np.ones(num_actuators)#np.zeros(num_actuators) for resetting the selected actuators
                #current[j] = 1, only needed for seperate control
                act_amp = steps * voltage * (i + 1) + prev_act_amp / ((0.3*i)**3 + 1) #standard coeff
                dm.setActuators(act_amp)
                time.sleep(s_time)  # in seconds
                # print(act_amp[0])
            #for i in range(steps):
            #act_amp = voltage_val / steps * current * (steps - i)
            #dm.setActuators(act_amp)
            #time.sleep(s_time)  # in seconds
            # print(act_amp[0])

                #dm.setActuators(np.zeros(len(dm)))
                #time.sleep(w_time)

            # decrease actuator voltage gradually, then reverse, hold at 0
            #for i in range(steps):
                #act_amp = -voltage_val / steps * current * (i + 1)
                #dm.setActuators(act_amp)
                #time.sleep(s_time)  # in seconds
                # print(act_amp[0])
            #for i in range(steps):
                #act_amp = -voltage_val / steps * current * (steps - i)
                #dm.setActuators(act_amp)
                #time.sleep(s_time)  # in seconds
                # print(act_amp[0])

            #dm.setActuators(np.zeros(len(dm)))
            prev_act_amp = act_amp
            time.sleep(w_time)
                
                
            img = grabframes(1, 1)
            cropped_img = img[0,384:640,480:800]
            sum_img = 0
            
            for k in range(np.shape(cropped_img)[0]):
                for j in range(np.shape(cropped_img)[1]):
                    if cropped_img[k,j] < 175:
                        cropped_img[k,j] = 0
                    sum_img = sum_img + cropped_img[k,h]
                    
                        
            if best_iteration > sum_img:
                best_iteration = sum_img
                voltage_best = voltage
                iteration_progress = np.vstack((iteration_progress, best_iteration))
                
                if iteration_progress[0] < best_iteration:
                    iteration_progress[0] = best_iteration  
            
                n = n+1
            
        plt.plot(iteration_progress)
        plt.ylabel('image value')
        plt.xlabel('iterations')
        plt.show()
        
        print(voltage_best)
                
        dm.setActuators(voltage_best)
        
        plt.figure()
        img = grabframes(1, 1)
        cropped_img = img[0,384:640,480:800]
        plt.imshow(cropped_img, cmap='gist_gray', interpolation='nearest')
        plt.colorbar()
        
        #dm.setActuators(np.zeros(len(dm)))
        


print('finished operation')
