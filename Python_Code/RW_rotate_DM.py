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
        cam.set_exposure(18.5)
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
        
        # Probability to move up or down
        prob = [0.25, 0.75] 
 
        # statically defining the starting position
        initial_voltage = random.randint(-10,10) 
        voltage = [initial_voltage]
 
        # creating the random points
        rr = np.random.random(walk_iter)
        downp = rr < prob[0]
        upp = rr > prob[1]
        j = 0
        prev_act_amp = 0
        
        evolution = np.array(walk_iter)
 
        # for loop for making the walking process
        for idownp, iupp in zip(downp, upp):
            down = idownp and voltage[-1] > -10
            up = iupp and voltage[-1] < 10
            voltage.append(voltage[-1] - down + up)
            a = True
            j = j+1
            voltage_val = float(voltage[j])*0.8/10.0
            
            while a == True:
                s_time = 0.5  # sleep time (small amount of time between steps)
                w_time = 2  # wait time around focus
                steps = 10
                pre_act_amp = 0
                # increase actuator voltage gradually, then reverse, hold at 0
                for i in range(steps):
                    current = np.ones(num_actuators)#np.zeros(num_actuators) for resetting the selected actuators
                    #current[j] = 1, only needed for seperate control
                    act_amp = voltage_val / steps * current * (i + 1) + prev_act_amp / ((0.3*i)**3 + 1) #standard coeff
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
                a = False

            img = grabframes(1, 1)
            cropped_img = img[0,320:960,256:768]
            for k in range(np.shape(cropped_img)[0]):
                for h in range(np.shape(cropped_img)[1]):
                    sum_img = cropped_img[k,h]
                    
            evolution(iupp+idownp) = sum_img
            
        plt.plot(evolution)
        plt.ylabel('image value')
        plt.xlabel('iterations')
        plt.show()
                
        dm.setActuators(np.zeros(len(dm)))
        


print('finish operation')
