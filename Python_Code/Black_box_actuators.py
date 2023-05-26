"""
Example for PC in right corner

Setup right corner:
    Camera 1: Imaging camera
    Camera 2: SH sensor
"""

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
    from dm.okotech.dm import OkoDM

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

        #move the mirrors individually
        num_actuators = len(dm)

        val_act = np.zeros(num_actuators)
        val_act[0] = 1
        val_act[1] = 0
        val_act[2] = 0
        val_act[3] = 0
        val_act[4] = 1
        val_act[5] = 0 
        val_act[6] = 0.5
        val_act[7] = 0  
        val_act[8] = 1
        val_act[9] = 0
        val_act[10] = 0
        val_act[11] = 0
        val_act[12] = 0
        val_act[13] = 0
        val_act[14] = 0
        val_act[15] = 0
        val_act[16] = 1
        val_act[17] = 0
        val_act[18] = 0

        a = True
 
        while a == True:
            s_time = 0.01  # sleep time (small amount of time between steps)
            w_time = 5  # wait time around focus
            steps = 10
            # increase actuator voltage gradually, then reverse, hold at 0
            for i in range(steps):
                current = val_act#np.zeros(num_actuators) for resetting the selected actuators
                #current[j] = 1, only needed for seperate control
                act_amp = 0.8 / steps * current * (i + 1) #standard coeff
                dm.setActuators(act_amp)
                time.sleep(s_time)  # in seconds
                # print(act_amp[0])
            time.sleep(w_time)
            for i in range(steps):
                act_amp = 0.8 / steps * current * (steps - i)
                dm.setActuators(act_amp)
                time.sleep(s_time)  # in seconds
                # print(act_amp[0])

            dm.setActuators(np.zeros(len(dm)))
            #time.sleep(w_time)

            # decrease actuator voltage gradually, then reverse, hold at 0
            for i in range(steps):
                act_amp = -0.8 / steps * current * (i + 1)
                dm.setActuators(act_amp)
                time.sleep(s_time)  # in seconds
                # print(act_amp[0])
            time.sleep(w_time)
            for i in range(steps):
                act_amp = -0.8 / steps * current * (steps - i)
                dm.setActuators(act_amp)
                time.sleep(s_time)  # in seconds
                # print(act_amp[0])

            dm.setActuators(np.zeros(len(dm)))
            #time.sleep(w_time)
            a = False

        if False:
            # send signal to DM
            dm.setActuators(np.zeros(len(dm)))
            # dm.setActuators(np.random.uniform(-0.5,0.5,size=len(dm)))
            time.sleep(1)

            plt.figure()
            img = grabframes(5, 1)
            plt.imshow(img[-1])
            plt.colorbar()

            plt.figure()
            img = grabframes(5, 2)
            plt.imshow(img[-1])
            plt.colorbar()


print('finish operation')
