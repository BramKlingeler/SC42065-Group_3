"""
Random Walk algorithm for PC in right corner

Setup right corner:
    Camera 1: Imaging camera
    Camera 2: SH sensor
"""
from dm.okotech.dm import OkoDM

from camera.ueye_camera import uEyeCamera
from pyueye import ueye

from scipy.optimize import minimize 
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import cv2

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
        



        def calculate_aberrations(image):
            # Convert the image to float for accurate calculations
            image = image.astype(np.float32)

            # Calculate the gradient of the image
            gradient_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
            gradient_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)

            # Calculate the wavefront error as the root mean square of the gradients
            wavefront_error = np.sqrt(np.mean(gradient_x**2 + gradient_y**2))

        return wavefront_error

        def apply_deformable_mirror(image, voltages):
            s_time = 0.01  # sleep time (small amount of time between steps)
            w_time = 0.05  # wait time around focus
            steps = 10
            prev_act_amp = 0
            # increase actuator voltage gradually, then reverse, hold at 0
            for i in range(steps):
                current = np.ones(num_actuators)#np.zeros(num_actuators) for resetting the selected actuators
                #current[j] = 1, only needed for seperate control
                act_amp = 0.8 / steps * voltages * (steps + 1) #+ prev_act_amp / ((0.3*i)**3 + 1) #standard coeff
                dm.setActuators(act_amp)
                time.sleep(s_time)  # in seconds
            
            img = grabframes(1, 1)
            modified_image = img[0,320:960,256:768]
            
            for i in range(steps):
                act_amp = 0.8 / steps * voltages * (steps - i)
                dm.setActuators(act_amp)
                time.sleep(s_time)  # in seconds
                # print(act_amp[0])
            #prev_act_amp = act_amp
            #time.sleep(w_time)
            
        return modified_image

        def random_walk_optimization(image, num_iterations, step_size):
            best_aberration = float('inf')
            best_image = np.copy(image)

            num_actuators = 19

            for _ in range(num_iterations):
                voltages = np.random.uniform(-step_size, step_size, size=num_actuators)
                voltages[17] = 0
                voltages[18] = 0

                modified_image = apply_deformable_mirror(voltages)

                aberration = calculate_aberrations(modified_image)

                if aberration < best_aberration:
                    best_aberration = aberration
                    best_image = modified_image

            return best_image, best_aberration

        # Load the image
        img = grabframes(1, 1)
        image = img[0,320:960,256:768]

        # Set the parameters
        num_iterations = 1000
        step_size = 10
        
        
        dm.setActuators(np.zeros(len(dm)))        

        # Perform the random walk optimization
        optimized_image, best_aberration = random_walk_optimization(image, num_iterations, step_size)

        # Display the optimized image and best aberration
        cv2.imshow('Optimized Image', optimized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print('Best Aberration:', best_aberration)

        
print('finish operation')
