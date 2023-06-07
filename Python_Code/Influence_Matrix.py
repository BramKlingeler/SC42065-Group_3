"""
Builds influence matrix
"""
from dm.okotech.dm import OkoDM

from camera.ueye_camera import uEyeCamera
from pyueye import ueye
import glob

import numpy as np
import matplotlib.pyplot as plt
import time
import random

SH_Sensor_Index = 2
Camera_Index = 1

def create_reference_fixed_grid(img):
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #find threshod
    retv ,thresh=cv2.threshold(img_gray,50,255,cv2.THRESH_BINARY)
    #find contour
    contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #draw contour
    #cv2.drawContours(img_original,contours,-1,(0,0,255),3,lineType=cv2.LINE_AA)
    #show image
    #cv2.imshow('Contours',img_original)
    #cv2.waitKey()
    #cv2.destroyAllWindows()
    points = np.zeros((len(contours),2))
    i=0
    for contour in contours:
        # Calculate the moments of the contour
        moments = cv2.moments(contour)
        # Calculate the centroid coordinates
        if moments['m00'] !=0:
            center_x = int(moments['m10'] / moments['m00'])
            center_y = int(moments['m01'] / moments['m00'])
        else:
            center_x = 0
            center_y = 0
        # Add the point to the list
        points[i][0]=center_x
        points[i][1]=center_y
        i=i+1
    return points

def get_slopes(image, reference_centers):
    # Draw grid around reference centers
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    grid_size = np.abs(reference_centers[1,0] - reference_centers[0,0])
    # Number of row and col of gird
    grid_rows = int((np.max(reference_centers[:, 1]) - np.min(reference_centers[:, 1])) / grid_size) + 1
    grid_cols = int((np.max(reference_centers[:, 0]) - np.min(reference_centers[:, 0])) / grid_size) + 1
    # draw the gird
    abs_centriod=np.zeros((len(reference_centers),2))
    for i in range(reference_centers.shape[0]):
        # Some definition
        width=int(grid_size/2)
        center_x = int(reference_centers[i,0])
        center_y = int(reference_centers[i,1])
        # Divide image into small grid matrix
        # Coordinate of opnecv and numpy.array is opposite!!
        small_grid = image[(center_y - width):(center_y + width), (center_x - width):(center_x + width)]
        # Find CoG of small matrix
        _, thresholded_image = cv2.threshold(small_grid, 50, 255, cv2.THRESH_BINARY)
        moments = cv2.moments(thresholded_image)
        r_centriod=np.zeros((2,1))
        if moments['m00'] !=0:
            r_centriod[0] = moments['m10'] / moments['m00']
            r_centriod[1] = moments['m01'] / moments['m00']
        else:
            r_centriod[0] = 0
            r_centriod[1] = 0
        abs_centriod[i,0] = center_x-width+r_centriod[0]
        abs_centriod[i,1] = center_y-width+r_centriod[1]
        # Draw the rectangle and position of CoG
    for i in range(reference_centers.shape[0]):
        center_x=int(reference_centers[i,0])
        center_y=int(reference_centers[i,1])
        draw_1=cv2.rectangle(image, (center_x-width,center_y-width),(center_x+width, center_y+width), (255,255,255), 2)
        draw_1=cv2.circle(draw_1, (int(abs_centriod[i,0]), int(abs_centriod[i,1])), 5, (255,255,255))
    cv2.imshow('grid',draw_1)
    cv2.waitKey()
    cv2.destroyAllWindows()
    # Calculate deviation from reference center
    deviations = abs_centriod-reference_centers
    return deviations

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
        for j in range(num_actuators):
            a == True

            while True:
                s_time = 0.01  # sleep time (small amount of time between steps)
                w_time = 0.5  # wait time around focus
                # increase actuator voltage gradually, then reverse, hold at 0
                for i in range(num_actuators):
                    current = np.zeros(num_actuators)
                    current[j] = 1
                    act_amp = current
                    dm.setActuators(act_amp)
                    time.sleep(s_time)  # in seconds
                    one_frame = grabframes(1)
                    filename = 'Voltage_1_actuator_{}.png'.format(i)  # Filename with the changing part
                    cv2.imwrite(filename, one_frame)


                dm.setActuators(np.zeros(len(dm)))
                time.sleep(w_time)

                # decrease actuator voltage gradually, then reverse, hold at 0
                for i in range(num_actuators):
                    current = np.zeros(num_actuators)
                    current[i] = -1
                    act_amp = current
                    dm.setActuators(act_amp)
                    time.sleep(s_time)  # in seconds
                    one_frame = grabframes(1)
                    filename = 'Voltage_-1_actuator_{}.png'.format(i)  # Filename with the changing part
                    cv2.imwrite(filename, one_frame)

                dm.setActuators(np.zeros(len(dm)))
                time.sleep(w_time)
                a == False

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


print('Finished')



num_actuators = len(dm)

img_original = cv2.imread(r"C:\Users\Tim\Desktop\61reference.png")

ref_grid = create_reference_fixed_grid(img_original)

for j in range(num_actuators):
    file_pattern_1 = 'Voltage_1_actuator_*.png'  # File pattern to match the filenames
    file_pattern_2 = 'Voltage_-1_actuator_*.png'

    # Get a list of matching file paths
    file_paths_1 = glob.glob(file_pattern_1)
    file_paths_2 = glob.glob(file_pattern_2)

    for file_path in file_paths:
        # Read the image using OpenCV
        image_1 = cv2.imread(file_path_1)
        image_2 = cv2.imread(file_path_2)
        column_slopes_1 = get_slopes(image_1,ref_grid)
        column_slopes_2 = get_slopes(image_2, ref_grid)
        column_total = (column_slopes_1 - column_slopes_2) / 2
        # Perform operations on the image as needed
        Z.append(column_total)

# Convert the matrix Z to a numpy array
Z = np.array(Z)





