"""
Builds influence matrix
"""
from dm.okotech.dm import OkoDM

from camera.ueye_camera import uEyeCamera
from pyueye import ueye
import cv2
from zernike import RZern
from scipy.ndimage.measurements import center_of_mass
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D        


SH_Sensor_Index = 2
Camera_Index = 1

act_initial=[-0.62627906, -0.57748853, -1.02261049, -0.27942378,  0.32247178,  1.19441498,
 -0.5018988,  -0.76241651,  0.1200851,  -0.53391441,  0.5560053,   1.21881588,
  0.34127717, -0.24623419, 0.82221155, -0.28707566, -0.96491517,  0.16536202,
 -0.86804879]


def create_reference_fixed_grid(img):
    img_gray=img
    #find threshod
    retv ,thresh=cv2.threshold(img_gray,50,255,cv2.THRESH_BINARY)
    #find contour
    thresh = thresh.astype(np.uint8)
    _, contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
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
    image_d=image
    grid_size=58*0.9
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
        draw_1=cv2.rectangle(image_d, (center_x-width,center_y-width),(center_x+width, center_y+width), (255,255,255), 2)
        draw_1=cv2.circle(draw_1, (int(abs_centriod[i,0]), int(abs_centriod[i,1])), 5, (255,255,255))
        draw_1=cv2.circle(draw_1, (center_x, center_y), 5, (0,255,255))
    new_draw = cv2.resize(draw_1, None, fx = 0.75, fy = 0.75)
    plt.figure()
    plt.imshow(new_draw)
    deviations = abs_centriod-reference_centers 
    return deviations

def grabframes(nframes, cameraIndex=0):
    #with uEyeCamera(device_id=1) as cam:
    with uEyeCamera(device_id=2) as cam:
        cam.set_colormode(ueye.IS_CM_MONO8)  # IS_CM_MONO8)
        w = 1280
        h = 1024
        cam.alloc(buffer_count=10)
        cam.set_exposure(0.908)
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

def normalize_coordinates(points):
    # Calculate the center of the points
    center = np.mean(points, axis=0)
    # Calculate the maximum distance from the center to any point
    max_distance = np.max(np.linalg.norm(points - center, axis=1))
    # Normalize the coordinates
    normalized_points = (points - center) / max_distance
    return normalized_points


def get_Bmatrix(points,Zorder):
    cart = RZern(Zorder)
    L, K = 2000, 2000
    ddx = np.linspace(-1.0, 1.0, K)
    ddy = np.linspace(-1.0, 1.0, L)
    xv, yv = np.meshgrid(ddx, ddy)
    cart.make_cart_grid(xv, yv)

    cart.ZZ = np.zeros((xv.size, cart.nk),order='F', dtype=cart.numpy_dtype)
    cart.shape = xv.shape
    rho = np.sqrt(np.square(xv) + np.square(yv))
    theta = np.arctan2(yv, xv)
    for k in range(cart.nk):
        prod = cart.radial(k, rho) * cart.angular(k, theta)
        #Code below can be comment
#        prod[rho > 1.0] = 0
        cart.ZZ[:, k] = cart.vect(prod)

    B=np.zeros((2*points.shape[0],cart.nk))
    for i in range(cart.nk):
        #Define coefficient
        c = np.zeros((cart.nk,1))
        c[i] = 1
        Phi = cart.eval_grid(c, matrix=True)
        for j in range(points.shape[0]):
            #find where is point in pixel
            pixel_x=int(points[j,0]*L/2)
            pixel_y=int(points[j,1]*K/2)
            #Gradient of y in certain point
            B[2*j+1,i]=(Phi[pixel_y+1,pixel_x]-Phi[pixel_y-1,pixel_x])/2
            #Gradient of x in certain point
            B[2*j,i]=(Phi[pixel_y,pixel_x+1]-Phi[pixel_y,pixel_x-1])/2
    return B,cart

def rotate_image(image, angle):
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, M, (cols, rows))
    return rotated_image



if __name__ == "__main__":
    from dm.okotech.dm import OkoDM

    print("started")
    # Use "with" blocks so the hardware doesn't get locked when you press ctrl-c
    with OkoDM(dmtype=1) as dm:
        print(f"Deformable mirror with {len(dm)} actuators")
        # set actuators to 0
        s_time = 0.01  # sleep time (small amount of time between steps)
        w_time = 0.5  # wait time around focus
        act = np.zeros([len(dm)])
        # get new image
        dm.setActuators(act)
        num_actuators = len(dm)
        time.sleep(w_time)
        img_original = grabframes(5,2)
        img_original_2 = img_original[-1]
        img_original_2 = rotate_image(img_original_2, 4)
        
        #get reference point 
        plt.figure()
        plt.imshow(img_original_2)
        ref_points = create_reference_fixed_grid(img_original_2)

        #Some aberration need to be fixed
        S_fix=get_slopes(img_original_2, ref_points)

        normalized_pos=normalize_coordinates(ref_points)
        B,cart=get_Bmatrix(normalized_pos,6)
        
        act[0] = -0.8
        dm.setActuators(act)
        time.sleep(w_time)
        imgs_new = grabframes(5,2)
        img_new = imgs_new[-1]
        img_new = rotate_image(img_new, 4)
        plt.figure()
        plt.imshow(img_new)
       
        D = np.reshape(get_slopes(img_new, ref_points)-S_fix,(-1,1))
        invB=np.linalg.pinv(B)
        cof=np.dot(invB,D)
        Phi=cart.eval_grid(cof, matrix=True)        
        
        #For 3D image
        L=2000
        xx = np.arange(L)
        yy = np.arange(L)
        X, Y = np.meshgrid(xx, yy)
        fig = plt.figure()  
        ax3 = plt.axes(projection='3d')
        ax3.plot_surface(X,Y,Phi,cmap='rainbow')       
        plt.show()








