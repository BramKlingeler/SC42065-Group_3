"""
Builds influence matrix
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

SH_Sensor_Index = 2
Camera_Index = 1

#act_initial=[-0.62627906, -0.57748853, -1.02261049, -0.27942378,  0.32247178,  1.19441498,
# -0.5018988,  -0.76241651,  0.1200851,  -0.53391441,  0.5560053,   1.21881588,
#  0.34127717, -0.24623419, 0.82221155, -0.28707566, -0.96491517,  0.16536202,
# -0.86804879]
act_initial= np.zeros([19])
print(act_initial)
#def create_reference_fixed_grid(img):
#    img_gray=img
#    #find threshod
#    retv ,thresh=cv2.threshold(img_gray,50,255,cv2.THRESH_BINARY)
#    #find contour
#    thresh = thresh.astype(np.uint8)
#    _, contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#    #draw contour
#    #cv2.drawContours(img_original,contours,-1,(0,0,255),3,lineType=cv2.LINE_AA)
#    #show image
#    #plt.imshow(draw)
#    #cv2.waitKey()
#    #cv2.destroyAllWindows()
#    points = np.zeros((len(contours),2))
#    i=0
#    for contour in contours:
#        # Calculate the moments of the contour
#        moments = cv2.moments(contour)
#        # Calculate the centroid coordinates
#        if moments['m00'] !=0:
#            center_x = int(moments['m10'] / moments['m00'])
#            center_y = int(moments['m01'] / moments['m00'])
#        else:
#            center_x = 0
#            center_y = 0
#        # Add the point to the list
#        points[i][0]=center_x
#        points[i][1]=center_y
#        i=i+1
#    return points
#
#
#def get_slopes(image, reference_centers):
#    # Draw grid around reference centers
#    image_d=image
#    #image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#    #grid_size = np.abs(reference_centers[1,0] - reference_centers[0,0])
#    grid_size=58*0.9
#    # Number of row and col of gird
#    grid_rows = int((np.max(reference_centers[:, 1]) - np.min(reference_centers[:, 1])) / grid_size) + 1
#    grid_cols = int((np.max(reference_centers[:, 0]) - np.min(reference_centers[:, 0])) / grid_size) + 1
#    # draw the gird
#    abs_centriod=np.zeros((len(reference_centers),2))
#    for i in range(reference_centers.shape[0]):
#        # Some definition
#        width=int(grid_size/2)
#        center_x = int(reference_centers[i,0])
#        center_y = int(reference_centers[i,1])
#        # Divide image into small grid matrix
#        # Coordinate of opnecv and numpy.array is opposite!!
#        small_grid = image[(center_y - width):(center_y + width), (center_x - width):(center_x + width)]
#        # Find CoG of small matrix
#        _, thresholded_image = cv2.threshold(small_grid, 50, 255, cv2.THRESH_BINARY)
#        moments = cv2.moments(thresholded_image)
#        r_centriod=np.zeros((2,1))
#        if moments['m00'] !=0:
#            r_centriod[0] = moments['m10'] / moments['m00']
#            r_centriod[1] = moments['m01'] / moments['m00']
#        else:
#            r_centriod[0] = 0
#            r_centriod[1] = 0
#        abs_centriod[i,0] = center_x-width+r_centriod[0]
#        abs_centriod[i,1] = center_y-width+r_centriod[1]
#        # Draw the rectangle and position of CoG
#    for i in range(reference_centers.shape[0]):
#        center_x=int(reference_centers[i,0])
#        center_y=int(reference_centers[i,1])
#        
#        draw_1=cv2.rectangle(image_d, (center_x-width,center_y-width),(center_x+width, center_y+width), (255,255,255), 2)
#        draw_1=cv2.circle(draw_1, (int(abs_centriod[i,0]), int(abs_centriod[i,1])), 5, (255,255,255))
#        draw_1=cv2.circle(draw_1, (center_x, center_y), 5, (0,255,255))
#    new_draw = cv2.resize(draw_1, None, fx = 0.75, fy = 0.75)
#    #cv2.imshow('grid',new_draw) 
#    #cv2.waitKey()
#    #cv2.destroyAllWindows()
#    # Calculate deviation from reference center
#    deviations = abs_centriod-reference_centers 
#    return deviations
#
#def grabframes(nframes, cameraIndex=0):
#    #with uEyeCamera(device_id=1) as cam:
#    with uEyeCamera(device_id=2) as cam:
#        cam.set_colormode(ueye.IS_CM_MONO8)  # IS_CM_MONO8)
#        w = 1280
#        h = 1024
#        # cam.set_aoi(0,0, w, h)
#
#        cam.alloc(buffer_count=10)
#        cam.set_exposure(0.908)
#        cam.capture_video(True)
#
#        imgs = np.zeros((nframes, h, w), dtype=np.uint8)
#        acquired = 0
#        # For some reason, the IDS cameras seem to be overexposed on the first frames (ignoring exposure time?).
#        # So best to discard some frames and then use the last one
#        while acquired < nframes:
#            frame = cam.grab_frame()
#            if frame is not None:
#                imgs[acquired] = frame
#                acquired += 1
#
#        cam.stop_video()
#
#    return imgs
#
#def normalize_coordinates(points):
#    # Calculate the center of the points
#    center = np.mean(points, axis=0)
#    # Calculate the maximum distance from the center to any point
#    max_distance = np.max(np.linalg.norm(points - center, axis=1))
#    # Normalize the coordinates
#    normalized_points = (points - center) / max_distance
#
#    return normalized_points
#
#
#def get_Bmatrix(points,Zorder):
#    cart = RZern(Zorder)
#    L, K = 2000, 2000
#    ddx = np.linspace(-1.0, 1.0, K)
#    ddy = np.linspace(-1.0, 1.0, L)
#    xv, yv = np.meshgrid(ddx, ddy)
#    cart.make_cart_grid(xv, yv)
#
#    cart.ZZ = np.zeros((xv.size, cart.nk),order='F', dtype=cart.numpy_dtype)
#    cart.shape = xv.shape
#    rho = np.sqrt(np.square(xv) + np.square(yv))
#    theta = np.arctan2(yv, xv)
#    for k in range(cart.nk):
#        prod = cart.radial(k, rho) * cart.angular(k, theta)
#        #Code below can be comment
#        prod[rho > 1.0] = 0
#        cart.ZZ[:, k] = cart.vect(prod)
#
#    B=np.zeros((2*points.shape[0],cart.nk))
#    for i in range(cart.nk):
#        #Define coefficient
#        c = np.zeros((cart.nk,1))
#        c[i] = 1
#        Phi = cart.eval_grid(c, matrix=True)
#        for j in range(points.shape[0]):
#            #find where is point in pixel
#            pixel_x=int(points[j,0]*L/2)
#            pixel_y=int(points[j,1]*K/2)
#            #Gradient of y in certain point
#            B[2*j+1,i]=(Phi[pixel_y+1,pixel_x]-Phi[pixel_y-1,pixel_x])/2
#            #Gradient of x in certain point
#            B[2*j,i]=(Phi[pixel_y,pixel_x+1]-Phi[pixel_y,pixel_x-1])/2
#    return B,cart


def create_reference_fixed_grid(img):
    img_gray=img
    #find threshod
    retv ,thresh=cv2.threshold(img_gray,50,255,cv2.THRESH_BINARY)
    #find contour
    thresh = thresh.astype(np.uint8)
    _, contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #draw contour
    #cv2.drawContours(img_original,contours,-1,(0,0,255),3,lineType=cv2.LINE_AA)
    #show image
    #plt.imshow(draw)
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
    image_d=image
    #image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #grid_size = np.abs(reference_centers[1,0] - reference_centers[0,0])*0.6
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
    #plt.figure()
    #plt.imshow(new_draw)
    #cv2.imshow('grid',new_draw) 
    #cv2.waitKey()
    #cv2.destroyAllWindows()
    # Calculate deviation from reference center
    deviations = abs_centriod-reference_centers 
    return deviations

def grabframes(nframes, cameraIndex=0):
    #with uEyeCamera(device_id=1) as cam:
    with uEyeCamera(device_id=2) as cam:
        cam.set_colormode(ueye.IS_CM_MONO8)  # IS_CM_MONO8)
        w = 1280
        h = 1024
        # cam.set_aoi(0,0, w, h)

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
        #prod[rho > 1.0] = 0
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

def translate_image(image, tx, ty):
    rows, cols = image.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_image = cv2.warpAffine(image, M, (cols, rows))
    return translated_image

def center_image(image):
    image_center = np.shape(image)
    #print(image_center)
    cog_x = np.mean(np.where(image > 50)[1])
    cog_y = np.mean(np.where(image > 50)[0])
    translated_image = translate_image(image, image_center[1]/2-cog_x, image_center[0]/2-cog_y)
    return translated_image

def image_fix(image):
    img_r = rotate_image(image, 4)
    img_c = center_image(img_r)
    return img_r

if __name__ == "__main__":
    from dm.okotech.dm import OkoDM

    print("started")
    # Use "with" blocks so the hardware doesn't get locked when you press ctrl-c
    with OkoDM(dmtype=1) as dm:
        print(f"Deformable mirror with {len(dm)} actuators")
        # set actuators to 0
        s_time = 0.01  # sleep time (small amount of time between steps)
        w_time = 0.1  # wait time around focus
        #act = np.zeros([len(dm)])
        act=act_initial
        dm.setActuators(act)
        num_actuators = len(dm)
        time.sleep(w_time)
        img_original = grabframes(5,2)
        img_original_2 = image_fix(img_original[-1])
        #get reference point
        ref_points = create_reference_fixed_grid(img_original_2)
        #Some aberration need to be fixed
        S_fix=get_slopes(img_original_2, ref_points)
        #update reference points
        normalized_pos=normalize_coordinates(ref_points)
        B,cart=get_Bmatrix(normalized_pos,6)
        print('Finished B matrix')

        DM_ones = np.zeros((np.size(ref_points),num_actuators))
        DM_negative_ones = np.zeros((np.size(ref_points),num_actuators))
        print(np.shape(ref_points))
        for j in range(num_actuators):
            # increase actuator voltage gradually, then reverse, hold at 0
            current = np.zeros(num_actuators)
            current[j] = 0.5
            act_amp = current+act_initial
            dm.setActuators(act_amp)
            time.sleep(w_time)  # in seconds
            one_frame = grabframes(5,2)
            one_frame = image_fix(one_frame[-2])
            
            #plt.savefig('ref.jpg')
            #one_frame=cv2.imread('ref.jpg')
                    
            slopes=np.reshape(get_slopes(one_frame,ref_points)-S_fix,(-1,1))
            #print(np.shape(get_slopes(one_frame,ref_grid)))
            for k in range(np.size(ref_points)):
                DM_ones[k,j] = slopes[k]
                #DM_ones[:,i] = np.reshape(get_slopes(one_frame,ref_grid),(-1,1))    
            
        dm.setActuators(np.zeros(len(dm)))
        time.sleep(w_time)

        #decrease actuator voltage gradually, then reverse, hold at 0
        for i in range(num_actuators):
            current = np.zeros(num_actuators)
            current[i] = -0.5
            act_amp = current+act_initial
            dm.setActuators(act_amp)
            time.sleep(w_time)  # in seconds
            one_frame = grabframes(5,2)
            one_frame = image_fix(one_frame[-2])
                    
            slopes=np.reshape(get_slopes(one_frame,ref_points)-S_fix,(-1,1))
            for k in range(np.size(ref_points)):
                DM_negative_ones[k,i] = slopes[k]
                #DM_negative_ones[i] = get_slopes(one_frame,ref_grid)

        dm.setActuators(np.zeros([len(dm)]))
        time.sleep(w_time)
        #a == False
        column_total = (DM_ones - DM_negative_ones) / 2
        #column_total = DM_ones
        # Perform operations on the image as needed

        # Convert the matrix Z to a numpy array
        C = np.array(2*column_total)

        print('Finished C matrix')
        

#%% Control test
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D        


if __name__ == "__main__":
    from dm.okotech.dm import OkoDM

    print("started")
    with OkoDM(dmtype=1) as dm:
        #For 3D image
        xx = np.arange(2000)
        yy = np.arange(2000)
        X, Y = np.meshgrid(xx, yy)
        
        Z_order=cart.nk
        #Reference Coeffecients
        z=np.zeros((Z_order,1))
        z[1]=100
        
        
        
        
        Phi0=cart.eval_grid(z, matrix=True)
        fig = plt.figure()  
        ax3 = plt.axes(projection='3d')
        ax3.plot_surface(X,Y,Phi0,cmap='rainbow')
        plt.title('Target')
        #ax3.contour(X,Y,Z, zdim='z',offset=-2，cmap='rainbow)   
        plt.show()
        
        #Zernike to slope, S is desired slope
        S=np.dot(B,z)

        #RMS=0
        #test C
        
#        invC=np.linalg.pinv(C)
#        desired_V=np.dot(invC,S)
#        act=desired_V
        
        #act=np.zeros(len(dm))
        act=act_initial
        act[0]=act[0]+0.6
#        act[-2]=0.05
#        act[-1]=-0.13
        
        dm.setActuators(act)
        V=np.zeros((19,1))
        V[0]=-0.6
        np.reshape(V, (19, 1))
        print(np.shape(V))
        time.sleep(s_time)
        one_frame = grabframes(5,2)
        one_frame = image_fix(one_frame[-1])
        #new slope is observed
        
        new_slopes=np.reshape(get_slopes(one_frame,ref_points)-S_fix,(-1,1))
        
        #test B matrix
        invB=np.linalg.pinv(B)
        test_cof=np.dot(invB,new_slopes)
        print(np.shape(V))
        
        Phi=cart.eval_grid(test_cof, matrix=True)
#        plt.figure()
#        plt.imshow(Phi)
#        plt.plot(Phi[:,1000])
#        plt.plot(Phi[1000,:])
        # 3D pattern
        fig = plt.figure()  #定义新的三维坐标轴
        ax3 = plt.axes(projection='3d')

        ax3.plot_surface(X,Y,Phi,cmap='rainbow')
        plt.title('Aberration')
        #ax3.contour(X,Y,Z, zdim='z',offset=-2，cmap='rainbow)   #等高线图，要设置offset，为Z的最小值
        plt.show()
        deltaS = S-new_slopes
        square=0
        for j in range(np.size(deltaS)):
            square=square+deltaS[j]**2
        RMS=np.sqrt(square)/(np.size(deltaS))

        print(RMS)
        invC=np.linalg.pinv(C)

        #control slope
        iterations=10
        # Kalman filter parameters
        dimension = len(new_slopes)
        A = np.zeros((dimension,dimension)) # State transition matrix
        B_K = C  # Control matrix (if applicable)
        H = np.eye(dimension)  # Measurement matrix
        Q = np.eye(dimension)  # Process noise covariance matrix
        R = np.eye(dimension)  # Measurement noise covariance matrix

        x = np.zeros((dimension, 1))  # Initial state estimate
        P = np.eye(dimension)  # Initial state covariance

        






        for i in range (iterations):

            #RMS_old=RMS
#            desired_V=np.dot(invC,S)
#            V_mean = np.mean(desired_V)
            
            # Prediction step
            x_predicted = np.dot(A, x)  # Predicted state estimate
            P_predicted = np.dot(A, np.dot(P, A.T)) + Q  # Predicted state covariance

            # Update step
            z = new_slopes  # Measurement (current slopes observation)
            innovation = z - np.dot(H, x_predicted)  # Innovation or measurement residual
            S_k = np.dot(H, np.dot(P_predicted, H.T)) + R  # Innovation covariance
            K = np.dot(P_predicted, np.dot(H.T, np.linalg.inv(S_k)))  # Kalman gain
            x = x_predicted + np.dot(K, innovation)  # Updated state estimate
            P = np.dot((np.eye(dimension) - np.dot(K, H)), P_predicted)  # Updated state covariance

            # Update actuator values (input)
            x_conv = np.dot(invC,x)
            V = V + 0.2 * x_conv  # Adjust the input based on the estimated state

            V_mean=np.mean(V)*np.ones((19,1))
#            print(np.shape(V))
            #V_mean[0:17]=np.mean(V[0:17])
            V=V-V_mean
            
            dm.setActuators(V)
            #print(V)
            
#            dm.setActuators(desired_V-V_mean)
            time.sleep(w_time)
            one_frame = grabframes(5,2)
            one_frame = image_fix(one_frame[-1])
            
            new_slopes=np.reshape(get_slopes(one_frame,ref_points)-S_fix,(-1,1))
            deltaS=S-new_slopes
            
            print(np.shape(deltaS))
            square=0
            for k in range(np.shape(deltaS)[0]):
                square=square+deltaS[k]**2
            RMS=np.sqrt(np.abs(square))/(np.size(deltaS))
            print(RMS)
#            delta_RMS=RMS-RMS_old
#            delta_RMS=delta_RMS/RMS
            
#            if delta_RMS <0.05:
#                print(i)
#                break
        
        #test B matrix
        cof=np.dot(invB,new_slopes)
#        print(test_cof)
        Phi1=cart.eval_grid(cof, matrix=True)
#        plt.figure()
#        plt.imshow(Phi1)

        act=act_initial
        dm.setActuators(act)
        
        
        fig = plt.figure()  
        ax3 = plt.axes(projection='3d')
        plt.title('Result')

        #定义三维数据

        #作图
        ax3.plot_surface(X,Y,Phi1,cmap='rainbow')
        #ax3.contour(X,Y,Z, zdim='z',offset=-2，cmap='rainbow)   #等高线图，要设置offset，为Z的最小值
        plt.show()

#num_actuators = len(dm)

#for j in range(num_actuators):
#    file_pattern_1 = 'Voltage_1_actuator_*.png'  # File pattern to match the filenames
#    file_pattern_2 = 'Voltage_-1_actuator_*.png'
#
#    # Get a list of matching file paths
#    file_paths_1 = glob.glob(file_pattern_1)
#    file_paths_2 = glob.glob(file_pattern_2)
#
#    for file_path in file_paths:
#        # Read the image using OpenCV
#        image_1 = cv2.imread(file_path_1)
#        image_2 = cv2.imread(file_path_2)
#        column_slopes_1 = get_slopes(image_1,ref_grid)
#        column_slopes_2 = get_slopes(image_2, ref_grid)
#        column_total = (column_slopes_1 - column_slopes_2) / 2
#        # Perform operations on the image as needed
#       Z.append(column_total)
#
## Convert the matrix Z to a numpy array
#Z = np.array(Z)





