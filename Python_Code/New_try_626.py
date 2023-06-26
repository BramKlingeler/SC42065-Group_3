import math
N = 20 # number of iterations
step = 0.5 # length of initial step
epsilon = 0.001
variables = 19 # number of variables
#x = np.random.uniform(-1,1,(19,1))# initial position
walk_num=1
step=1
Camera_Index=1
print("iterations:",N)
print("length initial step:",step)
print("epsilon:",epsilon)
print("number of variables:",variables)

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
        cam.set_exposure(0.5)
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

def function(img):
    intensity=np.amax(img[-1])
    return intensity

# loop of random walk
if __name__ == "__main__":
    from dm.okotech.dm import OkoDM
    with OkoDM(dmtype=1) as dm:
        act=np.zeros([len(dm)])
        dm.setActuators(act)
        time.sleep(0.1)
        while False:
            act=np.zeros([len(dm)])
            dm.setActuators(act)
            time.sleep(0.1)
            
        if True:
            k=1
            img_n=grabframes(5, Camera_Index)
            range_u=0.9
            u_0=act
            while k<25:
                img_o=img_n
                u = u_0 + [np.random.uniform(-range_u,range_u) for i in range(variables)] # random vector
                dm.setActuators(u)
                time.sleep(0.1)
                img_n=grabframes(5, Camera_Index)
                plt.imshow(img_n[-1])
                if(function(img_n)>function(img_o)): # if we find a better point
                    k = 1
                    img_n=img_n
                    plt.imshow(img_n[-1])
                    plt.figure()
                    u_op=u
                    print(function(img_n))
                    print(u)
                else:
                    k += 1
                    img_n=img_o
                    if k>20:
                        range_u=0.5*range_u
                        u_0 = u_op
                        step=step+1
                        k=1
                    if step>5:
                        break
                print(" %d time of random walk" % walk_num)
                walk_num += 1
#[-0.78783364  0.46091063  0.60442698  0.81947666  0.34710144 -0.04037511
#  0.33315949 -0.17724003  0.06887213  0.08255487  0.61794182  0.56773376
# -0.94965443  0.12436213  0.83913617  0.42414116  0.88031037  0.71285349
#  0.54134661]

#        u = [np.random.uniform(-1,1) for i in range(variables)] # random vector
#    # u1 normalize of random vector
#        u1 = [u[i]/math.sqrt(sum([u[i]**2 for i in range(variables)])) for i in range(variables)]
#        #x1 = [x[i] + step*u1[i] for i in range(variables)]
#        x1 = [x[i] + u1[i] for i in range(variables)]
#                act=np.zeros([len(dm)])
#                dm.setActuators(act)
#                time.sleep(0.1)
#
#                    print("I am false!")
#
#                if True:
#                    print(f"Deformable mirror with {len(dm)} actuators")
#                    img_o=grabframes(5, Camera_Index)
#                    time.sleep(0.1)
#                    dm.setActuators(x1)
#                    time.sleep(0.1)
#                    plt.figure()
#                    img_n=grabframes(5, Camera_Index)
#                    plt.imshow(img_n[-1])
#                    if(function(img_n)>function(img_o)): # if we find a better point
#                        k = 1
#                        x = x1
#                        step = step*0.8
#                        print(function(img_n))
#                    else:
#                        k += 1
#                    print(" %d time of random walk" % walk_num)
#                    walk_num += 1
# print("number of random walk:",walk_num-1)
# print("finally best solution:",x1)
# print("finally optimization value:",function(x))