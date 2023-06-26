N = 100 # number of iterations
step = 0.5 # length of initial step
epsilon = 0.001
variables = 19 # number of variables
#x = np.random.uniform(-1,1,(19,1))# initial position
x=np.zeros(19)
walk_num = 1 # initialize the number of random_walk
print("iterations:",N)
print("length initial step:",step)
print("epsilon:",epsilon)
print("number of variables:",variables)

#2.3 Codes 2 W A VEFRONT SENSORLESS CORRECTION PART I
print("initial position:",x)
# define a function to calculate the maximum value of intensity from figure
def function(x):
 intensity=np.amax(img[-1])
return intensity

# loop of random walk
while(step > epsilon):
 k = 1 # initialization of enumerator
 while(k < N):
  u = [np.random.uniform(-1,1) for i in range(variables)] # random vector
    # u1 normalize of random vector
#  u1 = [u[i]/math.sqrt(sum([u[i]**2 for i in range(variables)])) for i in range(variables)]
#  x1 = [x[i] + step*u1[i] for i in range(variables)]
  if __name__ == "__main__":
   from dm.thorlabs.dm import ThorlabsDM
   a=np.linspace(-0.9,0.9,21)
   with ThorlabsDM() as dm:
    while False:
     for i in range(21):
      act=np.ones([len(dm)])*a[i]
      print(act)
      dm.setActuators(act)
      time.sleep(0.1)
     for i in range(21):
      act=np.ones([len(dm)])*a[i]*-1
      dm.setActuators(act)
      time.sleep(0.1)

    if True:
     print(f"Deformable mirror with {len(dm)} actuators")
     dm.setActuators(x1)

     plt.figure()
     img=grabframes(5, Camera_Index)
     plt.imshow(img[-1])
   if(function(x1)>function(x)): # if we find a better point
    k = 1
    x = x1
   else:
    k += 1
    step = step/2
    print(" %d time of random walk" % walk_num)
    walk_num += 1
 print("number of random walk:",walk_num-1)
 print("finally best solution:",x1)
 print("finally optimization value:",function(x))