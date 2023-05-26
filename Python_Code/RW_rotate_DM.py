if __name__ == "__main__":
    from dm.okotech.dm import OkoDM
    import random

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
        

        # Code for 1-D random walk
        
        # Probability to move up or down
        prob = [0.15, 0.85] 
 
        # statically defining the starting position
        initial_voltage = random.randint(1,10) 
        voltage = [initial_voltage]
 
        # creating the random points
        rr = np.random.random(25)
        downp = rr < prob[0]
        upp = rr > prob[1]
 
        # for loop for making the walking process
        for idownp, iupp in zip(downp, upp):
            down = idownp and voltage[-1] > 0
            up = iupp and voltage[-1] < 10
            voltage.append(voltage[-1] - down + up)
            a = True
            voltage_val = voltage*0.8/10.0
            while a == True:
                s_time = 0.01  # sleep time (small amount of time between steps)
                w_time = 0.5  # wait time around focus
                steps = 10
                # increase actuator voltage gradually, then reverse, hold at 0
                for i in range(steps):
                    current = np.ones(num_actuators)#np.zeros(num_actuators) for resetting the selected actuators
                    #current[j] = 1, only needed for seperate control
                    act_amp = voltage_val / steps * current * (i + 1) #standard coeff                     dm.setActuators(act_amp)
                    time.sleep(s_time)  # in seconds
                    # print(act_amp[0])
                for i in range(steps):
                    act_amp = voltage_val / steps * current * (steps - i)
                    dm.setActuators(act_amp)
                    time.sleep(s_time)  # in seconds
                    # print(act_amp[0])

                dm.setActuators(np.zeros(len(dm)))
                time.sleep(w_time)

                # decrease actuator voltage gradually, then reverse, hold at 0
                for i in range(steps):
                    act_amp = -voltage_val / steps * current * (i + 1)
                    dm.setActuators(act_amp)
                    time.sleep(s_time)  # in seconds
                    # print(act_amp[0])
                for i in range(steps):
                    act_amp = -voltage_val / steps * current * (steps - i)
                    dm.setActuators(act_amp)
                    time.sleep(s_time)  # in seconds
                    # print(act_amp[0])

                dm.setActuators(np.zeros(len(dm)))
                time.sleep(w_time)
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
