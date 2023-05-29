from scipy.optimize import minimize 
import matplotlib.pyplot as plt
import numpy as np 

def func(x):
    x = x**2+2*np.sin(x*np.pi)
    return x

 
x = np.arange(-2, 2, 0.01)
y = func(x)

x0 = [-1, 1.9, 1.6, 1, 0.4]
ig, ax = plt.subplots(len(x0), figsize=(6, 8))
i = 0
for i in range(len(x0)):
    result = minimize(func,  x0[i], method="nelder-mead")
    ax[i].plot(x, y, label="y")
    ax[i].plot(result['x'], result['fun'], 'sr', label="minimum")
    ax[i].set_title("Starts from " + str(x0[i]))
    ax[i].legend(loc='best', fancybox=True, shadow=True)
    ax[i].grid()

minindex=ax.min()
print(minindex)

plt.tight_layout()        
plt.show()