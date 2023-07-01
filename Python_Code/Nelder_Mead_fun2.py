from dm.okotech.dm import OkoDM

from camera.ueye_camera import uEyeCamera
from pyueye import ueye


import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import minimize

class NelderMead:

	def __init__(self, dm) -> None: 
		self.dm = dm
#      self.history = []
#      self.solution = []
#      self.evaluation = []
		
	def f(self, x):
        # Set actuators
         dm.setActuators(x)
         time.sleep(0.09)
        
        # Record & report cost (Define of the light centroid and corresponding simplex)    
         img = grabframes(1, 1)
         cropped_img = img[0,320:960,256:768]
         sum_xy = 0
         weighted_sum_k=0
         weighted_sum_h=0
         bright_sum=0
         
         for k in range(np.shape(cropped_img)[0]):
                for h in range(np.shape(cropped_img)[1]):
                    weighted_sum_k=weighted_sum_k+cropped_img[k,h]*k
                    weighted_sum_h=weighted_sum_h+cropped_img[k,h]*h
                    bright_sum=bright_sum+cropped_img[k,h]
  
         k_c = weighted_sum_k / bright_sum_k
         h_c = weighted_sum_h / bright_sum_h
    
         weighted_sum_var=0
         for k in range(np.shape(cropped_img)[0]):
                for h in range(np.shape(cropped_img)[1]):
                    d2=(k-k_c)**2+(h-h_c)**2
                    weighted_sum_var = weighted_sum_var + d2*cropped_img[k,h]
            
            
         var_d =int(weighted_sum_var/bright_sum)
         
         return var_d
     
	# Nelder-Mead Optimization	
	def nelder_mead (self, x0=np.zeros(19)):
         history = []
         opts = {'maxiter': 5000, 'adaptive': True}
         def callback(x):
                fobj = self.f(x)
                history.append(fobj)

         result = minimize(self.f, x0, method='Nelder-Mead', tol=1e-6, callback=callback, options = opts)
		
         print('Status: %s' %result['message'])
         print('Total Evaluations: %d' %result['nfev'])

         solution=result['x']
         evaluation = result['fun']
         print('Solution: f(%s) = %.5f' %(solution, evaluation))
         #plot history
         plt.plot(history)
         plt.show()
		
if __name__ == '__main__':
	a = NelderMead(None)
	
	a.nelder_mead(np.ones(19))