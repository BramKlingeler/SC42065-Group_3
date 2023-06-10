import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import minimize

class NelderMead:

	def _init__(self, dm) -> None: 
		self.dm = dm
		
	def f(self, x):
		weighted_sum_k=0
        weighted_sum_h=0
        bright_sum_k=0
        bright_sum_h=0    
        for k in range(np.shape(cropped_img)[0]):
            for h in range(np.shape(cropped_img)[1]):
                weighted_sum_k=weighted_sum_k+cropped_img[k,h]*k
                weighted_sum_h=weighted_sum_h+cropped_img[k,h]*h
                bright_sum_k=bright_sum_k+cropped_img[k,h]
                bright_sum_h=bright_sum_h+cropped_img[k,h]
  
        k_c = weighted_sum_k/ bright_sum_k
        h_c = weighted_sum_h/bright_sum_h
    
        weighted_sum_var=0
        image_sum=0
        for k in range(np.shape(cropped_img)[0]):
            for h in range(np.shape(cropped_img)[1]):
                d2=(k-k_c)**2+(h-h_c)**2
                weighted_sum_var = weighted_sum_var + d2*cropped_img[k,h]
                image_sum = image_sum + cropped_img[k,h]
            
            
        var_d = weighted_sum_var/image_sum
    
        return var_d
		
	def nelder_mead (self, x0=np.zeros(43)):
		history = []
		
		def callback(x):
			fobj = self.f(x)
			history.append(fobj)
			
		result = minimize(self.f, x0, method='Nelder-Mead', tol=1e-6, callback=callback)
		
		print('Status: %s' %result['message'])
		print('Total Evaluations: %d' %result['nfev'])
		
		solution result['x']
		evaluation = result['fun']
		print('Solution: f(%s) = %.5f' %(solution, evaluation))
		#plot history
		plt.plot(history) 
		plt.show()
		
if __name__ == '__main__':
	a = NelderMead(None)
	
	a.nelder_mead(np.ones(2))