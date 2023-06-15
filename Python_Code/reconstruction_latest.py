import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import center_of_mass
from zernike import RZern
#from scipy.special import zernike


#img_original=cv2.imread(r"C:\Users\ASUS\Desktop\adaptive_optics\Project\Section5\get_slope\original.jpg")
#img_change1 = cv2.imread(r"C:\Users\ASUS\Desktop\adaptive_optics\Project\Section5\get_slope\change_1.jpg")
#img_change2 = cv2.imread(r"C:\Users\ASUS\Desktop\adaptive_optics\Project\Section5\get_slope\change_2.jpg")

img_original=cv2.imread(r"C:\Users\ASUS\Desktop\adaptive_optics\Project\Section5\get_slope\61reference.png")
img_change1 = cv2.imread(r"C:\Users\ASUS\Desktop\adaptive_optics\Project\Section5\get_slope\61change1.png")
img_change2 = cv2.imread(r"C:\Users\ASUS\Desktop\adaptive_optics\Project\Section5\get_slope\61change2.png")
#print(np.shape(img_original))
#Cut the picture
img_original=img_original[10:1300,(1280-800):(1280+700),:]
img_change1=img_change1[10:1300,(1280-800):(1280+700),:]
img_change2=img_change2[10:1300,(1280-800):(1280+700),:]


def create_reference_fixed_grid(img):
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #find threshod
    retv,thresh=cv2.threshold(img_gray,50,255,cv2.THRESH_BINARY)
    #find contour
    contours,hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
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
    image_d=image
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
        
        draw_1=cv2.rectangle(image_d, (center_x-width,center_y-width),(center_x+width, center_y+width), (255,255,255), 2)
        draw_1=cv2.circle(draw_1, (int(abs_centriod[i,0]), int(abs_centriod[i,1])), 5, (255,255,255))
        draw_1=cv2.circle(draw_1, (center_x, center_y), 5, (0,255,255))
    new_draw = cv2.resize(draw_1, None, fx = 0.75, fy = 0.75)
    cv2.imshow('grid',new_draw) 
    cv2.waitKey()
    cv2.destroyAllWindows()
    # Calculate deviation from reference center
    deviations = abs_centriod-reference_centers 
    return deviations

refer_points=create_reference_fixed_grid(img_original)

D1=get_slopes(img_original,refer_points)
D2=get_slopes(img_change1, refer_points)-D1
D3=get_slopes(img_change2, refer_points)-D1
D_test=np.zeros(np.shape(D3))
D_test[:,0]=1

# B Matrix
def normalize_coordinates(points):
    # Calculate the center of the points
    center = np.mean(points, axis=0)

    # Calculate the maximum distance from the center to any point
    max_distance = np.max(np.linalg.norm(points - center, axis=1))

    # Normalize the coordinates
    normalized_points = (points - center) / max_distance

    return normalized_points

normalized_pos=normalize_coordinates(refer_points)

def get_Bmatrix(points,Zorder):
    cart = RZern(Zorder)
    L, K = 1000, 1000
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
        prod[rho > 1.0] = 0
        cart.ZZ[:, k] = cart.vect(prod)

    B=np.zeros((2*points.shape[0],cart.nk))
    for i in range(cart.nk):
        #Define coefficient
        c = np.zeros((cart.nk,1))
        c[i] = 1
        Phi = cart.eval_grid(c, matrix=True)
        #Unncomment code below to check Zernike polynomial!
        #cv2.imshow('Phi',Phi) 
        #cv2.waitKey()
        #cv2.destroyAllWindows()
        for j in range(points.shape[0]):
            #find where is point in pixel
            pixel_x=int(points[j,0]*L/2)
            pixel_y=int(points[j,1]*K/2)
            #Gradient of y in certain point
            B[2*j+1,i]=(Phi[pixel_y+1,pixel_x]-Phi[pixel_y-1,pixel_x])/2
            #Gradient of x in certain point
            B[2*j,i]=(Phi[pixel_y,pixel_x+1]-Phi[pixel_y,pixel_x-1])/2
    return B,cart

B,cart=get_Bmatrix(normalized_pos,7)

#Find coeffecient
invB=np.linalg.pinv(B)
coef=np.dot(invB,D2.reshape((-1,1)))
Phi = cart.eval_grid(coef, matrix=True)

ax3 = plt.axes(projection='3d')
xx = np.linspace(-1.0, 1.0, 1000)
yy = np.linspace(-1.0, 1.0, 1000)
X, Y = np.meshgrid(xx, yy)
ax3.plot_surface(X,Y,Phi,cmap='rainbow')
#plt.imshow(Phi, origin='lower', extent=(-1, 1, -1, 1))
plt.show()





    






