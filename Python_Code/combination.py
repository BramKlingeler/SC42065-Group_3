import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import zernike

img_original=cv2.imread(r'C:\Users\ASUS\Desktop\adaptive_optics\Project\Section5\ex1.png')
#print(img_original)
def create_reference_fixed_grid(img):
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #find threshod
    retv,thresh=cv2.threshold(img_gray,50,255,cv2.THRESH_BINARY)
    #find contour
    contours,hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #draw contour
    cv2.drawContours(img_original,contours,-1,(0,0,255),3,lineType=cv2.LINE_AA)
    #show image
    #cv2.imshow('Contours',img_original)
    #cv2.waitKey()
    #cv2.destroyAllWindows()
    points = []
    for contour in contours:
        # Calculate the moments of the contour
        moments = cv2.moments(contour)   
        # Calculate the centroid coordinates
        center_x = int(moments['m10'] / moments['m00'])
        center_y = int(moments['m01'] / moments['m00'])   
        # Add the point to the list
        points.append((center_x, center_y))
    return points
    #print(len(points))

def get_slopes(reference_centers, measured_centers):
    # Draw grid around reference centers
    grid_size = reference_centers[1, 0] - reference_centers[0, 0]
    grid_rows = int((np.max(reference_centers[:, 1]) - np.min(reference_centers[:, 1])) / grid_size) + 1
    grid_cols = int((np.max(reference_centers[:, 0]) - np.min(reference_centers[:, 0])) / grid_size) + 1

    # Calculate center of mass for each grid position
    grid_sums = np.zeros((grid_rows, grid_cols))
    grid_counts = np.zeros((grid_rows, grid_cols))

    for measured_center in measured_centers:
        row = int((measured_center[1] - np.min(reference_centers[:, 1])) / grid_size)
        col = int((measured_center[0] - np.min(reference_centers[:, 0])) / grid_size)
        grid_sums[row, col] += measured_center[0]  # Assuming x-coordinate is needed
        grid_counts[row, col] += 1

    grid_centers = grid_sums / grid_counts

    # Calculate deviation from reference center
    deviations = grid_centers - reference_centers

    return deviations

# B Matrix



def calculate_gradient(reference_positions, delta_x, delta_y, mode):
    radial_order, azimuthal_order = zernike.noll_sequence(mode)
    rho, phi = convert_to_polar(reference_positions)

    z_reference = zernike1(rho, phi, radial_order, azimuthal_order)
    z_reference_dx = zernike1(rho + delta_x, phi, radial_order, azimuthal_order)
    x_gradient = (z_reference_dx - z_reference) / delta_x

    z_reference_dy = zernike1(rho, phi + delta_y, radial_order, azimuthal_order)
    y_gradient = (z_reference_dy - z_reference) / delta_y

    return x_gradient, y_gradient

def form_b_matrix(reference_positions, delta_x, delta_y, mode_range):
    grid_positions = convert_to_unit_grid(reference_positions)

    num_modes = len(mode_range)
    num_positions = len(reference_positions)
    b_matrix = np.zeros((num_modes, num_positions))

    for i, mode in enumerate(mode_range):
        for j, reference_position in enumerate(grid_positions):
            x_gradient, y_gradient = calculate_gradient(reference_position, delta_x, delta_y, mode)
            b_matrix[i, j] = x_gradient

    return b_matrix


refer_points=create_reference_fixed_grid(img_original)
aber_points=create_reference_fixed_grid(img_original)
deviations=get_slopes(refer_points, aber_points)
print(deviations)
#print(points)
#print(len(points))


    






