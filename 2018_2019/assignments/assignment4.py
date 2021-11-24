import ps_utils as pu
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy 
from scipy import sparse
from sklearn.preprocessing import normalize

Beethoven_raw_dataset = pu.read_data_file("Beethoven.mat")
Beet_I = Beethoven_raw_dataset[0]
Beet_mask = Beethoven_raw_dataset[1]
Beet_S = Beethoven_raw_dataset[2]

Buddha_raw_dataset = pu.read_data_file("Buddha.mat")
Budd_I = Buddha_raw_dataset[0]
Budd_mask = Buddha_raw_dataset[1]
Budd_S = Buddha_raw_dataset[2]

	# len(Beet_S) = len(Beet_I[m][n])

def make_normals(I,mask,S):

	n1 = np.zeros(mask.shape)
	n2 = np.zeros(mask.shape)
	n3 = np.zeros(mask.shape)

	S_inv = np.linalg.pinv(S)

	for row in xrange(0,len(I)): # For each n of image size (m,n)
		for col in xrange(0,len(I[row])): # For each m of image size (m,n)
			if mask[row][col]:

				m_uv = np.dot(S_inv,I[row][col]) 
				p_uv = np.linalg.norm(m_uv) # p_uv = |m_uv|
				n_uv = np.dot(np.reciprocal(p_uv), m_uv)

				n1[row][col] = n_uv[0]
				n2[row][col] = n_uv[1]
				n3[row][col] = n_uv[2]

	return np.array([n1, n2, n3]) # Returns (m,n) size lists of each of the 3 indexes in the normal vectors

def remove_zeros(vectorelement): # Help function to manipulate the data returned from make_normals

	v_elements = pu.tolist(vectorelement)

	zeroless = np.zeros(np.count_nonzero(v_elements))

	counter = 0

	for x in xrange(0,len(v_elements)):
		if v_elements[x] == 0.0:
			continue
		else: 
			zeroless[counter] = v_elements[x]
			counter += 1

	return zeroless # Returns the vectorelement with all 0 indexes removed. Also makes it a list.

def albedo_modulated_field(S,normals):

	nz = np.count_nonzero(normals[0])

	n1 = remove_zeros(normals[0])
	n2 = remove_zeros(normals[1])
	n3 = remove_zeros(normals[2])

	J = np.array([n1, n2, n3])
	#print normals.shape
	#print S.shape
	#print J.shape
	M = np.dot(np.linalg.pinv(S),J)

	return M # Returns a (3,nz) array

def extract_albedo(mask,albedo):

	n1 = np.zeros(mask.shape)
	n2 = np.zeros(mask.shape)
	n3 = np.zeros(mask.shape)

	counter = 0

	for row in xrange(0,len(mask)):
		for col in xrange(0,len(mask[row])):
			if mask[row][col]:
				n1[row][col] = albedo[0][counter]
				n2[row][col] = albedo[1][counter]
				n3[row][col] = albedo[2][counter]

				counter += 1

	return np.array([n1, n2, n3])
	
def beethoven_main():
	
	Beet_normals = make_normals(Beet_I,Beet_mask,Beet_S)
	Beet_n1 = Beet_normals[0]
	Beet_n2 = Beet_normals[1]
	Beet_n3 = Beet_normals[2]
	depth_map = pu.unbiased_integrate(Beet_n1,Beet_n2,Beet_n3,Beet_mask)
	pu.display_image(Beet_n1)
	pu.display_image(Beet_n2)
	pu.display_image(Beet_n3)
	#pu.display_depth(depth_map)

	Beet_M = albedo_modulated_field(Beet_S,Beet_normals)
	Beet_extracted_albedo = extract_albedo(Beet_mask,Beet_M)
	Beet_ea1 = Beet_extracted_albedo[0]
	Beet_ea2 = Beet_extracted_albedo[1]
	Beet_ea3 = Beet_extracted_albedo[2]
	pu.display_image(Beet_ea1)
	pu.display_image(Beet_ea2)
	pu.display_image(Beet_ea3)

	Beet_M_norm = normalize(Beet_M)
	Beet_ea_norm = extract_albedo(Beet_mask,Beet_M_norm)
	Beet_norm1 = Beet_ea_norm[0]
	Beet_norm2 = Beet_ea_norm[1]
	Beet_norm3 = Beet_ea_norm[2]
	depth_map_norm = pu.unbiased_integrate(Beet_norm1,Beet_norm2,Beet_norm3,Beet_mask)
	pu.display_depth(depth_map_norm)

def buddha_main():
	
	Budd_normals = make_normals(Budd_I,Budd_mask,Budd_S)
	Budd_n1 = Budd_normals[0]
	Budd_n2 = Budd_normals[1]
	Budd_n3 = Budd_normals[2]
	depth_map = pu.unbiased_integrate(Budd_n1,Budd_n2,Budd_n3,Budd_mask)
	pu.display_image(Budd_n1)
	pu.display_image(Budd_n2)
	pu.display_image(Budd_n3)
	#pu.display_depth(depth_map)
	

		# Albedo Modulated Field doesn't work properly for size (10,nz)
	"""Budd_M = albedo_modulated_field(Budd_S,Budd_normals) 
	Budd_extracted_albedo = extract_albedo(Budd_mask,Budd_M)
	Budd_ea1 = Budd_extracted_albedo[0]
	Budd_ea2 = Budd_extracted_albedo[1]
	Budd_ea3 = Budd_extracted_albedo[2]
	#pu.display_image(Budd_ea1)
	#pu.display_image(Budd_ea2)
	#pu.display_image(Budd_ea3)

	Budd_M_norm = normalize(Budd_M)
	Budd_ea_norm = extract_albedo(Budd_mask,Budd_M_norm)
	Budd_norm1 = Budd_ea_norm[0]
	Budd_norm2 = Budd_ea_norm[1]
	Budd_norm3 = Budd_ea_norm[2]
	depth_map_norm = pu.unbiased_integrate(Budd_norm1,Budd_norm2,Budd_norm3,Budd_mask)
	#pu.display_depth(depth_map_norm)"""


beethoven_main() # Calculates images in 2d and 3d for Beethoven. 3d images are commented out due to runtime
#buddha_main() # Calculates images in 2d for Buddha.
