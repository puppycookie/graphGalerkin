import numpy as np
import pdb

class create_ldof2gdof_dg(object):
	"""docstring for create_ldof2gdof_cg"""
	def __init__(self,ndof_per_node,nnode_per_elem,nelem,e2vcg):
		self.ndof_per_node=ndof_per_node
		self.nnode_per_elem=nnode_per_elem
		self.nelem=nelem
		self.e2vcg=e2vcg

		self.ndoftotal=self.ndof_per_node*self.nnode_per_elem*self.nelem
		shape=(self.ndof_per_node*self.nnode_per_elem,self.nelem)
		self.ldof2gdof=np.asarray(range(self.ndoftotal)).reshape(shape,order='F')
		for e in range(self.nelem):
			gdof=np.matlib.repmat(self.e2vcg[:, e].T*self.ndof_per_node,
				                  self.ndof_per_node,1)+\
				 np.asarray(range(self.ndof_per_node)).reshape(-1,1)
			self.ldof2gdof[:, e]=gdof.reshape(self.ndof_per_node*self.nnode_per_elem,order='F')
		self.ldof2gdof=self.ldof2gdof.astype('int')
	
	
	# 不连续单元自由度
#	def create_mapping(self):
#		"""Create local to global degree of freedom mapping"""
#		for e in range(self.nelem):
#            # For DG, each element has its own independent degrees of freedom
#			gdof = np.arange(e * self.ndof_per_node * self.nnode_per_elem,
#                             (e + 1) * self.ndof_per_node * self.nnode_per_elem)
#			self.ldof2gdof[:, e] = gdof	