import numpy as np
import pdb
from pyCaMOtk.ndist_mltdim import ndist_mltdim_hcube
from pyCaMOtk.tens_core import mltidx_from_linidx
from pyCaMOtk.geom_mltdim import Hypercube
from pyCaMOtk.mesh import Mesh
################################################################################
class mesh_hcube(object):
	"""docstring for mesh_hcube"""
	def __init__(self,etype,lims,nel,porder,**varargin):
		self.etype=etype # 单元类型
		self.lims=lims # 网格的边界
		self.nel=np.asarray(nel) # 每个维度的网格数初始化
		self.nel=self.nel[:] # 每个维度的网格数赋值
		self.porder=porder # 多项式的阶数
		self.varargin=varargin 

		self.ndim=np.max(self.nel.shape)
		self.nf=2*self.ndim
		self.nelem=np.prod(self.nel)
		self.coords_sigdim=[]
		for i in range(self.ndim): #遍历每个维度，生成坐标，均匀分布
			coord_sigmdim_=np.linspace(self.lims[i,0], 
				                       self.lims[i,1], 
				                       self.nel[i]*self.porder+1) # 网格数*多项式的阶数+1
			self.coords_sigdim.append(coord_sigmdim_) # 将每个维度的坐标添加到coords_sigdim中，形成一个列表
		self.nnodeperdim=[self.nel[i]*self.porder+1 for i in range(self.ndim)] # 每个维度的节点数，每个维度的网格数*多项式的阶数+1
		self.nnode=np.prod(np.asarray(self.nnodeperdim)) # 总的节点数，每个维度的节点数相乘
		self.nnodeperelem=(self.porder+1)**self.ndim # 每个单元的节点数，多项式的阶数+1的dimension次方 （2+1）^2=9
		self.xcg=np.zeros((self.ndim,self.nnode)) 
		for i in range(self.nnode): # 遍历每个节点
			mltidx=mltidx_from_linidx(self.nnodeperdim,i) # 通过节点的全局索引，获取多维索引
			for j in range(self.ndim): 
				self.xcg[j,i]=self.coords_sigdim[j][mltidx[j]]  # 通过多维索引，获取坐标，所以xcg是一个多维数组，存的是每个节点的坐标

		# if self.ndim==1: 
		# 	self.M=range(self.nnode) 
		# else:
		# 	self.M=np.reshape(range(self.nnode),self.nnodeperdim,order='F'); # 重整形成多维数组（方形），M就是节点的全局索引的多维数组
		# 修改 self.M 的定义
		if self.ndim == 1:
			self.M = np.arange(self.nnode)  # 使用 NumPy 数组
		else:
			self.M = np.reshape(np.arange(self.nnode), self.nnodeperdim, order='F')


		self.idx_start=[] 
		self.idx_offset=[] 
		for k in range(self.ndim):
			self.idx_start.append([i*self.porder for i in range(self.nel[k])]) # 每个维度的起始索引，计算方式是每个维度网格数*多项式的阶数，
			self.idx_offset.append([i for i in range(self.porder+1)]) # 每个维度的偏移索引，计算方式是多项式的阶数+1，比如第一个维度是0，1，2，

		self.strt=np.zeros(self.nelem) 
		self.off=np.zeros(self.nnodeperelem) 
		for n in range(self.nelem):
			mltidx1=mltidx_from_linidx(self.nel,n)
			mltidx2=[]
			for d in range(self.ndim):
				mltidx2.append(self.idx_start[d][mltidx1[d]])
			    # 一维时直接使用标量索引
			if self.ndim == 1:
				index = mltidx2[0]
			else:
				index = tuple(mltidx2)
			self.strt[n]=self.M[index]  # 存每个单元的起始节点
		self.strt=np.sort(self.strt) # 排序

		for n in range(self.nnodeperelem):
			mltidx1=mltidx_from_linidx([self.porder+1 for i in range(self.ndim)],n)
			mltidx2=[]
			for d in range(self.ndim):
				#pdb.set_trace()
				mltidx2.append(self.idx_offset[d][(mltidx1[d])])
			if self.ndim == 1:
				index = mltidx2[0]
			else:
				index = tuple(mltidx2)
			self.off[n]=self.M[index]
		self.off=np.sort(self.off)

		self.e2vcg=np.zeros((self.nnodeperelem,self.nelem)) 
		for e in range(self.nelem): #
			self.e2vcg[:,e]=self.strt[e]+self.off #
		self.e2vcg=self.e2vcg.astype('int') #e2vcg是每个单元的节点的全局索引，形状是（每个单元的节点数 x 单元数）


		self.e2bnd=np.zeros([2*self.ndim,self.nelem])+np.nan 
		self.refhcubeelem=Hypercube(self.ndim,self.porder,'unif')
		self.f2v=self.refhcubeelem.f2n
		for e in range(self.nelem):
			for f in range(self.nf):
				face_nodes=self.xcg[:,self.e2vcg[self.f2v[:,f],e]]
				for d in range(self.ndim):
					if float(np.linalg.norm(face_nodes[d,:]-self.lims[d,0]))==float(0): # 判断是否在边界上,如果在边界上，就赋值，否则就跳过
						self.e2bnd[f,e]=d
					elif float(np.linalg.norm(face_nodes[d,:]-self.lims[d,1]))==float(0):
						self.e2bnd[f,e]=self.ndim+d 
					else:
						pass
		if len(self.varargin)==0: 
			self.msh=Mesh(self.etype,self.xcg, self.e2vcg,self.e2bnd)

	def getmsh(self):
		return self.msh
		
		




		




		
#class ClassName(object):
#	"""docstring for ClassName"""
#	def __init__(self, arg):
#		super(ClassName, self).__init__()
#		self.arg = arg
		