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
		# 第一个维度（x方向）：均匀分割
    # 第一个维度（x方向）：均匀分割，节点数 = nel_x * porder + 1
		lim_min_x, lim_max_x = self.lims[0]
		nel_x = self.nel[0]
		nodes_x = np.linspace(lim_min_x, lim_max_x, nel_x * self.porder + 1)  # 修正节点数
		self.coords_sigdim.append(nodes_x)

		# 第二个维度（y方向）：每个x单元的y分界点独立生成
		lim_min_y, lim_max_y = self.lims[1]
		nel_y = self.nel[1]
		p = self.porder

		# 存储每个x单元对应的y坐标（形状: (nel_x, n_y_nodes_per_x_cell)）
		coords_y_per_x = []
		# 遍历每个x单元，生成对应的y分界点
		for x_idx in range(nel_x):
			# 生成随机分界点（确保两端固定）
			if nel_y > 1:
				total_length = lim_max_y - lim_min_y
				min_length =1.5* total_length / nel_y / 2  # 最小单元长度
				remaining = total_length - nel_y * min_length  # 剩余可分配长度
				
				# 生成随机增量（确保和为remaining）
				if remaining < 0:
					raise ValueError("无法满足最小单元长度，请减少nel_y或增大lim_max_y-lim_min_y")
				
				# 生成nel_y个区间的随机增量（Dirichlet分布保证和为1）
				rand_weights = np.random.dirichlet(np.ones(nel_y))  # 形状 (nel_y,)
				deltas = rand_weights * remaining  # 每个区间的额外长度
				
				# 构建严格递增的分界点
				nodes_y = [lim_min_y]
				current = lim_min_y
				for i in range(nel_y):
					current += min_length + deltas[i]
					nodes_y.append(current)
				nodes_y[-1] = lim_max_y  # 强制最后一个节点准确
				nodes_y = np.array(nodes_y)
			else:
				nodes_y = np.array([lim_min_y, lim_max_y])
			
			# 生成单元内部点（porder > 1时）
			coords_y = []
			for j in range(nel_y):
				start = nodes_y[j]
				end = nodes_y[j+1]
				
				if p > 1:
					# 在单元内生成p-1个随机点
					rand_offsets = np.sort(np.random.rand(p-1))
					internal_points = start + rand_offsets * (end - start)
				else:
					internal_points = np.array([])
				
				# 合并端点与内部点
				unit_points = np.concatenate([[start], internal_points, [end]])
				
				# 避免重复端点
				if j == 0:
					coords_y.extend(unit_points)
				else:
					coords_y.extend(unit_points[1:])
			
			coords_y_per_x.append(np.array(coords_y))  # 将每个x单元的y分界点添加到列表中

		# 存储每个x单元对应的y分界点
		self.coords_sigdim.append(np.vstack(coords_y_per_x))
		#添加到coords_sigdim中，形成一个列表
		self.nnodeperdim=[self.nel[i]*self.porder+1 for i in range(self.ndim)] # 每个维度的节点数，每个维度的网格数*多项式的阶数+1
		self.nnode=np.prod(np.asarray(self.nnodeperdim)) # 总的节点数，每个维度的节点数相乘
		self.nnodeperelem=(self.porder+1)**self.ndim # 每个单元的节点数，多项式的阶数+1的dimension次方 （2+1）^2=9
		self.nnodeperdim = [
				nel_x * self.porder + 1,
				len(coords_y_per_x[0])  # 假设所有x单元的y节点数相同
			]
		self.nnode = np.prod(np.asarray(self.nnodeperdim))

		# 生成全局坐标矩阵 self.xcg（形状: (ndim, nnode)）
		self.xcg = np.zeros((self.ndim, self.nnode))
		# 修改后的坐标赋值逻辑
		for i in range(self.nnode):
			mltidx = mltidx_from_linidx(self.nnodeperdim, i)
			# 第一个维度直接索引
			self.xcg[0, i] = self.coords_sigdim[0][mltidx[0]]
			# 第二个维度根据x单元索引调整
			x_cell_idx = mltidx[0] // self.porder  # 初始计算
			x_cell_idx = min(x_cell_idx, self.nel[0] - 1)  # 确保不超过nel_x - 1
			y_local_idx = mltidx[1] % (self.nel[1] * self.porder + 1)  # 局部索引
			self.xcg[1, i] = self.coords_sigdim[1][x_cell_idx, y_local_idx]

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
		