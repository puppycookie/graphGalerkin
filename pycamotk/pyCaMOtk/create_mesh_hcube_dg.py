import numpy as np
from pyCaMOtk.ndist_mltdim import ndist_mltdim_hcube
from pyCaMOtk.tens_core import mltidx_from_linidx
from pyCaMOtk.geom_mltdim import Hypercube
from pyCaMOtk.meshdg import Mesh

class mesh_hcube(object):
    """docstring for mesh_hcube"""
    def __init__(self, etype, lims, nel, porder, **varargin):
        self.etype = etype  # 单元类型
        self.lims = lims    # 网格边界
        self.nel = np.asarray(nel)  # 各维度单元数
        self.porder = porder  # 多项式阶数
        self.varargin = varargin

        self.ndim = self.nel.size  # 维度
        self.nf = 2 * self.ndim    # 每个单元的边界面数
        self.nelem = np.prod(self.nel)  # 总单元数
        self.coords_sigdim = []

        # 生成每个维度的坐标（DG格式，每个单元独立节点）
        for i in range(self.ndim):
            h = (self.lims[i, 1] - self.lims[i, 0]) / self.nel[i]
            coords = []
            for e in range(self.nel[i]):
                start = self.lims[i, 0] + e * h
                end = start + h
                coords_e = np.linspace(start, end, self.porder + 1)
                coords.extend(coords_e)
            self.coords_sigdim.append(np.array(coords))

        self.nnodeperdim = [self.nel[i] * (self.porder + 1) for i in range(self.ndim)]  # 各维度节点数
        self.nnode = np.prod(self.nnodeperdim)  # 总节点数
        self.nnodeperelem = (self.porder + 1) ** self.ndim  # 每个单元的节点数

        # 生成全局节点坐标
        self.xcg = np.zeros((self.ndim, self.nnode))
        for i in range(self.nnode):
            mltidx = mltidx_from_linidx(self.nnodeperdim, i)
            for j in range(self.ndim):
                self.xcg[j, i] = self.coords_sigdim[j][mltidx[j]]

        # 构建节点索引的多维数组（列优先）
        if self.ndim == 1:
            self.M = np.arange(self.nnode)
        else:
            self.M = np.reshape(np.arange(self.nnode), self.nnodeperdim, order='F')

        # 初始化每个单元的起始索引和偏移量
        self.idx_start = []
        self.idx_offset = []
        for k in range(self.ndim):
            # DG中每个单元的起始索引为 i*(porder+1)
            self.idx_start.append([i * (self.porder + 1) for i in range(self.nel[k])])
            self.idx_offset.append([i for i in range(self.porder + 1)])

        # 计算每个单元的起始节点索引
        self.strt = np.zeros(self.nelem, dtype=int)
        for n in range(self.nelem):
            mltidx_e = mltidx_from_linidx(self.nel, n)
            mltidx_start = [self.idx_start[d][mltidx_e[d]] for d in range(self.ndim)]
            self.strt[n] = self.M[tuple(mltidx_start)]
        self.strt = np.sort(self.strt)

        # 计算参考单元内节点的偏移量
        self.off = np.zeros(self.nnodeperelem, dtype=int)
        for n in range(self.nnodeperelem):
            mltidx_off = mltidx_from_linidx([self.porder + 1] * self.ndim, n)
            mltidx = [self.idx_offset[d][mltidx_off[d]] for d in range(self.ndim)]
            self.off[n] = self.M[tuple(mltidx)]
        self.off = np.sort(self.off)

        # 构建单元到节点的连接关系
        self.e2vcg = np.zeros((self.nnodeperelem, self.nelem), dtype=int)
        for e in range(self.nelem):
            self.e2vcg[:, e] = self.strt[e] + self.off

        self.strt0=np.zeros(self.nelem) 
        self.off0=np.zeros(self.nnodeperelem) 

        self.idx_start0=[] 
        self.idx_offset0=[] 
        for k in range(self.ndim):
            self.idx_start0.append([i*self.porder for i in range(self.nel[k])]) # 每个维度的起始索引，计算方式是每个维度网格数*多项式的阶数，
            self.idx_offset0.append([i for i in range(self.porder+1)]) # 每个维度的偏移索引，计算方式是多项式的阶数+1，比如第一个维度是0，1，2，

        for n in range(self.nelem):
            mltidx1=mltidx_from_linidx(self.nel,n)
            mltidx2=[]
            for d in range(self.ndim):
                mltidx2.append(self.idx_start0[d][mltidx1[d]])
            self.strt0[n]=self.M[tuple(mltidx2)]  # 存每个单元的起始节点
        self.strt0=np.sort(self.strt0) # 排序

        for n in range(self.nnodeperelem):
            mltidx1=mltidx_from_linidx([self.porder+1 for i in range(self.ndim)],n)
            mltidx2=[]
            for d in range(self.ndim):
                #pdb.set_trace()
                mltidx2.append(self.idx_offset0[d][(mltidx1[d])])
            self.off0[n]=self.M[tuple(mltidx2)]
        self.off0=np.sort(self.off0)

        self.e2vcg0=np.zeros((self.nnodeperelem,self.nelem)) 
        for e in range(self.nelem): #
            self.e2vcg0[:,e]=self.strt0[e]+self.off0 #
        self.e2vcg0=self.e2vcg0.astype('int') #e2vcg是每个单元的节点的全局索引，形状是（每个单元的节点数 x 单元数）


        # 处理边界标志
        self.e2bnd = np.zeros((2 * self.ndim, self.nelem)) + np.nan
        self.refhcubeelem = Hypercube(self.ndim, self.porder, 'unif')
        self.f2v = self.refhcubeelem.f2n
        for e in range(self.nelem):
            for f in range(self.nf):
                face_nodes = self.xcg[:, self.e2vcg[self.f2v[:, f], e]]
                for d in range(self.ndim):
                    if np.allclose(face_nodes[d, :], self.lims[d, 0]):
                        self.e2bnd[f, e] = d
                    elif np.allclose(face_nodes[d, :], self.lims[d, 1]):
                        self.e2bnd[f, e] = self.ndim + d

        # 创建Mesh对象
        self.msh = Mesh(self.etype, self.xcg, self.e2vcg,self.e2vcg0, self.e2bnd)

    def getmsh(self):
        return self.msh