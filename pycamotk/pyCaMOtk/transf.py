import pdb
import numpy as np

class ElementTransformation(object):
    def __init__(self):
        """
        xe - nodal positions
        xq, detG, Gi in volume  
        xqf, sigf, Gi, n on faces
        xqd on edges
        """
        self.xe = None
        self.xq = None # 实际积分点坐标

    def eval_transf_quant(self, xe, lfcnsp, zv=None, zf=None, zd=None):
        self.lfcnsp = lfcnsp
        self.xe = xe # 实际节点坐标
        
        
        # Start extracting info
        # if z is None, use quadrature nodes inside lfcnsp
        # Element nodes from physical domain
        #  Extract information from xe
        # Need to check 1d if this works
        self.ndim=xe.shape[0]              # dimension
        self.nxpe=xe.shape[1]              # number of node per elem
        self.nfpe=lfcnsp.geom.f2n.shape[1] # number of face
        self.nxpf=lfcnsp.geom.f2n.shape[0] # number of face node
        self.ndpe=lfcnsp.geom.d2n.shape[1] # number of edge
        self.nxpd=lfcnsp.geom.d2n.shape[0] # number of edge node
        self.nqpe=lfcnsp.zqv.shape[1]      #number of quad node in element
        self.nqpf=lfcnsp.rq.shape[1]       # number of quad node per face
        self.nqpd=lfcnsp.nq0
        
        # Start volume
        # Generate xq (the quadrature nodes in ref domain 
        #             mapped to physical domain)
        #pdb.set_trace()
        self.xq = np.matmul(self.xe, self.lfcnsp.Qvv[:,0,:]) #计算实际积分点坐标
        # Generate detG
        self.detG = np.zeros((self.nqpe,1))
        for i in range(self.nqpe): #对单元中的每个积分点
            self.detG[i,0] = np.linalg.det(np.matmul(self.xe, self.lfcnsp.Qvv[:,1:1+self.ndim,i])) #计算雅克比行列式，存的是导数
        # Generate Gi (the inverse of the deformation gradient of the isopa
        #              -parametric mapping, evaluated at each quadrature node)     
        self.Gi = np.zeros((self.ndim, self.ndim, self.nqpe))
        for i in range(self.nqpe):
            self.Gi[:,:,i] = np.linalg.inv(np.matmul(self.xe, self.lfcnsp.Qvv[:,1:1+self.ndim,i]))


                # Start face
                # Start face
        self.xqf = np.zeros((self.ndim, self.nqpf, self.nfpe))
        for i in range(self.nfpe):
            self.xqf[:, :, i] = np.matmul(self.xe, self.lfcnsp.Qvf[:, 0, :, i])

        self.sigf = np.zeros((self.nqpf, self.nfpe))
        self.Gif = np.zeros((self.ndim, self.ndim, self.nqpf, self.nfpe))

        for i in range(self.nfpe):
            for j in range(self.nqpf):
                xef = self.xe[:, self.lfcnsp.geom.f2n[:, i].astype('int')]
                Gf = np.matmul(self.xe, self.lfcnsp.Qvf[:, 1:1+self.ndim, j, i])
        
                # 修复：仅在 ndim > 1 时计算面导数
                if self.ndim > 1:
                    Ff = np.matmul(xef, self.lfcnsp.Qf[:, 1:1+self.ndim-1, j])
                    self.sigf[j, i] = np.sqrt(np.linalg.det(Ff.T.dot(Ff)))
                else:
                    # 1D情况：面为端点，sigf设为1.0（或根据实际意义调整）
                    self.sigf[j, i] = 1.0
        
                self.Gif[:, :, j, i] = np.linalg.inv(Gf)
                #self.sigf[j,i] = np.linalg.det(np.matmul(self.xe, self.lfcnsp.Qvf[:,1:1+self.ndim,j,i]))
        #for i in range(self.nfpe):
        #    for j in range(self.nqpf):
        #        self.Gif[:,:,j,i] = np.linalg.inv(np.matmul(self.xe, self.lfcnsp.Qvf[:,1:1+self.ndim,j,i]))
        
        # Start edge
        self.xqd = np.zeros((self.ndim, self.nqpd, self.ndpe))
        for i in range(self.ndpe):
            self.xqd[:,:,i] = np.matmul(self.xe, self.lfcnsp.Qvd[:,0,:,i])

        # Start point normal (Here we focus on points on face instead quad points?)
        self.n=np.zeros((self.ndim,self.nqpf,self.nfpe))
        for f in range(self.nfpe):
            for k in range(self.nqpf):
                n_=self.Gif[:, :, k, f].T.dot(self.lfcnsp.geom.N[:,f])
                #pdb.set_trace()
                self.n[:, k, f]=n_/np.linalg.norm(n_)
        #pdb.set_trace()
        pass