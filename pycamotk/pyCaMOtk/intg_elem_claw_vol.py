from __future__ import print_function
import numpy as np
import pdb
################################################################################
def intg_elem_claw_vol(Ue,transf_data,elem,elem_data,e):
	# Han Gao modify this for one more arg: e (element index)
	# Extract information from input : sizes
    

	# 输入参数：
	# Ue: element solution

	# transf_data: transformation data ?
	# elem: element object(create_element_strct) ?
	# elem_data: element data object(create_elem_data)

	# e: 单元索引，第e个单元

	[neqn_per_elem,neqn,ndimP1,nq]=elem.Tv_eqn_ref.shape
	# neqn_per_elem: 每个单元的基函数个数 = 9
	# nvar_per_elem: 每个单元的变量个数 = 9
	# neqn: 方程数 = 1
	# ndimP1: 空间维度+1 = 3
	# nq:  每个单元的积分点数 = 6
	[nvar_per_elem,nvar,_,_]=elem.Tv_var_ref.shape
	ndim=ndimP1-1
	wq=elem.wq # 积分权重
	detG=transf_data.detG[:,e] 
	Tvar=elem_data.Tv_var_phys[:,:,:,:,e].reshape([nvar_per_elem,nvar*(ndim+1)*nq],
												  order='F') # 物理空间的变量基函数
	Re=np.zeros([neqn_per_elem,1]) #初始化残差，维度是 9*1
	dRe=np.zeros([neqn_per_elem,nvar_per_elem]) #初始化残差对变量的导数，维度 9*9
	UQq=np.reshape(Tvar.T.dot(Ue),[nvar,ndim+1,nq],order='F') 
	w=wq*detG # 积分权重乘以雅各比行列式，进行了从【物理空间到参考空间的变换】
	for k in range(nq): 
		Teqn=elem_data.Tv_eqn_phys[:,:,:,k,e].reshape([neqn_per_elem,neqn*(ndim+1)],
			                                      order='F')
		Tvar=elem_data.Tv_var_phys[:,:,:,k,e].reshape([nvar_per_elem,nvar*(ndim+1)],
			                                      order='F')
		x=transf_data.xq[:,k,e] # 第k个积分点的物理坐标
		pars=elem_data.vol_pars[:,k,e] # 第k个积分点的参数
		SF, dSFdU=elem.eqn.srcflux(UQq[:,:,k],pars,x) #第k个积分点的源项和通量，以及对U的导数
		SF=SF.flatten(order='F') # 源项和通量展平
		dSFdU=np.reshape(dSFdU,[neqn*(ndim+1),nvar*(ndim+1)],order='F') # SF对U的导数
		Re=Re-w[k]*Teqn.dot(SF).reshape(Re.shape,order='F')
		dRe=dRe-w[k]*(Teqn.dot(dSFdU.dot(Tvar.T)))
	return Re, dRe
		


