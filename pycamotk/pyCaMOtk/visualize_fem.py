import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np
from pyCaMOtk.refine_mesh_soln import refine_mesh_soln

def visualize_fem(msh, udg=[], ufem=[], opts={}, which_bnd=[]):
    porder = msh.porder
    nv, nelem = msh.e2vcg.shape
    nsd = msh.nsd
    ndim = msh.ndim
    xcg = msh.xcg
    e2vcg = msh.e2vcg
    e2bnd = msh.e2bnd
    f2v = msh.lfcnsp.geom.f2n
    nnode = xcg.shape[1]

    # 处理参数
    nref = opts.get('nref', 0)
    colorlimit = opts.get('climit', None)
    plot_nodes = opts.get('plot_nodes', False)
    plot_elem = opts.get('plot_elem', False)

    # 确定是否传入解
    plot_udg = not (len(udg) == 0)
    plot_ufem = not (len(ufem) == 0)
    plot_diff = plot_udg and plot_ufem

    # 处理udg和ufem的数据
    def process_data(data, label):
        if len(data) == 0:
            return None, None
        data_nelem = data.size
        if data_nelem == nelem:
            data = np.repeat(data.flatten(), nv).reshape(1, nv, nelem, order='F')
        elif data_nelem == nv * nelem:
            data = data.reshape(1, nv, nelem, order='F')
        elif data_nelem == nnode:
        # 节点型数据：通过e2vcg提取每个单元的顶点值
            data = data.flatten()[e2vcg].reshape(1, nv, nelem, order='F')
        else:
            raise ValueError(f'{label} has incorrect shape')
        if ndim < 3:
            if nref == 0:
                xdg_ref = xcg[:, e2vcg]
                data_ref = data
            else:
                xdg_ref, data_ref, _ = refine_mesh_soln(msh, data, nref)
            data_ref = data_ref.reshape(-1, data_ref.shape[-1], order='F')
            return xdg_ref, data_ref
        return None, None

    # 处理udg和ufem
    xdg_udg, udg_ref = process_data(udg, 'udg')
    xdg_ufem, ufem_ref = process_data(ufem, 'ufem')

    # 计算误差
    epsilon = 1e-10
    if plot_diff and udg_ref is not None and ufem_ref is not None:
        # 绝对误差
        abs_diff = udg_ref - ufem_ref
        
        # 相对误差
        denominator = np.abs(ufem_ref) + epsilon
        rel_diff = (udg_ref - ufem_ref) / denominator
    else:
        abs_diff = rel_diff = None

    # 创建图形和子图
    if plot_diff:
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        ((ax1, ax2), (ax3, ax4)) = axes
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
        axes = [ax1]
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xlim(0, 0.5)  # 横轴范围 [0, 1]
            ax.set_ylim(-1, 1)  # 纵轴范围 [-1, 1]
            ax.set_aspect(4)  # 设置横纵比为 2:1（横轴是纵轴的两倍）

    # 设置颜色范围
    # 前两个图的颜色范围
    if colorlimit is None:
        if plot_udg and plot_ufem:
            vmin = min(udg_ref.min(), ufem_ref.min())
            vmax = max(udg_ref.max(), ufem_ref.max())
        elif plot_udg:
            vmin, vmax = udg_ref.min(), udg_ref.max()
        elif plot_ufem:
            vmin, vmax = ufem_ref.min(), ufem_ref.max()
        else:
            vmin, vmax = None, None
    else:
        vmin, vmax = colorlimit

    # 误差图的颜色范围
    abs_vmin, abs_vmax = (abs_diff.min(), abs_diff.max()) if abs_diff is not None else (None, None)
    rel_vmin, rel_vmax = (rel_diff.min(), rel_diff.max()) if rel_diff is not None else (None, None)  # 相对误差通常用对称范围

    # 绘制函数
    def draw_subplot(ax, data, title, xdg_ref, vmin=None, vmax=None):
        if data is None:
            return None
        patches = []
        idx = [0, 1, 3, 2] if msh.etype == 'hcube' else msh.lfcnsp.geom.v2n[0, :].astype(int)
        for i in range(data.shape[1]):
            verts = xdg_ref[:, idx, i].T
            polygon = Polygon(verts, closed=True)
            patches.append(polygon)
        pc = PatchCollection(patches, cmap='jet', alpha=1)
        pc.set_array(data.flatten())
        pc.set_clim(vmin, vmax)
        ax.add_collection(pc)
        ax.set_xlim(xdg_ref[0].min(), xdg_ref[0].max())
        ax.set_ylim(xdg_ref[1].min(), xdg_ref[1].max())
        ax.set_aspect('equal')
        ax.set_title(title)
        return pc

    # 绘制四个子图
    pc1 = pc2 = pc3 = pc4 = None
    if plot_diff:
        pc1 = draw_subplot(ax1, udg_ref, 'Numerical Solution', xdg_udg, vmin, vmax)
        pc2 = draw_subplot(ax2, ufem_ref, 'Exact Solution', xdg_ufem, vmin, vmax)
        pc3 = draw_subplot(ax3, abs_diff, 'Absolute Error', xdg_udg, abs_vmin, abs_vmax)
        pc4 = draw_subplot(ax4, rel_diff, 'Relative Error', xdg_udg, rel_vmin, rel_vmax)

    # 添加颜色条
    cbars = []
    if plot_diff:
        # 前两个图共享颜色条
        #cbar1 = fig.colorbar(pc1, ax=[ax1, ax2])
        #cbar1.set_label('Solution Value')
        
        # # 绝对误差颜色条
        cbar2 = fig.colorbar(pc3, ax=ax3, location='right')
        cbar2.set_label('Absolute Error')
        
        # # 相对误差颜色条
        cbar3 = fig.colorbar(pc4, ax=ax4, location='right')
        cbar3.set_label('Relative Error')
        
        cbars.extend([ cbar2, cbar3])

    plt.tight_layout()
    return fig, axes, cbars