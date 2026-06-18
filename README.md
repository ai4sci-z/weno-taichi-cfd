# weno-taichi-cfd

基于 Python / Taichi 的高精度 WENO 格式流体不稳定性数值模拟。

本项目源自本科毕业设计：相对论流体对流与不稳定性问题的初步研究。
现整理为可复现的独立模块，持续扩展。

## 已实现

| 模块 | 文件 | 说明 |
|------|------|------|
| 1D WENO5 对流方程 | `weno1d.py` | 线性对流，WENO5 空间离散 + RK3 时间推进 |
| 1D Burgers 方程 | `burgers1d.py` | 非线性对流，激波捕捉，误差收敛分析 |
| Rayleigh-Taylor 不稳定性 | `rt_instability.py` | 2D 密度界面不稳定性，Taichi GPU 加速 |
| Kelvin-Helmholtz 不稳定性 | `kh_instability.py` | 2D 剪切层涡卷，Taichi GPU 加速 |

## 快速开始

```bash
pip install numpy matplotlib taichi
python burgers1d.py        # 1D Burgers 激波演化 + 误差分析
python rt_instability.py   # Rayleigh-Taylor 不稳定性可视化
```

## 数值方法

- 空间离散：5 阶 WENO（Jiang & Shu 1996）
- 时间推进：3 阶 TVD Runge-Kutta
- Lax-Friedrichs 数值通量分裂

## 参考文献

- Jiang, G.-S. & Shu, C.-W. (1996). Efficient implementation of weighted ENO schemes. *J. Comput. Phys.* 126, 202–228.
- Shu, C.-W. (2009). High order weighted essentially nonoscillatory schemes for convection dominated problems. *SIAM Rev.* 51(1), 82–126.
