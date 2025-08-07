import matplotlib.pyplot as plt
from pathlib import Path

# 数据
x = list(range(1, 11))
y = [18.62, 20.55, 25.53, 27.54, 24.51, 30.32, 28.15, 29.13, 33.51, 32.89]

# 绘图
plt.figure()
plt.plot(x, y, marker='o', label='Data')
plt.axhline(12, linestyle='--', label='basemodel (12)')
plt.axhline(40, linestyle='--', label='TTRL (40)')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Data with Basemodel and TTRL Lines')
plt.legend()
plt.tight_layout()

# 目标路径
out_dir = Path("/home/jovyan/ICRL/results/icrl")
out_dir.mkdir(parents=True, exist_ok=True)          # 若目录不存在则创建
out_path = out_dir / "data_plot.png"

# 保存
plt.savefig(out_path)
print(f"Figure saved to: {out_path}")
