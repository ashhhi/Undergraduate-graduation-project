import os
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

rootDIR = r'C:\DataSet\DeepGlobal\output_ResUnet_CBAM'
pred_path = os.path.join(rootDIR, 'pred/')
gt_path = os.path.join(rootDIR, 'gt/')
# img_path = os.path.join(rootDIR, 'img/')

img_path = r'C:\DataSet\DeepGlobal\output\img'
start = 1000

pred = []
gt = []
img = []
for i in range(start,start+5):
    pred.append(os.path.join(pred_path,str(i)+'.png'))
    gt.append(os.path.join(gt_path, str(i) + '.png'))
    img.append(os.path.join(img_path, str(i) + '.png'))

all = img + gt + pred
# 创建图像布局，包含3行5列的子图
fig, axes = plt.subplots(3, 5, figsize=(8, 8))

# 在每个子图上显示图片
for i, ax_row in enumerate(axes):
    for j, ax in enumerate(ax_row):
        image = cv.imread(all[i*5+j])
        ax.imshow(image)   # 设置颜色映射为灰度图
        ax.axis('off')  # 关闭坐标轴
        if j == 0:  # 每行的第一个子图
            if i == 0:
                ax.set_title('Origin',y=-0.2)  # 添加标题
            elif i == 1:
                ax.set_title('Ground Truth',y=-0.2)
            else:
                ax.set_title('Pred',y=-0.2)

# for i, ax in enumerate(axes.flat):
#     image = cv.imread(all[i])
#     ax.imshow(image)  # 设置颜色映射为灰度图
#     ax.axis('off')  # 关闭坐标轴

# 调整子图之间的间距
plt.tight_layout()

# 显示图像
plt.show()