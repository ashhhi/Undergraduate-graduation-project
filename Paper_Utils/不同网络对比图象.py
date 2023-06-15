import os

from matplotlib import pyplot as plt
import cv2 as cv


output_file = [
    'FCN_8S',
    'SegNet',
    'RefineNet',
    'DeepLabv3',
    'ResUnet3Plus',
    'ResUnet3Plus_ASPP_CBAMPlus'
]
net_name = [
    'FCN-8S',
    'SegNet',
    'RefineNet',
    'DeepLab v3+',
    'ResUnet3+',
    'ResUnet3+_ASPP_CBAM+'
]
index = 558

pred_file = []
gt_file = []
img_file = fr'C:\DataSet\DeepGlobal\{output_file[-3]}\img\{index}.png'
for i in output_file:
    pred_file.append('C:/DataSet/DeepGlobal/' + i + '/pred')
    gt_file.append('C:/DataSet/DeepGlobal/' + i + '/gt')

pred = []
gt = []
img = []

for i in range(len(output_file)):
    pred.append(os.path.join(pred_file[i],str(index)+'.png'))
    gt.append(os.path.join(gt_file[i], str(index) + '.png'))
    img.append(img_file)

all = img + gt + pred
# 创建图像布局，包含3行5列的子图
fig, axes = plt.subplots(3, len(output_file), figsize=(8, 8))
print(len(output_file))

# 在每个子图上显示图片
for i, ax_row in enumerate(axes):
    for j, ax in enumerate(ax_row):
        image = cv.imread(all[i*len(output_file)+j])
        ax.imshow(image)   # 设置颜色映射为灰度图
        ax.axis('off')  # 关闭坐标轴
        if i == 2:  # 每行的第一个子图

            ax.set_title(net_name[j], y=-0.2, fontsize=9)  # 添加标题

# for i, ax in enumerate(axes.flat):
#     image = cv.imread(all[i])
#     ax.imshow(image)  # 设置颜色映射为灰度图
#     ax.axis('off')  # 关闭坐标轴

# 调整子图之间的间距
plt.tight_layout()

# 显示图像
plt.show()