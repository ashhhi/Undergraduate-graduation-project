#导入相关库
import matplotlib.pyplot as plt
import numpy as np

start=-5 #输入需要绘制的起始值（从左到右）
stop=5 #输入需要绘制的终点值
step=0.01#输入步长


# #函数
# g=lambda z:np.maximum(0,z)
# num=(stop-start)/step #计算点的个数
# x = np.linspace(start,stop,int(num))
# y = g(x)
# plt.plot(x, y,label='ReLU')
#
# g=lambda z:1/(1+np.exp(-x))
# num=(stop-start)/step #计算点的个数
# x = np.linspace(start,stop,int(num))
# y = g(x)
# plt.plot(x, y,label='Sigmoid')
#
# g=lambda z:(np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
# num=(stop-start)/step #计算点的个数
# x = np.linspace(start,stop,int(num))
# y = g(x)
# plt.plot(x, y,label='Tanh')

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)
num=(stop-start)/step #计算点的个数
x = np.linspace(start,stop,int(num))
y = softmax(x)
plt.plot(x, y,label='Softmax')

plt.grid(True)#显示网格
plt.legend()#显示旁注#注意：不会显示后来再定义的旁注
plt.show()
