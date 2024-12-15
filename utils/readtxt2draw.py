import matplotlib.pyplot as plt

def smooth_curve(points, factor=0):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

# 读取.txt文件中的数据
epoch = []
lr = []
tr_loss = []
val_loss = []
acc = []
f1 = []
iou = []
presion = []
recall = []
with open(r"E:\code\CDofHD\laymanofCD\result\tuerqi\FCCDN+BAM\1698240386.8586342.txt", 'r') as file:
    lines = file.readlines()[1:]
    for line in lines:
        values = line.split('\t')
        epoch.append(float(values[0]))
        lr.append(float(values[1]))
        tr_loss.append(float(values[2]))
        val_loss.append(float(values[3]))
        acc.append(float(values[4]))
        f1.append(float(values[5]))
        iou.append(float(values[6]))
        presion.append(float(values[7]))
        recall.append(float(values[8]))

tr_loss = smooth_curve(tr_loss)
val_loss = smooth_curve(val_loss)

# 创建折线图
plt.figure(figsize=(8, 6))
plt.plot(epoch, val_loss, linestyle='-')
plt.grid(True)
plt.grid(False)

# 显示图形
plt.show()