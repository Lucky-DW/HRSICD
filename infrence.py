import torchvision.transforms as transforms
from PIL import Image
import torch
import numpy as np
import rasterio
import time
from utils.utils import *
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

x1_transforms = transforms.Compose([transforms.ToTensor()])
x2_transforms = transforms.Compose([transforms.ToTensor()])

def get_opt_sar_DCprediction(image_path1, image_path2, modelfile, img_size, device=device):
    image_size = img_size   # 表示从大图中裁出的待预测小图大小（往模型里放）
    stride = img_size
    pre_channel = 3
    model = torch.load(modelfile)
    model.eval()

    with rasterio.open(image_path1) as ds:
        pre_img_A = ds.read([1, 2, 3])
        pre_img_A = np.array(pre_img_A)
        pre_img_A = np.uint8(pre_img_A)
        pre_img_A = Image.fromarray(np.transpose(pre_img_A, (1, 2, 0)))

    with rasterio.open(image_path2) as ds:
        pre_img_B = ds.read([1])
        pre_img_B = np.array(pre_img_B)
        pre_img_B = np.uint8(pre_img_B)
        pre_img_B = Image.fromarray(np.squeeze(pre_img_B)).convert('RGB')

    w = pre_img_A.size[0]  # 待预测影像宽
    h = pre_img_A.size[1]  # 待预测影像高

    padding_result = np.zeros((int(h / image_size) * image_size + image_size + stride,
                               int(w / image_size) * image_size + image_size + stride),
                              dtype=np.uint8)  # 创建预测结果的大影像   dtype=np.uint8



    with torch.no_grad():
        for i in range(0, w, stride):  # 通过步长来遍历整张大图
            for j in range(0, h, stride):
                box = (i, j, i + image_size, j + image_size)

                crop_img_A = pre_img_A.crop(box)  # 获取遍历图像（依次放进模型的小图像）
                crop_img_B = pre_img_B.crop(box)  # 获取遍历图像（依次放进模型的小图像）

                crop_img_A = x1_transforms(crop_img_A)
                crop_img_B = x2_transforms(crop_img_B)

                crop_img_A = crop_img_A.reshape(1, pre_channel, image_size,image_size).to(device)  # torch.Size([3, 512, 512])转为torch.Size([1, 3, 512, 512])
                crop_img_B = crop_img_B.reshape(1, pre_channel, image_size,image_size).to(device)  # 在torch里面，view函数相当于numpy的reshape,在函数的参数中经常可以看到-1例如x.view(-1, 4)这里-1表示一个不确定的数，就是你如果不确定你想要reshape成几行，但是你很肯定要reshape成4列，那不确定的地方就可以写成-1

                pre_data = model(crop_img_A, crop_img_B)

                mask_pre = (pre_data > 0.5).float()

                mask_pre = mask_pre.type(torch.FloatTensor)
                img_pre = torch.squeeze(mask_pre).numpy()  # torch.squeeze() 这个函数主要对数据的维度进行压缩，去掉维数为1的的维度，torch.unsqueeze()反之
                # .numpu()表示将数据类型转为ndarray格式
                padding_result[j: j + image_size, i: i + image_size] += (img_pre * 255).astype(np.uint8)

        padding_result_save = padding_result[0:h, 0:w]
        return padding_result_save


if __name__ == "__main__":
    i = 4
    path = r'result/MRSICD'

    modelfile = path + r'/best_epoch' + str(i) + '_model.pth'

    image_path1 = r'data/shuguang/t1/island.png'
    image_path2 = r'data/shuguang/t2/island.png'
    save_path = path + r'/shuguang.png'
    true_path = r'data/shuguang/gt/island.png'
    pre_path = save_path

    T1 = time.time()

    img_size = 256
    out = get_opt_sar_DCprediction(image_path1, image_path2, modelfile, img_size)
    IMG = Image.fromarray(out)
    IMG.save(save_path)

    T2 = time.time()

    pre = out
    true = np.array(Image.open(true_path))

    confusion_matrix = get_confusion_matrix(pre, true)
    accuracy, f1_score, iou, precision, recall = get_metric(confusion_matrix)
    print('{:.2f}'.format(f1_score * 100))
    print('{:.2f}'.format(iou * 100))
    print('{:.2f}'.format(precision * 100))
    print('{:.2f}'.format(recall * 100))
