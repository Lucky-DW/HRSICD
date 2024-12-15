import os
from torch.optim import lr_scheduler
import numpy as np

def get_metric(confusion_matrix):
    TP = confusion_matrix[0]
    FP = confusion_matrix[1]
    TN = confusion_matrix[2]
    FN = confusion_matrix[3]

    # 计算准确度（Accuracy）
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    # 计算精确度（Precision）
    if TP + FP != 0:
        precision = TP / (TP + FP)
    else:
        precision = 0.0

    # 计算召回率（Recall）
    if TP + FN != 0:
        recall = TP / (TP + FN)
    else:
        recall = 0.0

    # 计算F1分数
    if precision + recall != 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    # 计算IoU（Intersection over Union）
    iou = TP / (TP + FP + FN)

    return accuracy, f1_score, iou, precision, recall

def get_confusion_matrix(predicted_labels,true_labels):
    true_labels = (true_labels / true_labels.max()).astype(float)
    predicted_labels = (predicted_labels / predicted_labels.max()).astype(float)
    predicted_labels = predicted_labels.astype('uint8')
    true_labels = true_labels.astype('uint8')


    TP = np.sum(np.logical_and(true_labels == 1, predicted_labels == 1))
    FP = np.sum(np.logical_and(true_labels == 0, predicted_labels == 1))
    TN = np.sum(np.logical_and(true_labels == 0, predicted_labels == 0))
    FN = np.sum(np.logical_and(true_labels == 1, predicted_labels == 0))

    return [TP, FP, TN, FN]

def create_file(filename):
    # 检查文件是否存在
    if not os.path.exists(filename):
        return filename  # 如果文件不存在，直接返回原文件名

    # 分离文件名和扩展名
    file_parts = filename.split('.')
    base = file_parts[0]
    extension = '.' + file_parts[1] if len(file_parts) > 1 else ''

    # 尝试不同的数字，直到找到一个不存在的文件名
    counter = 1
    while True:
        new_filename = f"{base}_{counter}{extension}"
        if not os.path.exists(new_filename):
            return new_filename  # 如果新文件名不存在，返回它
        counter += 1  # 否则，增加计数器

class PolyLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_iter, power=0.9, last_epoch=-1):
        self.max_iter = max_iter
        self.power = power
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            base_lr * ((1 - self.last_epoch / self.max_iter) ** self.power)
            for base_lr in self.base_lrs
        ]