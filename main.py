from dataloader import Dataset_self
import torch
import argparse
import torch.nn as nn
import time
from tqdm import tqdm
from utils.utils import *
from model.HRSICD.HRSICD import HRSICD

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#参数
def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')

    parser.add_argument('--train_path',
                        default='data/shuguang/train',
                        help='train data path')

    parser.add_argument('--val_path',
                        default='data/shuguang/val',
                        help='val data path')

    parser.add_argument('--train_batch_size', default=16, help='validate data path')
    parser.add_argument('--val_batch_size', default=1, help='validate data path')
    parser.add_argument('--work_dir', default='result', help='the dir to save checkpoint and logs')
    parser.add_argument('--epoch', default=100, help='Total Epoch')
    parser.add_argument('--lr', default=0.0001, help='Initial learning rate')

    parser.add_argument('--net_name', default='HRSICD', help='Select your network')
    parser.add_argument('--path_name', default='HRSICD', help='Select your network')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.work_dir):
        os.mkdir(args.work_dir)

    data_loader_train = torch.utils.data.DataLoader(dataset=Dataset_self(args.train_path),
                                                    batch_size=args.train_batch_size,
                                                    shuffle=True, num_workers=4)

    data_loader_val = torch.utils.data.DataLoader(dataset=Dataset_self(args.val_path),
                                                    batch_size=args.val_batch_size,
                                                    shuffle=True, num_workers=4)


    net = HRSICD().to(device)

    loss_fuc = nn.BCELoss().to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))
    poly_lr_scheduler = PolyLR(optimizer,max_iter=args.epoch)

    minoitor = float('inf')

    begin_time = time.time()
    save_path = create_file(os.path.join(args.work_dir, args.path_name))
    os.mkdir(save_path)
    with open(os.path.join(save_path, str(begin_time) + '.txt'), 'a') as file:
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        file.write('Epoch'+'\t'+'lr' + '\t' + 'Train_Loss' + '\t' + 'Val_Loss' +
                    '\t' + 'Accuracy' + '\t' + 'F1_score' + '\t' + 'Kappa' + '\t'
                   + 'IoU' + '\n')
        file.close()
    for i in range(1,args.epoch+1):
        Train_Loss = 0
        net.train()
        for Iter, (image1, image2, gt) in tqdm(
                enumerate(data_loader_train),
                total=len(data_loader_train),
                desc="Train batches"):
            image1 = image1.to(device, dtype=torch.float)
            image2 = image2.to(device, dtype=torch.float)
            gt = gt.to(device, dtype=torch.float)
            pre = net(image1, image2)
            loss = loss_fuc(pre, gt)
            net.zero_grad()
            loss.backward()
            optimizer.step()
            Train_Loss += loss.data
        Train_Loss = Train_Loss/(Iter+1)
        print("Epoch:", "%04d" % (i), "train_loss =", "%.4f" % (Train_Loss))

        if True:
            Val_Loss = 0
            net.eval()
            confusion_matrix = [0,0,0,0]
            for Iter_val, (image1_val, image2_val, gt_val) in tqdm(
                    enumerate(data_loader_val),
                    total=len(data_loader_val),
                    desc="Val batches"):
                image1_val = image1_val.to(device, dtype=torch.float)
                image2_val = image2_val.to(device, dtype=torch.float)
                gt_val = gt_val.to(device, dtype=torch.float)
                pre_val = net(image1_val, image2_val)
                loss = loss_fuc(pre_val, gt_val)
                Val_Loss += loss.data

                pre_val = (pre_val > 0.5).float()
                pre_val = pre_val.cpu().detach().numpy()
                gt_val = gt_val.cpu().detach().numpy()
                confusion_matrix = np.sum([confusion_matrix, get_confusion_matrix(pre_val, gt_val)],
                                          axis=0).tolist()

            accuracy, f1_score, iou, precision, recall = get_metric(confusion_matrix)
            Val_Loss = Val_Loss / (Iter_val + 1)
            poly_lr_scheduler.step()

            if Val_Loss <= minoitor:
                minoitor = Val_Loss
                best_model_path = save_path + '/best_epoch' + str(i) + '_model.pth'
                torch.save(net, best_model_path)

            print("Epoch:", "%04d" % (i), "val_loss =", "%.4f" % (Val_Loss),
                  "accuracy =", "%.4f" % (accuracy), "f1_score =", "%.4f" % (f1_score),
                  "iou =", "%.4f" % (iou), "precision =", "%.4f" % (precision),"recall =", "%.4f" % (recall))

        with open(os.path.join(save_path, str(begin_time)+'.txt'), 'a') as file:
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            file.write(f'{i}\t{lr:.8f}\t{Train_Loss:.4f}\t{Val_Loss:.4f}'
                       f'\t{accuracy:.4f}\t{f1_score:.4f}\t{iou:.4f}'
                       f'\t{precision:.4f}\t{recall:.4f}\n')

            file.close()



