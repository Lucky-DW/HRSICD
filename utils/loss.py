import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BoundaryLoss(nn.Module):
    def __init__(self, weight_dice=0.5, weight_bce=0.5, smooth=1e-5):
        super(BoundaryLoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce
        self.smooth = smooth

    def forward(self, logits, targets):
        # 计算Dice损失
        intersection = (logits * targets).sum()
        union = logits.sum() + targets.sum() + self.smooth
        dice_loss = 1 - (2 * intersection + self.smooth) / union

        # 计算二进制交叉熵损失
        bce_loss = nn.BCELoss()(logits, targets)

        # 组合两种损失
        loss = self.weight_dice * dice_loss + self.weight_bce * bce_loss

        return loss


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class torch_MS_SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(torch_MS_SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        ssim_loss = _ssim(img1, img2, window, self.window_size, channel, self.size_average)

        bce_loss = nn.BCELoss()(img1, img2)

        loss = bce_loss + 1 - ssim_loss

        return loss / 2




if __name__ == '__main__':
    t1 = torch.rand(4, 1, 64, 64).to(device)
    t2 = torch.rand(4, 1, 64, 64).to(device)
    ms_ssim_criterion = torch_MS_SSIM()
    ms_ssim_value = ms_ssim_criterion(t1, t1)
    print(ms_ssim_value)
    BLS = BoundaryLoss()
    bls_loss = BLS(t1,t1)
    print(bls_loss)
