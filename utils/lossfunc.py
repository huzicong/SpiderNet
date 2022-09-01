import torch
import torch.nn as nn
import torch.nn.functional as F

###### LOSSES #######
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss


class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(1,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss


class UDLoss(nn.Module):
    def __init__(self):
        super(UDLoss, self).__init__()

    def forward(self, x, y, sigma):
        differ = torch.exp(sigma * -1) * pow(x - y, 2) + 2 * sigma
        loss = torch.mean(differ)
        return loss
        

class AUDLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(AUDLoss, self).__init__()
        self.delta = delta

    def forward(self, x, y, sigma):
        differ = abs(x - y)
        differ = abs(x - y)
        differ[differ > self.delta] = torch.exp(-sigma[differ > self.delta]) * self.delta * \
                                      (differ[differ > self.delta] - 0.5 * self.delta) \
                                      + 2 * sigma[differ > self.delta]
        differ[differ <= self.delta] = 0.5 * torch.exp(-sigma[differ <= self.delta]) * pow(differ[differ <= self.delta],
                                                                                           2) \
                                       + 2 * sigma[differ <= self.delta]
        loss = torch.mean(differ)
        return loss


if __name__ == '__main__':
    criterion = AUDLoss()
    a = torch.tensor([[1,2,3,4,5,6],
                      [4,5,6,7,8,9]], dtype=torch.float32)
    b = a+3
    print(b)
    sigma = torch.tensor([[10,11,12,13,14,15],
                      [16,17,18,19,20,21]], dtype=torch.float32)

    loss = criterion(a, b, sigma)
    print(loss)

