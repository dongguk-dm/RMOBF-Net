import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

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
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
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

class perceptual_loss(nn.Module):
    def __init__(self):
        num_classes = 636
        super(perceptual_loss, self).__init__()
        model = models.densenet161(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        model.cuda()
        # for feature extract
        for param in model.parameters():
            param.requires_grad = False

        model.load_state_dict(torch.load("./pretrained_fv.pth"))
        print("=====================================")
        print("pretrained finger_vein weight loaded")
        print("=====================================")

        print("=====================================")
        print("Get part of the pre-trained layer")
        print("=====================================")

        self.part_model = nn.Sequential()
        self.part_model.cuda()
        denseblock2_idx = 6

        for i, layer in enumerate(list(model.features)):
            self.part_model.add_module(str(i), layer)
            if i == denseblock2_idx: # denseblock2 이전까지만 불러오도록
                break;

    def forward(self, x, y):
        x = F.interpolate(x, size = 224)
        y = F.interpolate(y, size = 224)

        feat_x = self.part_model(x)
        feat_y = self.part_model(y)

        # euclindean distance
        diff = torch.square(feat_x - feat_y)
        sum = torch.sum(diff)
        perceptual_loss = torch.sqrt(sum)

        # mean
        mean_p_loss = perceptual_loss / (feat_x.shape[2] * feat_x.shape[3])
        
        return mean_p_loss