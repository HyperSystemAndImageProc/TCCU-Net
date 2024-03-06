import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from einops import rearrange
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch import nn
from torchmetrics.functional import accuracy

from ..vit import Transformer
# from transformer import Transformer
from .helpers import conv_1x1_bn, conv_nxn_bn




class CrossViT(pl.LightningModule):
    def __init__(self, image_size, num_classes, chs, dims, depths,
                 expansion=2, kernel_size=1, patch_size=(1, 1), norm_layer=nn.BatchNorm2d):#  1;samson:(3,3)(Best),(1,1);apex:(1,1);sim(1,1)
        super().__init__()

        midplanes =  outplanes = inplanes = 24

        self.conv1m = nn.Conv2d(inplanes, midplanes, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.bn1 = norm_layer(midplanes)
        self.conv2m = nn.Conv2d(inplanes, midplanes, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn2 = norm_layer(midplanes)
        self.conv3m = nn.Conv2d(midplanes, outplanes, kernel_size=1,  bias=True)
        self.pool1 = nn.AdaptiveAvgPool2d((None, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((1, None))
        self.relu = nn.ReLU(inplace=False)

        self.conv1 = conv_nxn_bn(image_size[0], chs[0], kernel_size, stride=2)

        self.mv2 = nn.ModuleList([])
        self.mv2.append(MV2Block(chs[0], chs[1], 1, expansion))
        self.mv2.append(MV2Block(chs[1], chs[2], 2, expansion))
        self.mv2.append(MV2Block(chs[2], chs[3], 1, expansion))
        self.mv2.append(MV2Block(chs[3], chs[3], 1, expansion))  # Repeat
        self.mv2.append(MV2Block(chs[3], chs[4], 2, expansion))
        self.mv2.append(MV2Block(chs[4], chs[5], 2, expansion))
        self.mv2.append(MV2Block(chs[5], chs[6], 2, expansion))

        self.mvit = nn.ModuleList([])
        self.mvit.append(MViTBlock(
            dims[0], depths[0], chs[4], kernel_size, patch_size, expansion=2))
        self.mvit.append(MViTBlock(
            dims[1], depths[1], chs[5], kernel_size, patch_size, expansion=4))
        self.mvit.append(MViTBlock(
            dims[2], depths[2], chs[6], kernel_size, patch_size, expansion=4))

        self.conv2 = conv_1x1_bn(chs[-2], chs[-1])

        _, H, W = image_size
        self.pool = nn.AvgPool2d((H // 32, W // 32), stride=1)
        self.fc = nn.Linear(chs[-1], num_classes, bias=False)

        self.hparams.lr = 2e-3
        self.save_hyperparameters()

    def forward(self, x):  # x:[1,24,95,95]
        x = self.conv1(x)
        _, _, h, w = x.size()
        x1 = self.pool1(x)
        x1 = self.conv1m(x1)
        x1 = self.bn1(x1)
        x1 = x1.expand(-1, -1, h, w)
        # x1 = F.interpolate(x1, (h, w))

        x2 = self.pool2(x)
        x2 = self.conv2m(x2)
        x2 = self.bn2(x2)
        x2 = x2.expand(-1, -1, h, w)
        # x2 = F.interpolate(x2, (h, w))

        x = self.relu(x1 + x2)
        x = self.conv3m(x).sigmoid() #1，24，48，48

         #[1,16,48,48]
        x = self.mv2[0](x) #[1,32 48,48]

        x = self.mv2[1](x) #[1,64,24,24]
        x = self.mv2[2](x)   #[1,64,24,24]
        x = self.mv2[3](x)  # Repeat [1,64,24,24]

        x = self.mv2[-3](x)  #[1,96,12,12]
        x = self.mvit[-3](x) #[1,96,12,12]

        x = self.mv2[-2](x)   #[1,128,6,6]
        x = self.mvit[-2](x)  #[1,128,6,6]

        x = self.mv2[-1](x)  # [1,160,3,3]
        x = self.mvit[-1](x) #  [1,160,3,3]
        x = self.conv2(x) # -->【1，640，3，3】= [1,chs[7],3,3]

        x = self.pool(x).view(-1, x.shape[1])# -->【4，chs[7]】
        x = self.fc(x) # -->[4,150]  = [4,num_class]
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y, label_smoothing=0.1)
        acc = accuracy(logits.argmax(dim=1), y)
        self.log('Loss/Train', loss, on_step=True)
        self.log('Accuracy/Train', acc, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y, label_smoothing=0.1)
        acc = accuracy(logits.argmax(dim=1), y)
        self.log('Loss/Valid', loss, on_epoch=True, reduce_fx=torch.mean)
        self.log('Accuracy/Valid', acc, on_epoch=True, reduce_fx=torch.mean)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=2500,
            cycle_decay=0.50,
            cycle_limit=10,
            warmup_t=2500,
            warmup_lr_init=2e-4,
            warmup_prefix=True,
        )
        return [optimizer], [scheduler]

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step(epoch=self.global_step)

    @property
    def lr(self):
        return self.hparams.lr

    @lr.setter
    def lr(self, lr):
        self.hparams.lr = lr


class MV2Block(nn.Module):

    def __init__(self, c_in, c_out, stride=1, expansion=4):
        super().__init__()
        assert stride in (1, 2)
        self.stride = stride

        h_dim = int(c_in * expansion)
        self.use_res_connect = stride == 1 and c_in == c_out

        self.conv = nn.Sequential(
            conv_1x1_bn(c_in, h_dim),
            conv_nxn_bn(h_dim, h_dim, 3, stride=stride, groups=h_dim),
            conv_1x1_bn(h_dim, c_out, activation=False),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MViTBlock(nn.Module):
    def __init__(self, dim, depth, channels, kernel_size, patch_size,
                 expansion=4, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channels, channels, kernel_size)
        self.conv2 = conv_1x1_bn(channels, dim)

        self.transformer = Transformer(dim, depth, 4, 8, expansion, dropout)

        self.conv3 = conv_1x1_bn(dim, channels)
        self.conv4 = conv_nxn_bn(2 * channels, channels, kernel_size)

    def forward(self, x):
        # clone input tensor for concatenation in fusion step
        x_hat = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # Global representations
        _, _, h, w = x.shape
        ph, pw = min(h, self.ph), min(w, self.pw)
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d',
                      ph=ph, pw=pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)',
                      ph=ph, pw=pw, h=(h // ph), w=(w // pw))

        # Fusion
        x = self.conv3(x) #[1,96,12,12]
        x = torch.cat((x, x_hat), axis=1) #[1,96+96=192,12,12]
        x = self.conv4(x)

        return x
