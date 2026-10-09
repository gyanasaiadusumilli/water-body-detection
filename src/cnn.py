"""U-Net style CNN for pixel-wise water segmentation on multispectral tiles."""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """Encoder-decoder with skip connections.

    The encoder learns spatial context (shorelines, river shapes, shadows next to buildings)
    that a pixel-wise classifier cannot see; skip connections keep fine boundary detail.
    """

    def __init__(self, in_channels=10, base=32, depth=4):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]
        self.encoders = nn.ModuleList()
        c = in_channels
        for ch in chs[:-1]:
            self.encoders.append(ConvBlock(c, ch))
            c = ch
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(chs[-2], chs[-1], dropout=0.2)
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for ch in reversed(chs[:-1]):
            self.ups.append(nn.ConvTranspose2d(ch * 2, ch, 2, stride=2))
            self.decoders.append(ConvBlock(ch * 2, ch))
        self.head = nn.Conv2d(chs[0], 1, 1)

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            x = dec(torch.cat([x, skip], dim=1))
        return self.head(x)            # logits, (B, 1, H, W)
