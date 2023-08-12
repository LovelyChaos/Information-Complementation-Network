import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.nn import Dropout, Softmax, Linear, Conv3d, LayerNorm
from torch.nn.modules.utils import _pair, _triple
import configs as configs
from torch.distributions.normal import Normal

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu}
class DoubleConv(nn.Module): #双层卷积
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm3d( out_channels ),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm3d(out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)


        )

    def forward(self, x):
        return self.maxpool_conv(x)

class CNNEncoder(nn.Module):
    def __init__(self, config, n_channels=2):
        super(CNNEncoder, self).__init__()
        self.n_channels = n_channels
        decoder_channels = config.decoder_channels
        encoder_channels = config.encoder_channels#(16, 32, 32)
        self.down_num = config.down_num # 2
        self.inc = DoubleConv(n_channels, encoder_channels[0])
        self.down1 = Down(encoder_channels[0], encoder_channels[1])
        self.down2 = Down(encoder_channels[1], encoder_channels[2])
        self.width = encoder_channels[-1]
    def forward(self, x):
        features = []
        x1 = self.inc(x)#第一层卷积
        features.append(x1)
        x2 = self.down1(x1)#第二层卷积
        features.append(x2)
        feats = self.down2(x2)#第三层卷积
        features.append(feats)
        feats_down = feats
        for i in range(self.down_num):
            feats_down = nn.MaxPool3d(2)(feats_down)#利用最大池化得到最后2层跳跃连接
            features.append(feats_down)
        #print("features0:", features[0].size())#[2, 16, 160, 192, 224]
        #print( "features1;", features[1].size() )#[2, 32, 80, 96, 112]
        #print( "features2:", features[2].size() )#[2, 32, 40, 48, 56]
        #print( "features3:", features[3].size() )#[2, 32, 20, 24, 28]
        #print( "features4:", features[4].size() )#[2, 32, 10, 12, 14]
        return feats, features[::-1] # [::-1]所有元素反向

class Mlp(nn.Module):#前向神经网络，多层感知机，也叫人工神经网络（ANN，Artificial Neural Network）有三层神经网络
    def __init__(self, in_features, hidden_features, dropout_rate):
        super(Mlp, self).__init__()
        self.fc1 = Linear(in_features, hidden_features)
        self.fc2 = Linear(hidden_features, in_features)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(dropout_rate)# 0.1

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x) # dropout可以避免过拟合，并增强模型的泛化能力。
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Attention3D(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False,  attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim#[64, 128, 256, 512]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv3d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W, D):
        B, N, C = x.shape
        #print("x.shape---------------------------", x.shape)
        q = self.q(x)
        q = q.reshape(B, N, self.num_heads, C // self.num_heads)
        q = q.permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W, D)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        #print("q:", q.shape)
        #print("k:", k.shape)
        #print("v:", v.shape)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self,dim, num_heads, dropout_rate, mlp_ratio=4., sr_ratio=1):
        super(Block, self).__init__()
        self.hidden_size = dim # 252
        self.attention_norm = LayerNorm(dim, eps=1e-6)#LayerNorm 层归一化，train()和eval()对LayerNorm没有影响
        self.ffn_norm = LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ffn = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, dropout_rate=dropout_rate)
        self.attn = Attention3D(dim=dim, num_heads=num_heads, attn_drop=0., proj_drop=0., sr_ratio=sr_ratio)

    def forward(self, x, H, W, D):
        h = x

        x = self.attention_norm(x)
        x= self.attn(x, H, W, D)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x


class Encoder(nn.Module):
    def __init__(self,  dim, num_heads, dropout_rate, mlp_ratio, depth, sr_ratio=1):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()#在创建 ModuleList 的时候传入一个 module 的 列表，还可以使用extend 函数和 append 函数来添加模型
        self.encoder_norm = LayerNorm(dim, eps=1e-6)
        for _ in range(depth):
            layer = Block( dim=dim, num_heads=num_heads, dropout_rate=dropout_rate, mlp_ratio=mlp_ratio, sr_ratio=sr_ratio)
            self.layer.append(copy.deepcopy(layer))#和list 的append 方法一样，将 一个 Module 添加到ModuleList
            #deepcopy是因为layer里有数组，每次数组指向同一个数组，所以不能用copy

    def forward(self, hidden_states, H, W, D):
        for layer_block in self.layer:
            hidden_states= layer_block(hidden_states, H, W, D)
        encoded = self.encoder_norm(hidden_states)
        return encoded


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, img_size, patch_size, in_channels, dim, dropout_rate):
        super(Embeddings, self).__init__()
        patch_size = _triple(patch_size)#
        self.patch_size = _triple(patch_size)  #
        n_patches = int((img_size[0]// patch_size[0]) * (img_size[1]// patch_size[1]) * (img_size[2]// patch_size[2]))
        # 5*6*7=210,/2**down_factor是因为经过2层卷积后，初始图像尺寸变了
        self.patch_embeddings = Conv3d(in_channels=in_channels,
                                       out_channels=dim,#252
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, dim))#(1, 210, 252)
        # nn.Parameter将一个固定不可训练的tensor转换成可以训练的类型parameter
        self.dropout = Dropout(dropout_rate)#0.1

    def forward(self, x):
        B, C, H, W, D = x.shape
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))[2, 252, 5, 6, 7]
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)转置
        embeddings = x + self.position_embeddings#[2, 210, 252]
        embeddings = self.dropout(embeddings)
        H, W , D= H // self.patch_size[0], W // self.patch_size[1], D // self.patch_size[2]
        return embeddings, (H, W, D)

class Transformer(nn.Module):
    def __init__(self,config, img_size, patch_size, in_channels, dim, dropout_rate, num_heads, mlp_ratio, depth, sr_ratio=1):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings( img_size=img_size, patch_size=patch_size
                                      ,in_channels=in_channels, dim=dim, dropout_rate=dropout_rate)
        self.encoder = Encoder(  dim=dim, num_heads=num_heads
                                , dropout_rate=dropout_rate, mlp_ratio=mlp_ratio, depth=depth, sr_ratio=sr_ratio)

    def forward(self, x):
        B = x.shape[0]
        x, (H, W, D)= self.embeddings(x)#embedding_output=[2, 210, 252]
        x = self.encoder(x, H, W, D)  # (B, n_patch, hidden)encoded=[2, 210, 252]
        x = x.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()
        return x

class PVTVNet(nn.Module):
    def __init__(self, config, img_size=(160, 192, 224), mode='bilinear'):
        super(PVTVNet, self).__init__()
        self.inc = DoubleConv(2, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        self.down4 = Down(128, 256)
        self.transformer1 = Transformer( config, img_size = (160, 192, 224), patch_size = 8, in_channels = 16,
                                        dim = 64, dropout_rate = 0.1, num_heads = 8, mlp_ratio = 16, depth =8, sr_ratio=4)
        self.transformer2 = Transformer( config, img_size = (40, 48, 56), patch_size = 2, in_channels = 66,
                                        dim = 252, dropout_rate = 0.1, num_heads = 3, mlp_ratio = 8, depth = 4, sr_ratio=8)
        self.transformer3 = Transformer( config, img_size = (20, 24, 28), patch_size = 2, in_channels = 64,
                                        dim = 120, dropout_rate = 0.1, num_heads = 10, mlp_ratio = 4, depth = 12, sr_ratio=2)
        self.transformer4 = Transformer( config, img_size = (10, 12, 14), patch_size = 2, in_channels = 120,
                                        dim = 252, dropout_rate = 0.1, num_heads = 12, mlp_ratio = 2, depth = 12, sr_ratio=1)
        self.decoder = DecoderCup(config, img_size)
        self.reg_head = RegistrationHead(
            in_channels = config.decoder_channels[-1],  # 16
            out_channels = config['n_dims'],  # 3
            kernel_size = 3,
        )
        self.spatial_trans = SpatialTransformer(img_size, mode)
        # self.integrate = VecInt(img_size, int_steps)

    def forward(self, x):
        # x:[2, 2, 160, 192, 224]
        moving = x[:, 0:1, :, :]  # [2, 1, 160, 192, 224]
        x = self.inc(x)
        skip1 = x#[2, 16, 160, 192, 224]
        x = self.down1(x)
        skip2 = x#2, 32, 80, 96, 112
        x = self.down2(x)
        skip3 = x#2, 32, 40, 48, 56
        x = self.down3(x)
        skip4 = x
        x = self.down4(x)
        skip5 = x

        x = self.transformer1(skip1)  # (B, n_patch, hidden),x=([2, 210, 252])
        x = self.transformer3(x)
        x = self.transformer4(x)

        ##########decoder#############
        #skip4 = nn.MaxPool3d(2)(skip3)  # [2, 32, 20, 24, 28]
        #skip5 = nn.MaxPool3d(2)(skip4)  # [2, 32, 10, 12, 14]
        features = []
        features.append(skip5)
        features.append(skip4)
        features.append(skip3)
        features.append(skip2)
        features.append(skip1)
        x = self.decoder(x, features)  # ([2, 16, 160, 192, 224])
        flow = self.reg_head(x)  # [2, 3, 160, 192, 224]
        out = self.spatial_trans(moving, flow)  # [2, 1, 160, 192, 224])

        return out, flow


class Conv3dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.LeakyReLU(inplace=True)

        IN = nn.InstanceNorm3d(out_channels)

        super(Conv3dReLU, self).__init__(conv, IN, relu)

class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv3dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        #mode是可使用的上采样算法，scale_factor输出为输入的多少倍数， trilinear是三线性插值

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DecoderCup(nn.Module):
    def __init__(self, config, img_size):
        super().__init__()
        self.config = config
        self.down_factor = config.down_factor
        head_channels = config.conv_first_channel#512
        self.img_size = img_size
        self.conv_more = Conv3dReLU(
            252,#经过transformer后的通道数，并将其变为
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels
        self.patch_size = _triple(config.patches["size"])
        skip_channels = self.config.skip_channels#(32, 32, 32, 32, 16)
        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, features=None):
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
                #print(skip.shape)
            else:
                skip = None
            x = decoder_block(x, skip=skip)
            '''0
            x.shape: ---- torch.Size([2, 96, 10, 12, 14])
            1
            x.shape: ---- torch.Size([2, 48, 20, 24, 28])
            2
            x.shape: ---- torch.Size([2, 32, 40, 48, 56])
            3
            x.shape: ---- torch.Size([2, 32, 80, 96, 112])
            4
            x.shape: ---- torch.Size([2, 16, 160, 192, 224])'''
        return x

class RegistrationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        conv3d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv3d.weight.shape))
        conv3d.bias = nn.Parameter(torch.zeros(conv3d.bias.shape))
        super().__init__(conv3d)

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer

    Obtained from https://github.com/voxelmorph/voxelmorph
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations

        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

CONFIGS = {
    'ViT-V-Net': configs.get_3DReg_config(),
}