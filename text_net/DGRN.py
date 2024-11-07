import torch.nn as nn
import torch
from .deform_conv import DCN_layer
import clip
import torch.distributed as dist

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class DGM(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, clip_model):
        super(DGM, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size

        self.dcn = DCN_layer(self.channels_in, self.channels_out, kernel_size,
                             padding=(kernel_size - 1) // 2, bias=False)
        self.sft = SFT_layer(self.channels_in, self.channels_out, clip_model)

        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x, inter, text_prompt):
        '''
        :param x: feature map: B * C * H * W
        :inter: degradation map: B * C * H * W
        '''
        dcn_out = self.dcn(x, inter)
        sft_out = self.sft(x, inter, text_prompt)
        out = dcn_out + sft_out
        out = x + out

        return out

# Projection Head 정의
class TextProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TextProjectionHead, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        ).float()
    
    def forward(self, x):
        return self.proj(x.float())



class SFT_layer(nn.Module):
    def __init__(self, channels_in, channels_out, clip_model):
        super(SFT_layer, self).__init__()
        self.clip_model = clip_model
        self.text_embed_dim = self.clip_model.text_projection.shape[1]
        self.conv_gamma = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_out, channels_out, 1, 1, 0, bias=False),
        )
        self.conv_beta = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_out, channels_out, 1, 1, 0, bias=False),
        )

        self.text_proj_head = TextProjectionHead(self.text_embed_dim, channels_out)

        '''
        self.text_gamma = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_out, channels_out, 1, 1, 0, bias=False),
        ).float()
        self.text_beta = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_out, channels_out, 1, 1, 0, bias=False),
        ).float()
        '''

        self.cross_attention = nn.MultiheadAttention(embed_dim=channels_out, num_heads=2)


    def forward(self, x, inter, text_prompt):
        '''
        :param x: degradation representation: B * C
        :param inter: degradation intermediate representation map: B * C * H * W
        '''
        # img_gamma = self.conv_gamma(inter)
        # img_beta = self.conv_beta(inter)

        B, C, H, W = inter.shape #cross attention

        rank = dist.get_rank()
        device = torch.device(f'cuda:{rank}')
        text_tokens = clip.tokenize(text_prompt).to(device)  # Tokenize the text prompts (Batch size)
        with torch.no_grad():
            text_embed = self.clip_model.encode_text(text_tokens)

        text_proj = self.text_proj_head(text_embed).float()

        # 텍스트 임베딩 차원 확장: (B, C, H, W)로 변경 #concat 
        # text_proj_expanded = text_proj.unsqueeze(-1).unsqueeze(-1).expand(B, self.conv_gamma[0].out_channels, H, W)
        text_proj_expanded = text_proj.unsqueeze(-1).unsqueeze(-1).expand(B, C, H, W)
        
        # 이미지 중간 표현과 텍스트 임베딩 결합 (concat)
        combined = inter * text_proj_expanded
        # combined = torch.cat([inter, text_proj_expanded], dim=1)

        # 이미지와 텍스트 기반 gamma와 beta 계산
        img_gamma = self.conv_gamma(combined)
        img_beta = self.conv_beta(combined)
        
        ''' simple concat
        text_gamma = self.text_gamma(text_proj.unsqueeze(-1).unsqueeze(-1))  # Reshape to match (B, C, H, W)
        text_beta = self.text_beta(text_proj.unsqueeze(-1).unsqueeze(-1))  # Reshape to match (B, C, H, W)
        '''

        '''
        text_proj = text_proj.unsqueeze(1).expand(-1, H*W, -1)  # B * (H*W) * C

        # 이미지 중간 표현 변환: B * (H*W) * C로 변경
        inter_flat = inter.view(B, C, -1).permute(2, 0, 1)  # (H*W) * B * C
        
        # Cross-attention 적용
        attn_output, _ = self.cross_attention(text_proj.permute(1, 0, 2), inter_flat, inter_flat)
        attn_output = attn_output.permute(1, 2, 0).view(B, C, H, W)  # B * C * H * W

        # Gamma와 Beta 계산
        img_gamma = self.conv_gamma(attn_output)
        img_beta = self.conv_beta(attn_output)
'''
        # concat으로 text 결합 실험
        return x * img_gamma + img_beta


class DGB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, clip_model):
        super(DGB, self).__init__()

        # self.da_conv1 = DGM(n_feat, n_feat, kernel_size)
        # self.da_conv2 = DGM(n_feat, n_feat, kernel_size)
        self.dgm1 = DGM(n_feat, n_feat, kernel_size, clip_model)
        self.dgm2 = DGM(n_feat, n_feat, kernel_size, clip_model)
        self.conv1 = conv(n_feat, n_feat, kernel_size)
        self.conv2 = conv(n_feat, n_feat, kernel_size)

        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x, inter, text_prompt):
        '''
        :param x: feature map: B * C * H * W
        :param inter: degradation representation: B * C * H * W
        '''

        out = self.relu(self.dgm1(x, inter, text_prompt))
        out = self.relu(self.conv1(out))
        out = self.relu(self.dgm2(out, inter, text_prompt))
        out = self.conv2(out) + x

        return out


class DGG(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, n_blocks, clip_model):
        super(DGG, self).__init__()
        self.n_blocks = n_blocks
        modules_body = [
            DGB(conv, n_feat, kernel_size, clip_model) \
            for _ in range(n_blocks)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size))

        self.body = nn.Sequential(*modules_body)

    def forward(self, x, inter, text_prompt):
        '''
        :param x: feature map: B * C * H * W
        :param inter: degradation representation: B * C * H * W
        '''
        res = x
        for i in range(self.n_blocks):
            res = self.body[i](res, inter, text_prompt)
        res = self.body[-1](res)
        res = res + x

        return res


class DGRN(nn.Module):
    def __init__(self, opt, clip, conv=default_conv):
        super(DGRN, self).__init__()

        self.n_groups = 5
        n_blocks = 5
        n_feats = 64
        kernel_size = 3

        # head module
        modules_head = [conv(3, n_feats, kernel_size)]
        self.head = nn.Sequential(*modules_head)

        # body
        modules_body = [
            DGG(default_conv, n_feats, kernel_size, n_blocks, clip) \
            for _ in range(self.n_groups)
        ]
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*modules_body)

        # tail
        modules_tail = [conv(n_feats, 3, kernel_size)]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x, inter, text_prompt):
        # head
        x = self.head(x)

        # body
        res = x
        for i in range(self.n_groups):
            res = self.body[i](res, inter, text_prompt)
        res = self.body[-1](res)
        res = res + x

        # tail
        x = self.tail(res)

        return x
