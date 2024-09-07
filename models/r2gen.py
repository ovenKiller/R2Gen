import torch
import torch.nn as nn
import numpy as np

from modules.visual_extractor import VisualExtractor
from modules.encoder_decoder import EncoderDecoder


class R2GenModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = EncoderDecoder(args, tokenizer)
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        elif args.dataset_name == 'mimic_cxr':
            self.forward = self.forward_mimic_cxr
        elif args.dataset_name == 'CT_RATE':
            self.forward = self.forward_CT_RATE

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images, targets=None, mode='train'):
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output
    def forward_CT_RATE(self, images, targets=None, mode='train'):
        # 初始化列表来存储每张图片的特征
        fc_feats_list = []
        att_feats_list = []

        # 遍历每一张图片，并提取特征
        for i in range(images.size(1)):  # 假设images的形状是 (batch_size, n, C, H, W)
            att_feats, fc_feats = self.visual_extractor(images[:, i])
            fc_feats_list.append(fc_feats)
            att_feats_list.append(att_feats)

        # 将所有特征在特征维度上拼接
        fc_feats = torch.cat(fc_feats_list, dim=1)  # 拼接所有fc_feats
        att_feats = torch.cat(att_feats_list, dim=1)  # 拼接所有att_feats

        # 根据模式执行相应的操作
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError("Mode must be 'train' or 'sample'")
        
        return output

    def forward_mimic_cxr(self, images, targets=None, mode='train'):
        att_feats, fc_feats = self.visual_extractor(images)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output

