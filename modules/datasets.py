import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
import random

class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())

        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample
class CT_RATE_MultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['ids']
        image_paths = example['images']  # 这里是一个包含多个图片路径的列表
        split = example['split']
        # 加载所有图片并应用转换
        images = []
        # 原始数据集有48张图片。计算负担太大，这里只取其中的16张。
        for img_path in image_paths[::3]:
            filePath = os.path.join(self.image_dir, img_path.split('_')[0])
            filePath = os.path.join(filePath, img_path)
            image = Image.open(filePath).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)
        
        # 将所有图片堆叠成一个张量
        image = torch.stack(images, 0)  # 维度为 (num_images, C, H, W)
        
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        
        # 返回的样本包括多个图片
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample
class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample
