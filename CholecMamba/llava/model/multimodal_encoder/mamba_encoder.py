import torch
import torch.nn as nn
from .swin_umamba import VCMamba, VMambaImageProcessor


class VSSMVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = {}

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_path))
            return
        # 初始化 VSSMEncoder 模型
        self.image_processor = VMambaImageProcessor()
        #self.vision_tower = torch.load('/data/wzp/Code/MambaR1/LLaVA/llava/model/multimodal_encoder/encoder_model.pth')
        self.vision_tower = VCMamba()  # 根据你的 VSSMEncoder 初始化参数调整
        # print("stem.1 weight:", self.vision_tower.stem[1].weight)
        # print("stem.1 bias:", self.vision_tower.stem[1].bias)
        self.vision_tower.load_state_dict(torch.load('./encoder_state_0220.pth'))  # 加载权重
        self.vision_tower.requires_grad_(False)  # 冻结权重
        # print("stem.1 weight:", self.vision_tower.stem[1].weight)
        # print("stem.1 bias:", self.vision_tower.stem[1].bias)
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs
        if self.select_feature == 'patch':
            #image_features = image_features[:, 1:]
            image_features = image_features
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0))
                image_feature = self.feature_select(image_forward_out[4]).to(image.dtype)  # 取第 4 层特征
                image_features.append(image_feature.permute(0, 2, 3, 1).reshape(-1, 64, 1024))
            image_features = torch.cat(image_features, dim=0)    
            
        else:
            #print("222")
            image_forward_outs = self.vision_tower(images.to(device=self.device) )
            image_features = self.feature_select(image_forward_outs[4].permute(0, 2, 3, 1).reshape(-1, 64, 1024))  # 取第 4 层特征

            if image_features.shape[0] == 10:
                image_features = image_features.reshape(1, -1, 1024)
                #print("list1:", image_features.shape) 
            else:
                image_features = image_features.repeat(1, 10, 1) 
                #print("list2:", image_features.shape) 

            
            

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return torch.float32

    @property
    def device(self):
        return torch.device('cuda')

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        # 最后一层特征图的通道数
        return 1024

    @property
    def num_patches_per_side(self):
        # 特征图的空间分辨率
        return 8

    @property
    def num_patches(self):
        # 总 patch 数量
        return self.num_patches_per_side ** 2  # 64