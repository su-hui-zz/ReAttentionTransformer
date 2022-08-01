import torch
import torch.nn as nn
from functools import partial


from .vision_transformer import VisionTransformer, _cfg, Attention, Block 
from timm.models.registry import register_model
import math
import torch.nn.functional as F
import pdb


__all__ = [
    'deit_tscam_small_patch16_224', 'deit_tscam_base_patch16_224', 'deit_base_patch16_224', 'deitpos_tscam_base_patch16_224',
    'deit_trt_base_patch16_224', 'deit_trt_fuse_base_patch16_224'
]



class TSCAM(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = nn.Conv2d(self.embed_dim, self.num_classes, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.head.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to return patch embedding outputs
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        attn_weights = []
        for blk in self.blocks:
            x, weights = blk(x)
            attn_weights.append(weights)

        x = self.norm(x)
        return x[:, 0], x[:, 1:], attn_weights

    def forward(self, x, return_cam=False):
        x_cls, x_patch, attn_weights = self.forward_features(x)
        n, p, c = x_patch.shape
        x_patch = torch.reshape(x_patch, [n, int(p**0.5), int(p**0.5), c])
        x_patch = x_patch.permute([0, 3, 1, 2])
        x_patch = x_patch.contiguous()
        x_patch = self.head(x_patch)
        x_logits = self.avgpool(x_patch).squeeze(3).squeeze(2)

        if self.training:
            return x_logits
        else:
            # # origin
            # attn_weights = torch.stack(attn_weights)        # 12 * B * H * N * N
            # attn_weights = torch.mean(attn_weights, dim=2)  # 12 * B * N * N

            # feature_map = x_patch.detach().clone()    # B * C * 14 * 14
            # n, c, h, w = feature_map.shape
            # cams = attn_weights.sum(0)[:, 0, 1:].reshape([n, h, w]).unsqueeze(1)
            # cams = cams * feature_map                           # B * C * 14 * 14

            # # no featuremap
            attn_weights = torch.stack(attn_weights)        # 12 * B * H * N * N
            attn_weights = torch.mean(attn_weights, dim=2)  # 12 * B * N * N

            feature_map = x_patch.detach().clone()    # B * C * 14 * 14
            n, c, h, w = feature_map.shape
            cams = attn_weights.sum(0)[:, 0, 1:].reshape([n, h, w]).unsqueeze(1).expand(n, c, h, w)
            cams = cams #* feature_map                           # B * C * 14 * 14

            # # way1 - attn_weights sum(dim=-2)
            # attn_weights = torch.stack(attn_weights)        # 12 * B * H * N * N
            # attn_weights = torch.mean(attn_weights, dim=2)  # 12 * B * N * N

            # feature_map = x_patch.detach().clone()    # B * C * 14 * 14
            # n, c, h, w = feature_map.shape
            # cams = attn_weights.sum(0)[:, 1:, 1:].sum(dim=-2).reshape([n, h, w]).unsqueeze(1)
            # cams = cams * feature_map                           # B * C * 14 * 14 

            # # way2 - class-specific and cls-agnostic fusion
            # attn_w = attn_weights[0].mean(dim=1) # B*N*N
            # att_total = attn_w[:,0,1:]
            # for att in attn_weights[1:]:
            #     att = att.mean(dim=1)            # B*N*N
            #     att_total += (att[:,0,1:].unsqueeze(dim=2) * attn_w[:,1:,1:]).sum(dim=1) # B*N
            #     attn_w  = att
            # att_total = att_total / len(attn_weights) # B*N

            # feature_map = x_patch.detach().clone()    # B * C * 14 * 14
            # n, c, h, w = feature_map.shape
            # cams = att_total.reshape([n, h, w]).unsqueeze(1)
            # cams = cams * feature_map                           # B * C * 14 * 14 

            # # way3 best in CUB (no feature map) -- way3和way2的差别仅仅是没有feature map的乘积，但是效果却更好了。（当前层的cls-atten和上一层的token-atten相乘）
            # attn_w = attn_weights[0].mean(dim=1) # B*N*N
            # att_total = attn_w[:,0,1:]
            # for att in attn_weights[1:]:
            #     att = att.mean(dim=1)            # B*N*N
            #     att_total += (att[:,0,1:].unsqueeze(dim=2) * attn_w[:,1:,1:]).sum(dim=1) # B*N
            #     attn_w  = att
            # att_total = att_total / len(attn_weights) # B*N

            # feature_map = x_patch.detach().clone()    # B * C * 14 * 14
            # n, c, h, w = feature_map.shape
            # cams = att_total.reshape([n, h, w]).unsqueeze(1).expand(n, c, h, w) # B * C * 14 * 14 

            # # way4 ： way3基础上，将attn_w取所有层和的均值，效果不如way3。attn_w还是仅取上一层的效果好（还是way3好，多层融合无论是乘法还是加法，都降点明显）
            # attn_w = attn_weights[0].mean(dim=1) # B*N*N
            # att_total = attn_w[:,0,1:]
            # for att in attn_weights[1:]:
            #     att = att.mean(dim=1)            # B*N*N
            #     att_total += (att[:,0,1:].unsqueeze(dim=2) * attn_w[:,1:,1:]).sum(dim=1) # B*N
            #     attn_w  = att + attn_w # B*N*N
            #     attn_w  = attn_w / attn_w.sum(dim=-1,keepdim=True)
            # att_total = att_total / len(attn_weights) # B*N

            # feature_map = x_patch.detach().clone()    # B * C * 14 * 14
            # n, c, h, w = feature_map.shape
            # cams = att_total.reshape([n, h, w]).unsqueeze(1).expand(n, c, h, w) # B * C * 14 * 14                          

            # # way5 ： 当前层的cls-atten和当前层的token-atten相乘 (肯定是不如way3好)
            # att_total = attn_weights[0].mean(dim=1)[:,0,1:] # B*N
            # for att in attn_weights[1:]:
            #     att = att.mean(dim=1)            # B*N*N
            #     att_total += (att[:,0,1:].unsqueeze(dim=2) * att[:,1:,1:]).sum(dim=1) # B*N
            # att_total = att_total / len(attn_weights) # B*N

            # feature_map = x_patch.detach().clone()    # B * C * 14 * 14
            # n, c, h, w = feature_map.shape
            # cams = att_total.reshape([n, h, w]).unsqueeze(1).expand(n, c, h, w) # B * C * 14 * 14 

            # # way6 - class-specific and cls-agnostic fusion
            # feature_map = x_patch.detach().clone()    # B * C * 14 * 14
            # n, c, h, w = feature_map.shape
            # attn_w = attn_weights[0].mean(dim=1) # B*N*N
            # att_total = attn_w[:,0,1:]
            # for att in attn_weights[1:]:
            #     att = att.mean(dim=1)            # B*N*N
            #     cama = att[:,0,1:].reshape([n, h, w]).unsqueeze(1) * feature_map # B * C * 14 * 14 
            #     x_logitsv, x_logitsi = x_logits.sort(dim=-1,descending=True) 
            #     cama = cama.reshape(n,c,h*w) # B , C , 14 * 14 
            #     cama = torch.gather(cama,dim=1,index = x_logitsi[:,0].unsqueeze(-1).unsqueeze(-1).expand(n,1,h*w))
            #     att_total += (cama.reshape(n,h*w).unsqueeze(dim=2) * attn_w[:,1:,1:]).sum(dim=1) # B*N
            #     attn_w  = att
            # att_total = att_total / len(attn_weights) # B*N

            # cams = att_total.reshape([n, h, w]).unsqueeze(1).expand(n, c, h, w)
            # cams = cams * feature_map                           # B * C * 14 * 14 

            # # # way7 - class-specific and cls-agnostic fusion  (和 way6的区别仅仅是token-atten对应的层, 在imagenet上两者效果差不多，但是cub数据集上效果更差点)
            # feature_map = x_patch.detach().clone()    # B * C * 14 * 14
            # n, c, h, w = feature_map.shape
            # att_total = torch.zeros(n,h*w, device = feature_map.device)
            # for att in attn_weights:
            #     att = att.mean(dim=1)            # B*N*N
            #     cama = att[:,0,1:].reshape([n, h, w]).unsqueeze(1) * feature_map # B * C * 14 * 14 
            #     x_logitsv, x_logitsi = x_logits.sort(dim=-1,descending=True) 
            #     cama = cama.reshape(n,c,h*w) # B , C , 14 * 14 
            #     cama = torch.gather(cama,dim=1,index = x_logitsi[:,0].unsqueeze(-1).unsqueeze(-1).expand(n,1,h*w))
            #     att_total += (cama.reshape(n,h*w).unsqueeze(dim=2) * att[:,1:,1:]).sum(dim=1) # B*N
            # att_total = att_total / len(attn_weights) # B*N

            # cams = att_total.reshape([n, h, w]).unsqueeze(1).expand(n, c, h, w)
            # cams = cams * feature_map 

            return x_logits, cams


class Deit(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, x, return_cam = True):
        x, attn_weights = self.forward_features(x)
        x = self.head(x)
        if self.training:
            return x
        else:
            attn_weights = torch.stack(attn_weights)        # 12 * B * H * N * N
            attn_weights = torch.mean(attn_weights, dim=2)  # 12 * B * N * N
            attn_weights = attn_weights.sum(0)              # B * N * N
            n, seq_num, _ = attn_weights.size()
            h,w = int(math.sqrt(seq_num-1)),int(math.sqrt(seq_num-1))
            cams = attn_weights[:, 0, 1:].reshape([n, h, w]).unsqueeze(1).expand(n, self.num_classes, h, w)
            return x, cams


class DeitPos224(VisionTransformer):
    def __init__(self, *args, **kwargs):
        self.select_num = kwargs['select_num'] # 50
        self.score_num = kwargs['score_num'] # 25
        del kwargs['select_num']
        del kwargs['score_num']
        super().__init__(*args, **kwargs)
        
        self.num_classes = kwargs['num_classes']
        self.score_layer = Block(
                dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                drop=self.drop_rate, attn_drop=self.attn_drop, drop_path=self.dpr[-1], norm_layer=self.norm_layer, vis=self.vis)
        self.score_norm = self.norm_layer(self.embed_dim, eps=1e-6)
        self.score_head = nn.Linear(self.embed_dim, 1)

    def forward_featurees(self, x):
        return None

    def forward(self, x, return_cam=False):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to return patch embedding outputs
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        attn_weights = []
        for blk in self.blocks[:-1]:
            x, weights = blk(x)
            attn_weights.append(weights)
        bz,seq_num,feat_num = x.size()

        weight_plc = torch.zeros(bz,seq_num-1,dtype=weights.dtype, device=weights.device)
        for weights in attn_weights:
            weight_plc += weights[:,:,0,1:].mean(dim=1)
        wval, wind = weight_plc.sort(dim=1, descending=True)
        score_states = torch.gather(x[:,1:],dim=1,index=wind.unsqueeze(-1).expand(bz,seq_num-1,feat_num))
        score_states = score_states[:,:self.select_num]
    
        score_tokens, _ = self.score_layer(score_states)   
        score_tokens= self.score_norm(score_tokens)
        scores      = self.score_head(score_tokens)
        scores      = torch.nn.functional.softmax(scores, dim=1) # token num

        # score_sort_v, score_sort_i = scores.sort(dim=1,descending=True)
        # select_tokens = torch.gather(score_states, dim=1, index = score_sort_i[:,:self.score_num].expand(bz,self.score_num,feat_num))
        #total_tokens, _  = self.blocks[-1](torch.cat([x[:,0].unsqueeze(dim=1),select_tokens],dim=1))
        select_tokens = (score_states * scores.expand(bz, self.select_num, feat_num)).sum(1)
        total_tokens, _  = self.blocks[-1](torch.stack([x[:,0],select_tokens],dim=1))
        total_encoded = self.norm(total_tokens)  
        logits = self.head(total_encoded[:,0])
        #x = self.norm(x)

        if self.training:
            return logits
        else:
            # test fuse weight_plc and select token
            mask = torch.zeros(bz,seq_num-1,dtype=weights.dtype, device=weights.device)
            mask = mask.scatter(1, wind[:,:self.select_num], 1)
            weight_plc = weight_plc / weight_plc.sum(dim=-1, keepdim=True)
            weight_rat = (weight_plc*mask).sum(dim=-1, keepdim=True)
            scores = scores/ scores.sum(dim=1, keepdim=True) * weight_rat.unsqueeze(-1)
        
            weight_plc = weight_plc.scatter(1, wind[:,:self.select_num], scores.view(bz,self.select_num))

            feat_sz = int(math.sqrt(seq_num-1))
            return logits,weight_plc.reshape([bz,feat_sz,feat_sz]).unsqueeze(dim=1).expand(bz, self.num_classes, feat_sz,feat_sz)


class TRT(VisionTransformer):
    def __init__(self, *args, **kwargs):
        # self.select_num = kwargs['select_num'] # 50
        # self.score_num = kwargs['score_num'] # 25
        self.thresh = kwargs['thresh']
        #del kwargs['select_num']
        #del kwargs['score_num']
        del kwargs['thresh']
        super().__init__(*args, **kwargs)
        
        self.num_classes = kwargs['num_classes']
        self.score_layer = Block(
                dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                drop=self.drop_rate, attn_drop=self.attn_drop, drop_path=self.dpr[-1], norm_layer=self.norm_layer, vis=self.vis)
        self.score_norm = self.norm_layer(self.embed_dim, eps=1e-6)
        self.score_head = nn.Linear(self.embed_dim, 1)

    def forward_featurees(self, x):
        return None

    def forward(self, x, return_cam=False, eps = 1e-6, thresh=None):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to return patch embedding outputs
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        attn_weights = []
        for blk in self.blocks[:-1]:
            x, weights = blk(x)
            attn_weights.append(weights)
        bz,seq_num,feat_num = x.size()

        weight_plc = torch.zeros(bz,seq_num-1,dtype=weights.dtype, device=weights.device)
        for weights in attn_weights:
            weight_plc += weights[:,:,0,1:].mean(dim=1) #+ weights[:,:,1:,1:].mean(dim=1).mean(dim=1)
        weight_plc = weight_plc/weight_plc.sum(dim=1, keepdim=True)

        #cumsum
        wval, wind = weight_plc.sort(dim=1, descending=True)
        wval_cum   = wval.cumsum(dim=1)
        wval_policy= (wval_cum < self.thresh) .to(torch.float16).view(bz,seq_num-1)  # 占据80%重要性的token进行re-attention，剩余20%重要性的token不进行舍弃 
        part_policy= torch.zeros(bz,seq_num-1,dtype=wval_policy.dtype, device=wval_policy.device)
        #part_policy = part_policy.scatter(1,(wval_policy*wind).long(),1.) # 挑选出wval_policy为1的wind系数，对应构建part_policy
        part_policy = part_policy.scatter(dim=1,index=wind,src=wval_policy)

        # pdb.set_trace()
        # print(part_policy)
        #print("policy_num:",part_policy.sum())
        #part_policy[wval_policy*wind] = 1 

        score_tokens, _ = self.score_layer(x[:,1:], part_policy)   
        score_tokens= self.score_norm(score_tokens)
        scores      = self.score_head(score_tokens)
        scores      = scores.exp_() * part_policy.unsqueeze(-1)
        scores      = (scores) / (scores.sum(dim=1, keepdim=True) + eps)

        select_tokens = (x[:,1:] * scores.expand(bz, seq_num-1, feat_num)).sum(1)
        total_tokens, _  = self.blocks[-1](torch.stack([x[:,0],select_tokens],dim=1))
        total_encoded = self.norm(total_tokens)  
        logits = self.head(total_encoded[:,0])
        #x = self.norm(x)

        if self.training:
            return logits
        else:
            # test fuse weight_plc and select token
            #mask = torch.zeros(bz,seq_num-1,dtype=weights.dtype, device=weights.device)
            #mask = mask.scatter(1, wind[:,:self.select_num], 1)
            weight_plc = weight_plc / weight_plc.sum(dim=-1, keepdim=True)
            weight_rat = (weight_plc*part_policy).sum(dim=-1, keepdim=True)
            scores     = scores.view(bz,seq_num-1) #* weight_plc
            scores     = scores/ scores.sum(dim=1, keepdim=True) * weight_rat
            scores     = scores + (1-part_policy) * weight_plc 
            feat_sz = int(math.sqrt(seq_num-1))
            return logits,scores.reshape([bz,feat_sz,feat_sz]).unsqueeze(dim=1).expand(bz, self.num_classes, feat_sz,feat_sz)


class CAM6FuseBlock(nn.Module):
    def __init__(self, dim, num_heads, num_classes, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, vis=False):
        super().__init__()
        self.embed_dim = dim
        self.num_classes = num_classes
        self.cam_tr  = Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop, attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, vis=vis)
        self.cam_norm = norm_layer(dim, eps=1e-6)
        self.cam_head = nn.Conv2d(dim, self.num_classes, kernel_size=3, stride=1, padding=1)
        self.cam_avgpool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self,x):
        # cam 
        bz,seq_num,feat_num = x.size()
        #cam_x = self.cam_norm(x[:,1:])
        #cam_x = self.cam_l1(x[:,1:])
        cam_trx,_ = self.cam_tr(x)  # x 表示倒数第二个transformer block 输出的特征
        cam_trx = self.cam_norm(cam_trx)
        cam_x = torch.reshape(cam_trx[:,1:], [bz, int((seq_num-1)**0.5), int((seq_num-1)**0.5), self.embed_dim])
        cam_x = cam_x.permute([0,3,1,2])
        cam_x = cam_x.contiguous()
        #cam_x = self.cam_c1(cam_x)
        cam_x = self.cam_head(cam_x)
        cam_logits = self.cam_avgpool(cam_x).squeeze(3).squeeze(2)
        return cam_logits, cam_x


# 与DeitPosThresh224FuseCAM差异，cam之前一个transformer block, 一个conv layer, 针对倒数第二个transformer block输出的特征
class TRTFuse(VisionTransformer):
    def __init__(self, *args, **kwargs):
        #self.select_num = kwargs['select_num'] # 50
        #self.score_num = kwargs['score_num'] # 25
        self.thresh = kwargs['thresh']
        #del kwargs['select_num']
        #del kwargs['score_num']
        del kwargs['thresh']
        super().__init__(*args, **kwargs)
        
        self.num_classes = kwargs['num_classes']
        self.score_layer = Block(
                dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                drop=self.drop_rate, attn_drop=self.attn_drop, drop_path=self.dpr[-1], norm_layer=self.norm_layer, vis=self.vis)
        self.score_norm = self.norm_layer(self.embed_dim, eps=1e-6)
        self.score_head = nn.Linear(self.embed_dim, 1)

        # cam
        self.camfuse_block = CAM6FuseBlock(dim=self.embed_dim, num_heads=self.num_heads, num_classes = self.num_classes, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,drop=self.drop_rate, attn_drop=self.attn_drop, drop_path=self.dpr[-1], norm_layer=self.norm_layer, vis=self.vis)

        self.camfuse_block.cam_head.apply(self._init_weights)
        # self.cam_tr  = Block(dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
        #                 drop=self.drop_rate, attn_drop=self.attn_drop, drop_path=self.dpr[-1], norm_layer=self.norm_layer, vis=self.vis)
        # self.cam_norm = self.norm_layer(self.embed_dim, eps=1e-6)
        # self.cam_head = nn.Conv2d(self.embed_dim, self.num_classes, kernel_size=3, stride=1, padding=1)
        # self.cam_avgpool = nn.AdaptiveAvgPool2d(1)
        # self.cam_head.apply(self._init_weights)
    

    def forward_featurees(self, x):
        return None

    def forward(self, x, labels=None, return_cam=False, return_atten= False, eps = 1e-6):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to return patch embedding outputs
        
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        attn_weights = []
        for blk in self.blocks[:-2]:
            x, weights = blk(x)
            attn_weights.append(weights)
        partial_x, weights = self.blocks[-2](x)
        attn_weights.append(weights)
        bz,seq_num,feat_num = x.size()

        weight_plc = torch.zeros(bz,seq_num-1,dtype=weights.dtype, device=weights.device)
        for weights in attn_weights:
            weight_plc += weights[:,:,0,1:].mean(dim=1)
        weight_plc = weight_plc/weight_plc.sum(dim=1, keepdim=True)

        #cumsum
        wval, wind = weight_plc.sort(dim=1, descending=True)
        wval_cum   = wval.cumsum(dim=1)
        wval_policy= (wval_cum < self.thresh) .to(torch.float16).view(bz,seq_num-1)  # 占据80%重要性的token进行re-attention，剩余20%重要性的token不进行舍弃 
        part_policy= torch.zeros(bz,seq_num-1,dtype=wval_policy.dtype, device=wval_policy.device)
        #part_policy = part_policy.scatter(1,(wval_policy*wind).long(),1.) # 挑选出wval_policy为1的wind系数，对应构建part_policy
        part_policy = part_policy.scatter(dim=1,index=wind,src=wval_policy)

        score_tokens, attn_policy_infs = self.score_layer(partial_x[:,1:], part_policy)   
        score_tokens= self.score_norm(score_tokens)
        scores      = self.score_head(score_tokens)
        scores      = scores.exp_() * part_policy.unsqueeze(-1)
        scores      = (scores) / (scores.sum(dim=1, keepdim=True) + eps)

        select_tokens = (partial_x[:,1:] * scores.expand(bz, seq_num-1, feat_num)).sum(1)
        total_tokens, _  = self.blocks[-1](torch.stack([partial_x[:,0],select_tokens],dim=1))
        total_encoded = self.norm(total_tokens)  
        logits = self.head(total_encoded[:,0])
        #x = self.norm(x)

        # cam 
        cam_logits, cam_x = self.camfuse_block.forward(x)
        # cam_trx,_ = self.cam_tr(x)  # x 表示倒数第二个transformer block 输出的特征
        # cam_trx = self.cam_norm(cam_trx)
        # cam_x = torch.reshape(cam_trx[:,1:], [bz, int((seq_num-1)**0.5), int((seq_num-1)**0.5), self.embed_dim])
        # cam_x = cam_x.permute([0,3,1,2])
        # cam_x = cam_x.contiguous()
        # #cam_x = self.cam_c1(cam_x)
        # cam_x = self.cam_head(cam_x)
        # cam_logits = self.cam_avgpool(cam_x).squeeze(3).squeeze(2)

        if labels != None or return_cam==False:
            #return logits
            return cam_logits
        else:
            # 整体融合
            # test fuse weight_plc and select token
            weight_rat = (weight_plc*part_policy).sum(dim=-1, keepdim=True)
            scores = scores.view(bz,seq_num-1) * weight_rat
            scores = weight_plc - weight_plc * part_policy + scores

            feat_sz = int(math.sqrt(seq_num-1))
            feat_cls_agnostic = scores.reshape([bz,feat_sz,feat_sz]).unsqueeze(dim=1).expand(bz, self.num_classes, feat_sz,feat_sz)
            cams = feat_cls_agnostic * cam_x

            # 通用部分
            if return_atten == True:
                return logits, cams, feat_cls_agnostic, cam_x, attn_weights, part_policy, attn_policy_infs[-1]

            #return cam_logits,cams
            return logits, cams



@register_model
def deit_tscam_small_patch16_224(pretrained=False, **kwargs):
    model = TSCAM(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        # checkpoint = torch.hub.load_state_dict_from_url(
        #     url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
        #     map_location="cpu", check_hash=True
        # )['model']
        checkpoint = torch.load("./pretraineds/deit_small_patch16_224-cd65a155.pth",map_location="cpu")['model']
        model_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


@register_model
def deit_tscam_base_patch16_224(pretrained=False, **kwargs):
    model = TSCAM(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        # checkpoint = torch.hub.load_state_dict_from_url(
        #     url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
        #     map_location="cpu", check_hash=True
        # )['model']
        checkpoint = torch.load("./pretraineds/deit_base_patch16_224-b5f2ef4d.pth",map_location="cpu")['model']
        model_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    model = Deit(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        # checkpoint = torch.hub.load_state_dict_from_url(
        #     url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
        #     map_location="cpu", check_hash=True
        # )['model']
        checkpoint = torch.load("./pretraineds/deit_base_patch16_224-b5f2ef4d.pth",map_location="cpu")['model']
        model_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


@register_model
def deitpos_tscam_base_patch16_224(pretrained=False, **kwargs):

    model = DeitPos224(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()

    if pretrained:
        # checkpoint = torch.hub.load_state_dict_from_url(
        #     url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
        #     map_location="cpu", check_hash=True
        # )['model']
        checkpoint = torch.load("./pretraineds/deit_base_patch16_224-b5f2ef4d.pth",map_location="cpu")['model']
        model_dict = model.state_dict()
        for k, v in model_dict.items():
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                del checkpoint[k]
                print(k + ' delete')
            elif k not in checkpoint and 'score_layer' in k:
                checkpoint[k] = checkpoint['blocks.11.' + '.'.join(k.split('.')[1:])]
                print(k + ' copy from ' + 'blocks.11.' + '.'.join(k.split('.')[1:]))
            elif k not in checkpoint and 'score_norm' in k:
                checkpoint[k] = checkpoint[k.replace('score_norm','norm')]

        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)
    return model





@register_model
def deit_trt_base_patch16_224(pretrained=False, **kwargs):

    model = TRT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()

    if pretrained:
        # checkpoint = torch.hub.load_state_dict_from_url(
        #     url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
        #     map_location="cpu", check_hash=True
        # )['model']

        checkpoint = torch.load("./pretraineds/deit_base_patch16_224-b5f2ef4d.pth",map_location="cpu")['model']
        model_dict = model.state_dict()
        for k, v in model_dict.items():
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                del checkpoint[k]
                print(k + ' delete')
            elif k not in checkpoint and 'score_layer' in k:
                checkpoint[k] = checkpoint['blocks.11.' + '.'.join(k.split('.')[1:])]
                print(k + ' copy from ' + 'blocks.11.' + '.'.join(k.split('.')[1:]))
            elif k not in checkpoint and 'score_norm' in k:
                checkpoint[k] = checkpoint[k.replace('score_norm','norm')]

        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)
    return model#, fine_param    


@register_model
def deit_trt_fuse_base_patch16_224(pretrained=False, **kwargs):

    model = TRTFuse(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()


    # only for cub 

    # for ilsvrc
    # if pretrained:
    #     # checkpoint = torch.hub.load_state_dict_from_url(
    #     #     url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
    #     #     map_location="cpu", check_hash=True
    #     # )['model']

    #     checkpoint = torch.load("./pretraineds/deit_base_patch16_224-b5f2ef4d.pth",map_location="cpu")['model']
    #     model_dict = model.state_dict()
    #     for k,v in model_dict.items():
    #         if 'cam_tr' in k and k not in checkpoint:
    #             checkpoint[k] = checkpoint['blocks.11.' + '.'.join(k.split('.')[2:])]
    #             print('copy '+ k + ' from ' + 'blocks.11.' + '.'.join(k.split('.')[2:]))
    #         elif 'cam_norm' in k and k not in checkpoint:
    #             checkpoint[k] = checkpoint[k.replace('camfuse_block.cam_norm','norm')]
    #             print('copy '+ k + ' from ' + k.replace('camfuse_block.cam_norm','norm'))   
    #         elif 'score_layer' in k and k not in checkpoint:
    #             checkpoint[k] = checkpoint['blocks.11.' + '.'.join(k.split('.')[1:])]
    #             print(k + ' copy from ' + 'blocks.11.' + '.'.join(k.split('.')[1:]))
    #         elif 'score_norm' in k and k not in checkpoint:
    #             checkpoint[k] = checkpoint[k.replace('score_norm','norm')]

    #     model_dict.update(checkpoint)
    #     model.load_state_dict(model_dict)

    # only for ilsvrc, load pretrained weights
    #if pretrained:
        # checkpoint = torch.hub.load_state_dict_from_url(
        #     url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
        #     map_location="cpu", check_hash=True
        # )['model']
        #checkpoint = torch.load("./pretraineds/deit_base_patch16_224-b5f2ef4d.pth",map_location="cpu")['model']

        #checkpoint = torch.load("./ckpt/ImageNet/deit_trt_base_patch16_224_0.95_0.688/ckpt/model_best_top1_loc.pth",map_location="cpu")['state_dict']
        
        # check_dict = {}
        # for k,v in checkpoint.items():
        #     check_dict[k.replace('module.','')] = v
        # model_dict = model.state_dict()
        # model_dict.update(check_dict)

        # checkpoint_pre = torch.load("./pretraineds/deit_base_patch16_224-b5f2ef4d.pth",map_location="cpu")['model']
        # for k,v in model_dict.items():
        #     if 'cam_tr' in k and k not in checkpoint and k not in checkpoint_pre:
        #         model_dict[k] = checkpoint_pre['blocks.11.' + '.'.join(k.split('.')[2:])]
        #         print('copy '+ k + ' from ' + 'blocks.11.' + '.'.join(k.split('.')[2:]))
        #     elif 'cam_norm' in k and k not in checkpoint and k not in checkpoint_pre:
        #         model_dict[k] = checkpoint_pre[k.replace('camfuse_block.cam_norm','norm')]
        #         print('copy '+ k + ' from ' + k.replace('camfuse_block.cam_norm','norm'))   

        # model.load_state_dict(model_dict)
    
    if pretrained:
        checkpoint = torch.load("./pretraineds/deit_base_patch16_224-b5f2ef4d.pth",map_location="cpu")['model']
        model_dict = model.state_dict()
        
        for k,v in model_dict.items():
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                del checkpoint[k]
                print(k + ' delete')
            elif 'score_layer' in k and k not in checkpoint:
                checkpoint[k] = checkpoint['blocks.11.' + '.'.join(k.split('.')[1:])]
                print(k + ' copy from ' + 'blocks.11.' + '.'.join(k.split('.')[1:]))
            elif 'score_norm' in k and k not in checkpoint:
                checkpoint[k] = checkpoint[k.replace('score_norm','norm')]
            elif 'cam_tr' in k and k not in checkpoint:
                checkpoint[k] = checkpoint['blocks.11.' + '.'.join(k.split('.')[2:])]
                print('copy '+ k + ' from ' + 'blocks.11.' + '.'.join(k.split('.')[2:]))
            elif 'cam_norm' in k and k not in checkpoint:
                checkpoint[k] = checkpoint[k.replace('camfuse_block.cam_norm','norm')]
                print('copy '+ k + ' from ' + k.replace('camfuse_block.cam_norm','norm')) 
        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)

    return model



@register_model
def deit_trt_base_patch16_384(pretrained=False, **kwargs):

    model = TRT(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()

    if pretrained:
        # checkpoint = torch.hub.load_state_dict_from_url(
        #     url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
        #     map_location="cpu", check_hash=True
        # )['model']
        checkpoint = torch.load("./pretraineds/deit_base_patch16_384-8de9b5d1.pth",map_location="cpu")['model']
        model_dict = model.state_dict()
        for k, v in model_dict.items():
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                del checkpoint[k]
                print(k + ' delete')
            elif k not in checkpoint and 'score_layer' in k:
                checkpoint[k] = checkpoint['blocks.11.' + '.'.join(k.split('.')[1:])]
                print(k + ' copy from ' + 'blocks.11.' + '.'.join(k.split('.')[1:]))
            elif k not in checkpoint and 'score_norm' in k:
                checkpoint[k] = checkpoint[k.replace('score_norm','norm')]

        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)
    return model



@register_model
def deit_trt_fuse_base_patch16_384(pretrained=False, **kwargs):

        model = TRTFuse(
                img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        model.default_cfg = _cfg()

        # only for cub
        if pretrained:
        #     checkpoint = torch.load("./pretraineds/deit_base_patch16_224-b5f2ef4d.pth",map_location="cpu")['model']
            checkpoint = torch.load("./ckpt_save/CUB/deit_trt_base_patch16_384_TOKENTHR0.65_BS128_0.946/ckpt/model_best_top1_loc.pth",map_location="cpu")['state_dict']
            check_dict = {}
            for k,v in checkpoint.items():
                check_dict[k.replace('module.','')] = v
            model_dict = model.state_dict()
            model_dict.update(check_dict)
            checkpoint_pre = torch.load("./pretraineds/deit_base_patch16_224-b5f2ef4d.pth",map_location="cpu")['model']
            for k,v in model_dict.items():
                if 'cam_tr' in k and k not in checkpoint and k not in checkpoint_pre:
                    model_dict[k] = checkpoint_pre['blocks.11.' + '.'.join(k.split('.')[2:])]
                    print('copy '+ k + ' from ' + 'blocks.11.' + '.'.join(k.split('.')[2:]))
                elif 'cam_norm' in k and k not in checkpoint and k not in checkpoint_pre:
                    model_dict[k] = checkpoint_pre[k.replace('camfuse_block.cam_norm','norm')]
            model.load_state_dict(model_dict)
        return model

