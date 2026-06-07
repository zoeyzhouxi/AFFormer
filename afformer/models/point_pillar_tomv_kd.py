import torch
import torch.nn as nn

from afformer.models.sub_modules.pillar_vfe import PillarVFE
from afformer.models.sub_modules.point_pillar_scatter import PointPillarScatter
from afformer.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from afformer.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from afformer.models.sub_modules.downsample_conv import DownsampleConv
from afformer.models.fuse_modules.fuse_utils import regroup
from afformer.models.sub_modules.naive_compress import NaiveCompressor
from afformer.models.sub_modules.tomv_basic import Encoder

class PointPillarTomvKd(nn.Module):
    def __init__(self, args):
        super(PointPillarTomvKd, self).__init__()

        self.max_cav = args['max_cav']
        self.fading = args['fading']

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        if 'resnet' in args['base_bev_backbone']:
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64)
        else:
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)

        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.out_channel = args['shrink_header']['dim'][-1]

        # self.fading_simulator = FadingSimulator(args['fading_noise'])
        if 'compression' in args and args['compression'] > 0:
            self.compression = True
            #compress spatially
            if 'stride' in args:
                stride = args['stride']
            else:
                stride = 1 
            self.naive_compressor = NaiveCompressor(256, args['compression'], stride)
            # print('using compression ratio {}, stride {}:'.format(args['compression'], stride))
        else:
            self.compression = False

        # encoder_type = args['encoder_type']

        # if encoder_type == 'ugf':
        #     from afformer.models.sub_modules.tomv_ablation import UgfEncoder
        #     self.fusion_net = UgfEncoder(args['transformer'])
        # elif encoder_type == 'iat':
        #     from afformer.models.sub_modules.tomv_ablation import IatEncoder
        #     self.fusion_net = IatEncoder(args['transformer'])
        # elif encoder_type == 'tat':
        #     from afformer.models.sub_modules.tomv_ablation import TatEncoder
        #     self.fusion_net = TatEncoder(args['transformer'])
        # elif encoder_type == 'mata':
        #     from afformer.models.sub_modules.tomv_ablation import MataEncoder
        #     self.fusion_net = MataEncoder(args['transformer'])
        # elif encoder_type == 'dualsa':
        #     from afformer.models.sub_modules.tomv_ablation import DualsaEncoder
        #     self.fusion_net = DualsaEncoder(args['transformer'])
        # else: 
        
        self.fusion_net = Encoder(args['transformer'])

        self.cls_head = nn.Conv2d(self.out_channel, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(self.out_channel, 7 * args['anchor_number'],
                                  kernel_size=1)

        self.use_dir = False
        if 'dir_args' in args.keys():
            self.use_dir = True
            self.dir_head = nn.Conv2d(self.out_channel, args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2

        if args['backbone_fix']:
            self.backbone_fix()

    def backbone_fix(self):
        """
        Fix the parameters of encoder during channel fading.
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

    def forward(self, data_dict):
       
        processed_lidar_torch_dict = data_dict['k_processed_lidar']
        record_len = data_dict['record_len']
        fading_gain = data_dict['fading_gain']
        SNR_db = data_dict['SNR_db']

        all_regroup_feature = []
        all_mask = []
        for i in range(len(processed_lidar_torch_dict)):
            voxel_features = processed_lidar_torch_dict[i]['voxel_features']
            voxel_coords = processed_lidar_torch_dict[i]['voxel_coords']
            voxel_num_points = processed_lidar_torch_dict[i]['voxel_num_points']

            batch_dict = {'voxel_features': voxel_features,
                        'voxel_coords': voxel_coords,
                        'voxel_num_points': voxel_num_points,
                        'record_len': record_len}

            # n, 4 -> n, c ('pillar_features')
            batch_dict = self.pillar_vfe(batch_dict)

            # n, c -> N, C, H, W (batch_cav_size, C, H, W) 
            # put pillars into spatial feature map ('spatial_features')
            batch_dict = self.scatter(batch_dict)
    
            batch_dict = self.backbone(batch_dict)

            spatial_features_2d = batch_dict['spatial_features_2d'] # torch.Size([(CAV_num1+CAV_num2)*k, 384, 96, 352])

            # downsample feature to reduce memory
            if self.shrink_flag:
                spatial_features_2d = self.shrink_conv(spatial_features_2d) # torch.Size([(CAV_num1+CAV_num2)*k, 256, 48, 176]) 

        
            regroup_feature, mask= regroup(spatial_features_2d, record_len, self.max_cav, self.fading, fading_gain, SNR_db)
            
            all_regroup_feature.append(regroup_feature.unsqueeze(2))
            all_mask.append(mask.unsqueeze(2))
        
        all_regroup_feature = torch.cat(all_regroup_feature, dim=2) # torch.Size([2, 5, 3, 256, 48, 176])
        all_mask = torch.cat(all_mask, dim=2) # torch.Size([2, 5, 3])

        # b l k c h w -> b l k h w c
        all_regroup_feature = all_regroup_feature.permute(0, 1, 2, 4, 5, 3) # torch.Size([B, max_cav, k, 48, 176, 256])
       
        # transformer fusion
        fused_feature, fused_embedding = self.fusion_net(all_regroup_feature, all_mask[:, :, 0]) # (B, C, H, W)
        # from afformer.visualization.vis_utils import visualize_feature_map
        # visualize_feature_map(fused_feature, 'fused_feature')
        
        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {'feature': fused_feature,
                       'embedding': fused_embedding,
                       'cls_preds': psm,
                       'reg_preds': rm}
        if self.use_dir:
            output_dict.update({'dir_preds': self.dir_head(fused_feature)})

        return output_dict

if __name__ == '__main__':

    args = {
            "anchor_number": 2,
            "backbone_fix": True,
            "fading": True,
            "encoder_type": "aff",
            "base_bev_backbone": {
                "compression": 0,
                "layer_nums": [3, 4, 5],
                "layer_strides": [2, 2, 2],
                "num_filters": [64, 128, 256],
                "num_upsample_filter": [128, 128, 128],
                "resnet": True,
                "upsample_strides": [1, 2, 4],
                "voxel_size": [0.4, 0.4, 5],
                },
            "compression": 64,
            "dir_args": {"anchor_yaw": [0, 90], "dir_offset": 0.7853, "num_bins": 2},
            "lidar_range": [-100.8, -40, -3.5, 100.8, 40, 1.5],
            "max_cav": 2,
            "pillar_vfe": {
                "num_filters": [64],
                "use_absolute_xyz": True,
                "use_norm": True,
                "with_distance": False,
                },
            "point_pillar_scatter": {"grid_size": [504, 200, 1], "num_features": 64},
            "seq_len": 2,
            "shrink_header": {
                "dim": [256],
                "input_dim": 384,
                "kernal_size": [3],
                "padding": [1],
                "stride": [1],
                },
            "transformer": {
                "cav_att_config": {
                    "depth": 1,
                    "dim": 256,
                    "dim_head": 32,
                    "dropout": 0.3,
                    "eps": 1e-05,
                    "heads": 8,
                    },
                "dim_in": 256,
                "dim_out": 64,
                "feed_forward": {
                    "bias": True,
                    "dropout": 0.3,
                    "ffn_expansion_factor": 2.66,
                    "mlp_dim": 256,
                    },
                "height_att_config": {
                    "bias": True,
                    "depth": 1,
                    "dim": 256,
                    "dim_head": 32,
                    "dropout": 0.3,
                    "eps": 1e-05,
                    "heads": 8,
                    },
                "temporal_att_config": {
                    "depth": 1,
                    "dim": 256,
                    "dim_head": 32,
                    "dropout": 0.3,
                    "eps": 1e-05,
                    "heads": 8,
                    },
                "width_att_config": {
                    "bias": True,
                    "depth": 1,
                    "dim": 256,
                    "dim_head": 32,
                    "dropout": 0.3,
                    "eps": 1e-05,
                    "heads": 8,
                    },
                },
            "voxel_size": [0.4, 0.4, 5],
            }

    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = PointPillarTomvKd(args).to(device).eval()

    from afformer.utils.complexity_utils import make_dummy_data_dict,profile_model
    data_dict = make_dummy_data_dict()
    profile_model(model, (data_dict))