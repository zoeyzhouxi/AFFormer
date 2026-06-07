# intermediate fusion dataset
import math
from collections import OrderedDict
import numpy as np
import torch
from afformer.utils import box_utils as box_utils
from afformer.utils.camera_utils import (
    sample_augmentation,
    img_transform,
    normalize_img,
    img_to_tensor,
)
from afformer.utils.common_utils import merge_features_to_dict
from afformer.utils.transformation_utils import x1_to_x2, get_pairwise_transformation
from afformer.utils.pcd_utils import (
    mask_points_by_range,
    mask_ego_points,
    shuffle_points,
    downsample_lidar_minimum,
)
from afformer.utils.channel_utils import fspl, winner_los, winner_nlos, rician_h_ss, rayleigh_h_ss


def getIntermediateFusionDataset(cls):
    """
    cls: the Basedataset.
    """
    class IntermediateFusionDataset(cls):
        def __init__(self, params, visualize, train=True, mode=None):
            super().__init__(params, visualize, train, mode)
            # intermediate and supervise single
            self.supervise_single = True if ('supervise_single' in params['model']['args'] and params['model']['args']['supervise_single']) \
                                        else False
            self.proj_first = True if 'proj_first' not in params['fusion']['args']\
                                         else params['fusion']['args']['proj_first']

            self.anchor_box = self.post_processor.generate_anchor_box()
            self.anchor_box_torch = torch.from_numpy(self.anchor_box)

            self.kd_flag = params.get('kd_flag', False) # disconet 如果params中有kd_flag,则使用设置的值, 如果配置参数中没有设置 kd_flag,则默认为 False
            self.dataset = params['fusion']['dataset']
            self.channel_params = params['channel_params']
            self.comm_range = params['comm_range']

        def get_item_single_car(self, selected_cav_base, ego_pose):
            """
            Process a single CAV's information for the train/test pipeline.

            Parameters
            ----------
            selected_cav_base : dict
                The dictionary contains a single CAV's raw information.
                including 'params', 'camera_data'
            ego_pose : list, length k
                The ego vehicle lidar pose under world coordinate.

            Returns
            -------
            selected_cav_processed : dict
                The dictionary contains the cav's processed information.
            """
            selected_cav_processed = {}
            
            # lidar
            if self.load_lidar_file or self.visualize:
                k_lidar_np = []
                k_projected_lidar = []
                k_processed_lidar = OrderedDict()
                k_transformation_matrix = {}
                for i in range(self.k):
                    # calculate the transformation matrix
                    k_transformation_matrix[i] = \
                        x1_to_x2(selected_cav_base['k_frames'][i]['params']['lidar_pose'],
                                ego_pose[i]) # T_ego_cav
                    # process lidar
                    lidar_np_i = selected_cav_base['k_frames'][i]['lidar_np']
                    lidar_np_i = shuffle_points(lidar_np_i)
                    # remove points that hit itself
                    lidar_np_i = mask_ego_points(lidar_np_i)
                    # project the lidar to ego space
                    # x,y,z in ego space
                    projected_lidar_i = \
                        box_utils.project_points_by_matrix_torch(lidar_np_i[:, :3],
                                                                    k_transformation_matrix[i])

                    if self.proj_first:
                        lidar_np_i[:, :3] = box_utils.project_points_by_matrix_torch(lidar_np_i[:, :3],
                                                                    k_transformation_matrix[i])
                        projected_lidar_i = lidar_np_i
                    else:
                        projected_lidar_i = lidar_np_i
                        projected_lidar_i[:, :3] = box_utils.project_points_by_matrix_torch(projected_lidar_i[:, :3],
                                                                    k_transformation_matrix[i])

                    projected_lidar_i = mask_points_by_range(projected_lidar_i,
                                        self.params['preprocess'][
                                            'cav_lidar_range'])
                    k_projected_lidar.append(projected_lidar_i)

                    k_lidar_np.append(lidar_np_i)

                    k_processed_lidar[i] = self.pre_processor.preprocess(lidar_np_i)
                
                selected_cav_processed.update({'k_processed_features': k_processed_lidar})

                
                selected_cav_processed.update({'current_projected_lidar': k_projected_lidar[0]})

                # if self.kd_flag:
                #     current_lidar_proj_np = copy.deepcopy(k_lidar_np[0])
                #     current_lidar_proj_np[:,:3] = k_projected_lidar[0][:, :3]

                #     selected_cav_processed.update({'current_projected_lidar': current_lidar_proj_np})

            
            # generate targets label single GT, note the reference pose is itself.
            object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center(
                [selected_cav_base], selected_cav_base['k_frames'][0]['params']['lidar_pose']
            )
            label_dict = self.post_processor.generate_label(
                gt_box_center=object_bbx_center, anchors=self.anchor_box, mask=object_bbx_mask
            )
            selected_cav_processed.update({
                                "single_label_dict": label_dict,
                                "single_object_bbx_center": object_bbx_center,
                                "single_object_bbx_mask": object_bbx_mask})

            # camera
            if self.load_camera_file:
                camera_data_list = selected_cav_base["camera_data"]

                params = selected_cav_base["params"]
                imgs = []
                rots = []
                trans = []
                intrins = []
                extrinsics = []
                post_rots = []
                post_trans = []

                for idx, img in enumerate(camera_data_list):
                    camera_to_lidar, camera_intrinsic = self.get_ext_int(params, idx)

                    intrin = torch.from_numpy(camera_intrinsic)
                    rot = torch.from_numpy(
                        camera_to_lidar[:3, :3]
                    )  # R_wc, we consider world-coord is the lidar-coord
                    tran = torch.from_numpy(camera_to_lidar[:3, 3])  # T_wc

                    post_rot = torch.eye(2)
                    post_tran = torch.zeros(2)

                    img_src = [img]

                    # depth
                    if self.load_depth_file:
                        depth_img = selected_cav_base["depth_data"][idx]
                        img_src.append(depth_img)
                    else:
                        depth_img = None

                    # data augmentation
                    resize, resize_dims, crop, flip, rotate = sample_augmentation(
                        self.data_aug_conf, self.train
                    )
                    img_src, post_rot2, post_tran2 = img_transform(
                        img_src,
                        post_rot,
                        post_tran,
                        resize=resize,
                        resize_dims=resize_dims,
                        crop=crop,
                        flip=flip,
                        rotate=rotate,
                    )
                    # for convenience, make augmentation matrices 3x3
                    post_tran = torch.zeros(3)
                    post_rot = torch.eye(3)
                    post_tran[:2] = post_tran2
                    post_rot[:2, :2] = post_rot2

                    # decouple RGB and Depth

                    img_src[0] = normalize_img(img_src[0])
                    if self.load_depth_file:
                        img_src[1] = img_to_tensor(img_src[1]) * 255

                    imgs.append(torch.cat(img_src, dim=0))
                    intrins.append(intrin)
                    extrinsics.append(torch.from_numpy(camera_to_lidar))
                    rots.append(rot)
                    trans.append(tran)
                    post_rots.append(post_rot)
                    post_trans.append(post_tran)
                    

                selected_cav_processed.update(
                    {
                    "image_inputs": 
                        {
                            "imgs": torch.stack(imgs), # [Ncam, 3or4, H, W]
                            "intrins": torch.stack(intrins),
                            "extrinsics": torch.stack(extrinsics),
                            "rots": torch.stack(rots),
                            "trans": torch.stack(trans),
                            "post_rots": torch.stack(post_rots),
                            "post_trans": torch.stack(post_trans),
                        }
                    }
                )

            # anchor box
            selected_cav_processed.update({"anchor_box": self.anchor_box})

            # note the reference pose ego
            object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center([selected_cav_base],
                                                        ego_pose[0])

            selected_cav_processed.update(
                {
                    "object_bbx_center": object_bbx_center[object_bbx_mask == 1],
                    "object_bbx_mask": object_bbx_mask,
                    "object_ids": object_ids,
                    'k_transformation_matrix': k_transformation_matrix
                }
            )
            if self.dataset == 'v2xset':
                object_bbx_corner = self.post_processor.generate_object_corner([selected_cav_base],
                                                        ego_pose[0])
            else:
                object_bbx_corner = self.post_processor.generate_object_corner_dairv2x([selected_cav_base],
                                                        ego_pose[0])
            selected_cav_processed.update({
                'object_bbx_corner': object_bbx_corner}
                )

            return selected_cav_processed

        def __getitem__(self, idx):
            base_data_dict = self.retrieve_base_data(idx)
            # base_data_dict = add_noise_data_dict(base_data_dict,self.params['noise_setting'])

            processed_data_dict = OrderedDict()
            processed_data_dict['ego'] = {}

            ego_id = -1
            k_ego_lidar_pose = []

            # first find the ego vehicle's lidar pose
            for cav_id, cav_content in base_data_dict.items():
                if cav_content['ego']:
                    
                    ego_id = cav_id
                    
                    for i, content in cav_content['k_frames'].items():
                        k_ego_lidar_pose.append(content['params']['lidar_pose']) # list with len k. 每个元素为list with len 6.
                    break
             
            assert cav_id == list(base_data_dict.keys())[
                0], "The first element in the OrderedDict must be ego"
            assert ego_id != -1
            # import ipdb; ipdb.set_trace()
            assert len(k_ego_lidar_pose[0]) > 0

            too_far = []
            cav_id_list = []            
            k_lidar_poses_list = []

            # loop over all CAVs to process information
            for cav_id, selected_cav_base in base_data_dict.items():
                # check if the cav is within the communication range with ego
                distance = \
                    math.sqrt((selected_cav_base['k_frames'][0]['params']['lidar_pose'][0] -
                            k_ego_lidar_pose[0][0]) ** 2 + (
                                    selected_cav_base['k_frames'][0]['params'][
                                        'lidar_pose'][1] - k_ego_lidar_pose[0][
                                        1]) ** 2)

                # if distance is too far, we will just skip this agent
                if distance > self.params['comm_range']:
                    too_far.append(cav_id)
                    continue
                    
                
                lidar_pose_list = []
                
                for k in range(len(selected_cav_base['k_frames'])):
                    lidar_pose_list.append(selected_cav_base['k_frames'][k]['params']['lidar_pose']) # 6dof pose
                k_lidar_poses = np.array(lidar_pose_list).reshape(-1, 6)  # [k, 6]
               
                k_lidar_poses_list.append(k_lidar_poses) # n_CAV x [k, 6]
               
                cav_id_list.append(cav_id)   

            for cav_id in too_far:
                base_data_dict.pop(cav_id)

            k_pairwise_t_matrix = \
                get_pairwise_transformation(base_data_dict,
                                                self.max_cav,
                                                self.proj_first) # [k, L, L, 4, 4]

            k_lidar_poses = np.array(k_lidar_poses_list)  # [N_cav, k, 6]
            
            # merge preprocessed features from different cavs into the same dict
            cav_num = len(cav_id_list)
            fading_gain_list = []
            SNR_db_list = []
            agents_image_inputs = []
            processed_features = []
            object_stack = []
            object_id_stack = []
            single_label_list = []
            single_object_bbx_center_list = []
            single_object_bbx_mask_list = []
            projected_lidar_stack = []

            early_fusion_lidar_stack = []
            object_bbx_corner_stack = []
            
            for _i, cav_id in enumerate(cav_id_list):
                selected_cav_base = base_data_dict[cav_id]

                selected_cav_processed = self.get_item_single_car(
                    selected_cav_base,
                    k_ego_lidar_pose)

                object_stack.append(selected_cav_processed['object_bbx_center'])
                object_id_stack += selected_cav_processed['object_ids']
                if self.load_lidar_file:
                    processed_features.append(
                        selected_cav_processed['k_processed_features'])
                if self.load_camera_file:
                    agents_image_inputs.append(
                        selected_cav_processed['image_inputs'])

                if self.visualize or self.kd_flag:
                    projected_lidar_stack.append(
                        selected_cav_processed['current_projected_lidar'])
                
                if self.supervise_single:
                    single_label_list.append(selected_cav_processed['single_label_dict'])
                    single_object_bbx_center_list.append(selected_cav_processed['single_object_bbx_center'])
                    single_object_bbx_mask_list.append(selected_cav_processed['single_object_bbx_mask'])

                object_bbx_corner_stack.append(selected_cav_processed['object_bbx_corner'])
                early_fusion_lidar_stack.append(selected_cav_processed['current_projected_lidar'])

                if cav_id != ego_id: # channel fading & path loss 
                    scenario_type = selected_cav_base['scenario_info']['scenario_type']
                    V2I_K_dB = selected_cav_base['scenario_info']['V2I_K_dB']
                    V2V_K_dB = selected_cav_base['scenario_info']['V2V_K_dB']
                    f_GHz = self.channel_params['freq_GHz']
                    d = math.sqrt((selected_cav_base['k_frames'][0]['params']['lidar_pose'][0] - k_ego_lidar_pose[0][0]) ** 2 + \
                            (selected_cav_base['k_frames'][0]['params']['lidar_pose'][1] - k_ego_lidar_pose[0][1]) ** 2 + \
                            (selected_cav_base['k_frames'][0]['params']['lidar_pose'][2] - k_ego_lidar_pose[0][2]) ** 2 )
                    d0 = 0.5 * self.comm_range
                    if int(cav_id) < 0 or scenario_type == 'V2I-LOS':
                        pl_dB = fspl(d, f_GHz)
                        pl0_dB = fspl(d0, f_GHz)                
                        h_ss = rician_h_ss(V2I_K_dB)           
                        Pr_dBm = self.channel_params['Pt_dBm'] + self.channel_params['Gt_rsu_dBi'] + self.channel_params['Gr_dBi'] - pl_dB
                    else:
                        if scenario_type == 'NLOS':
                            d1 = np.abs(selected_cav_base['k_frames'][0]['params']['lidar_pose'][0] - k_ego_lidar_pose[0][0])
                            d2 = np.abs(selected_cav_base['k_frames'][0]['params']['lidar_pose'][1] - k_ego_lidar_pose[0][1])
                            pl_dB = winner_nlos(d1, d2, f_GHz)
                            pl0_dB = winner_nlos(d0/2, d0/2, f_GHz)
                            h_ss = rayleigh_h_ss()
                        else:
                            pl_dB = winner_los(d, f_GHz)
                            pl0_dB = winner_los(d0, f_GHz)
                            h_ss = rician_h_ss(V2V_K_dB)
                        Pr_dBm = self.channel_params['Pt_dBm'] + self.channel_params['Gt_cav_dBi'] + self.channel_params['Gr_dBi'] - pl_dB
                    delta_pl_dB = pl_dB - pl0_dB      
                    pl_gain_rel_amp = 10**(-delta_pl_dB/20)
                    fading_gain = pl_gain_rel_amp * np.abs(h_ss)      
                    noise_dBm = -174 + 10*np.log10(self.channel_params['bandwidth_Hz']) + self.channel_params['NF_dB']
                    SNR_dB = Pr_dBm - noise_dBm 
                else:
                    fading_gain = 1
                    SNR_dB = 100
                
                fading_gain_list.append(fading_gain)
                if self.channel_params['SNR_fix']:
                    SNR_db_list.append(self.channel_params['SNR'])
                else:
                    SNR_db_list.append(SNR_dB)
        
            fading_gain_list = torch.from_numpy(np.array(fading_gain_list))
            SNR_db_list = torch.from_numpy(np.array(SNR_db_list))
            # import ipdb; ipdb.set_trace()

            processed_data_dict['ego'].update({'fading_gain_list': fading_gain_list,
                                               'SNR_db_list': SNR_db_list})
            early_fusion_all_lidar = np.vstack(early_fusion_lidar_stack)
            nums = len(object_bbx_corner_stack)
            processed_features_paint = []
            all_inside_points_list = []
            color_projected_lidar_stack = []
            for i in range(nums):
                if len(object_bbx_corner_stack[i]) == 0:
                    temp = np.hstack((early_fusion_lidar_stack[i], np.zeros((early_fusion_lidar_stack[i].shape[0], 1))))
                    all_inside_points_list.append(temp)
                    temp = self.pre_processor.preprocess_paint(temp)
                    processed_features_paint.append(temp)
                    continue
                object_merged_dict = {}

                for object_id, object_bbx in object_bbx_corner_stack[i].items():
                        object_merged_dict[object_id] = object_bbx
                all_inside_points = self.object_all_inside_points(early_fusion_all_lidar,object_bbx_corner_stack[i])

                all_inside_points = np.hstack((all_inside_points, np.ones((all_inside_points.shape[0], 1))))
                    
                all_inside_points_list.append(all_inside_points)
                    
                all_outside_points  = self.object_all_outside_points(early_fusion_lidar_stack[i],object_bbx_corner_stack[i])
                all_outside_points = np.hstack((all_outside_points, np.zeros((all_outside_points.shape[0], 1))))

                rec_points = np.concatenate((all_inside_points,all_outside_points), axis=0)

                color_projected_lidar_stack.append(rec_points)
                    
                processed_rec_lidar = self.pre_processor.preprocess_paint(rec_points)
                processed_features_paint.append(processed_rec_lidar)

            for all_inside_points in all_inside_points_list:
                processed_all_inside_points = self.pre_processor.preprocess_paint(all_inside_points)
                processed_features_paint.append(processed_all_inside_points)
            
            early_fusion_processed_features = self.pre_processor.preprocess(early_fusion_all_lidar)
            processed_data_dict['ego'].update({
                    'early_fusion_processed_features':early_fusion_processed_features})

            # generate single view GT label
            if self.supervise_single:
                single_label_dicts = self.post_processor.collate_batch(single_label_list)
                single_object_bbx_center = torch.from_numpy(np.array(single_object_bbx_center_list))
                single_object_bbx_mask = torch.from_numpy(np.array(single_object_bbx_mask_list))
                processed_data_dict['ego'].update({
                    "single_label_dict_torch": single_label_dicts,
                    "single_object_bbx_center_torch": single_object_bbx_center,
                    "single_object_bbx_mask_torch": single_object_bbx_mask,
                    })

            if self.kd_flag:
                stack_lidar_np = np.vstack(projected_lidar_stack)
                stack_lidar_np = mask_points_by_range(stack_lidar_np,
                                            self.params['preprocess'][
                                                'cav_lidar_range'])
                stack_feature_processed = self.pre_processor.preprocess(stack_lidar_np)
                processed_data_dict['ego'].update({'teacher_processed_lidar':
                stack_feature_processed})

            
            # exclude all repetitive objects    
            unique_indices = \
                [object_id_stack.index(x) for x in set(object_id_stack)]
            object_stack = np.vstack(object_stack)
            object_stack = object_stack[unique_indices]

            # make sure bounding boxes across all frames have the same number
            object_bbx_center = \
                np.zeros((self.params['postprocess']['max_num'], 7))
            mask = np.zeros(self.params['postprocess']['max_num'])
            object_bbx_center[:object_stack.shape[0], :] = object_stack
            mask[:object_stack.shape[0]] = 1
            
            if self.load_lidar_file:
                merged_feature_dict = merge_features_to_dict(processed_features)
                merged_feature_dict_paint = merge_features_to_dict(processed_features_paint)
                for i in range(self.k):
                    merged_feature_dict[i] = merge_features_to_dict(merged_feature_dict[i])
                processed_data_dict['ego'].update({'k_processed_lidar': merged_feature_dict})
                '''
                k:
                    'voxel_features': cav_num * [voxel_num, 32, 4]
                    'voxel_coords':  cav_num * [voxel_num, 3]
                    'voxel_num_points': cav_num * [voxel_num]
                '''
            if self.load_camera_file:
                merged_image_inputs_dict = merge_features_to_dict(agents_image_inputs, merge='stack')
                processed_data_dict['ego'].update({'image_inputs': merged_image_inputs_dict})


            # generate targets label
            label_dict = \
                self.post_processor.generate_label(
                    gt_box_center=object_bbx_center,
                    anchors=self.anchor_box,
                    mask=mask)

            processed_data_dict['ego'].update(
                {'object_bbx_center': object_bbx_center,
                'object_bbx_mask': mask,
                'object_ids': [object_id_stack[i] for i in unique_indices],
                'anchor_box': self.anchor_box,
                'processed_lidar_paint': merged_feature_dict_paint,
                'label_dict': label_dict,
                'cav_num': cav_num,
                'k_pairwise_t_matrix': k_pairwise_t_matrix,
                'k_lidar_poses': k_lidar_poses})


            if self.visualize:
                processed_data_dict['ego'].update({'origin_lidar':
                    np.vstack(
                        projected_lidar_stack)})


            processed_data_dict['ego'].update({'sample_idx': idx,
                                                'cav_id_list': cav_id_list})

            return processed_data_dict


        def collate_batch_train(self, batch):
            # Intermediate fusion is different the other two
            output_dict = {'ego': {}}

            object_bbx_center = []
            object_bbx_mask = []
            object_ids = []
            processed_lidar_list = []
            processed_lidar_paint_list = []
            image_inputs_list = []
            # used to record different scenario
            record_len = []
            label_dict_list = []
            lidar_pose_list = []
            origin_lidar = []

            fading_gain = []
            SNR_db = []

            # pairwise transformation matrix
            k_pairwise_t_matrix_list = []

            # disconet
            teacher_processed_lidar_list = []

            early_fusion_processed_lidar_list = []
            ### 2022.10.10 single gt ####
            if self.supervise_single:
                pos_equal_one_single = []
                neg_equal_one_single = []
                targets_single = []
                object_bbx_center_single = []
                object_bbx_mask_single = []

            for i in range(len(batch)):
                ego_dict = batch[i]['ego']
                fading_gain.append(ego_dict['fading_gain_list'])
                SNR_db.append(ego_dict['SNR_db_list'])
                object_bbx_center.append(ego_dict['object_bbx_center'])
                object_bbx_mask.append(ego_dict['object_bbx_mask'])
                object_ids.append(ego_dict['object_ids'])
                lidar_pose_list.append(ego_dict['k_lidar_poses']) # ego_dict['k_lidar_pose'] is np.ndarray [N_cav, k, 6]
                if self.load_lidar_file:
                    processed_lidar_list.append(ego_dict['k_processed_lidar'])
                    processed_lidar_paint_list.append(ego_dict['processed_lidar_paint'])
                if self.load_camera_file:
                    image_inputs_list.append(ego_dict['image_inputs']) # different cav_num, ego_dict['image_inputs'] is dict.
                
                #####################################################################
                early_fusion_processed_lidar_list.append(ego_dict['early_fusion_processed_features'])
                #####################################################################
                
                record_len.append(ego_dict['cav_num'])
                label_dict_list.append(ego_dict['label_dict'])
                k_pairwise_t_matrix_list.append(ego_dict['k_pairwise_t_matrix'])

                if self.visualize:
                    origin_lidar.append(ego_dict['origin_lidar'])

                if self.kd_flag:
                    teacher_processed_lidar_list.append(ego_dict['teacher_processed_lidar'])

                ### 2022.10.10 single gt ####
                if self.supervise_single:
                    pos_equal_one_single.append(ego_dict['single_label_dict_torch']['pos_equal_one'])
                    neg_equal_one_single.append(ego_dict['single_label_dict_torch']['neg_equal_one'])
                    targets_single.append(ego_dict['single_label_dict_torch']['targets'])
                    object_bbx_center_single.append(ego_dict['single_object_bbx_center_torch'])
                    object_bbx_mask_single.append(ego_dict['single_object_bbx_mask_torch'])


            # convert to numpy, (B, max_num, 7)
            object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
            object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

            if self.load_lidar_file:
                early_fusion_merged_feature_dict = merge_features_to_dict(early_fusion_processed_lidar_list)
                early_fusion_processed_lidar_torch_dict = self.pre_processor.collate_batch(early_fusion_merged_feature_dict)  
                output_dict['ego'].update({
                                        'early_fusion_processed_lidar':early_fusion_processed_lidar_torch_dict
                                        }) 
                                        
                merged_feature_dict = merge_features_to_dict(processed_lidar_list)
                merged_feature_paint_dict = merge_features_to_dict(processed_lidar_paint_list)
                processed_lidar_paint_torch_dict = \
                    self.pre_processor.collate_batch(merged_feature_paint_dict)
                for i in range(self.k):
                    merged_feature_dict[i] = merge_features_to_dict(merged_feature_dict[i])
                    merged_feature_dict[i] = self.pre_processor.collate_batch(merged_feature_dict[i])
                
                output_dict['ego'].update({'k_processed_lidar': merged_feature_dict})
                '''
                k:
                    'voxel_features': [B * cav_num * voxel_num, 32, 4]
                    'voxel_coords':  [B * cav_num * voxel_num, 3]
                    'voxel_num_points': [B * cav_num * voxel_num]
                '''
                output_dict['ego'].update({'processed_lidar': merged_feature_dict[0]})

            if self.load_camera_file:
                merged_image_inputs_dict = merge_features_to_dict(image_inputs_list, merge='cat')

                output_dict['ego'].update({'image_inputs': merged_image_inputs_dict})
            
            record_len = torch.from_numpy(np.array(record_len, dtype=int))
            lidar_pose = torch.from_numpy(np.concatenate(lidar_pose_list, axis=0)) # [BxN_cav, k, 6]
            fading_gain = torch.from_numpy(np.concatenate(fading_gain, axis=0)) # (B*cav_num)
            SNR_db = torch.from_numpy(np.concatenate(SNR_db, axis=0)) # (B*cav_num)
            label_torch_dict = \
                self.post_processor.collate_batch(label_dict_list)

            # for centerpoint
            label_torch_dict.update({'object_bbx_center': object_bbx_center,
                                     'object_bbx_mask': object_bbx_mask})

            # (B, max_cav)
            k_pairwise_t_matrix = torch.from_numpy(np.array(k_pairwise_t_matrix_list)) # [B, k, L, L, 4, 4]

            # add pairwise_t_matrix to label dict
            label_torch_dict['pairwise_t_matrix'] = k_pairwise_t_matrix[:, 0, :, :, :, :] # [B, L, L, 4, 4]
            label_torch_dict['record_len'] = record_len # (B,)
            

            # object id is only used during inference, where batch size is 1.
            # so here we only get the first element.
            output_dict['ego'].update({'object_bbx_center': object_bbx_center,
                                    'object_bbx_mask': object_bbx_mask,
                                    'processed_lidar_paint': processed_lidar_paint_torch_dict,
                                    'record_len': record_len,
                                    'label_dict': label_torch_dict,
                                    'object_ids': object_ids[0],
                                    'k_pairwise_t_matrix': k_pairwise_t_matrix,
                                    'pairwise_t_matrix': k_pairwise_t_matrix[:, 0, :, :, :, :],
                                    'k_lidar_poses': lidar_pose,
                                    'lidar_pose': lidar_pose[0],
                                    'fading_gain': fading_gain,
                                    'SNR_db': SNR_db,
                                    'anchor_box': self.anchor_box_torch})


            if self.visualize:
                origin_lidar = \
                    np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
                origin_lidar = torch.from_numpy(origin_lidar)
                output_dict['ego'].update({'origin_lidar': origin_lidar})

            if self.kd_flag:
                teacher_processed_lidar_torch_dict = \
                    self.pre_processor.collate_batch(teacher_processed_lidar_list)
                output_dict['ego'].update({'teacher_processed_lidar':teacher_processed_lidar_torch_dict})


            if self.supervise_single:
                output_dict['ego'].update({
                    "label_dict_single":{
                            "pos_equal_one": torch.cat(pos_equal_one_single, dim=0),
                            "neg_equal_one": torch.cat(neg_equal_one_single, dim=0),
                            "targets": torch.cat(targets_single, dim=0),
                            # for centerpoint
                            "object_bbx_center_single": torch.cat(object_bbx_center_single, dim=0),
                            "object_bbx_mask_single": torch.cat(object_bbx_mask_single, dim=0)
                        },
                    "object_bbx_center_single": torch.cat(object_bbx_center_single, dim=0),
                    "object_bbx_mask_single": torch.cat(object_bbx_mask_single, dim=0)
                })


            return output_dict

        def collate_batch_test(self, batch):
            assert len(batch) <= 1, "Batch size 1 is required during testing!"
            output_dict = self.collate_batch_train(batch)
            if output_dict is None:
                return None

            # check if anchor box in the batch
            if batch[0]['ego']['anchor_box'] is not None:
                output_dict['ego'].update({'anchor_box':
                    self.anchor_box_torch})

            # save the transformation matrix (4, 4) to ego vehicle
            # transformation is only used in post process (no use.)
            # we all predict boxes in ego coord.
            transformation_matrix_torch = \
                torch.from_numpy(np.identity(4)).float()
            transformation_matrix_clean_torch = \
                torch.from_numpy(np.identity(4)).float()

            output_dict['ego'].update({'transformation_matrix':
                                        transformation_matrix_torch,
                                        'transformation_matrix_clean':
                                        transformation_matrix_clean_torch,})

            output_dict['ego'].update({
                "sample_idx": batch[0]['ego']['sample_idx'],
                "cav_id_list": batch[0]['ego']['cav_id_list']
            })

            return output_dict


        def post_process(self, data_dict, output_dict):
            """
            Process the outputs of the model to 2D/3D bounding box.

            Parameters
            ----------
            data_dict : dict
                The dictionary containing the origin input data of model.

            output_dict :dict
                The dictionary containing the output of the model.

            Returns
            -------
            pred_box_tensor : torch.Tensor
                The tensor of prediction bounding box after NMS.
            gt_box_tensor : torch.Tensor
                The tensor of gt bounding box.
            """
            pred_box_tensor, pred_score = \
                self.post_processor.post_process(data_dict, output_dict)
            gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

            return pred_box_tensor, pred_score, gt_box_tensor

        @staticmethod
        def object_all_inside_points(points,object_dict):
            points_xyz = points[:, :3]
            all_inside_points = []

            corner_points = np.array([obj[0] for obj in object_dict.values()])
            expanded_points = np.expand_dims(points_xyz, axis=1)
            inside_mask = box_utils.is_point_inside_any_box(expanded_points, corner_points)
            inside_indices = np.any(inside_mask, axis=1)
            all_inside_points = points[inside_indices]

            return all_inside_points

        @staticmethod
        def object_all_outside_points(points,object_dict):
            points_xyz = points[:, :3]

            corner_points = np.array([obj[0] for obj in object_dict.values()])
            expanded_points = np.expand_dims(points_xyz, axis=1)
            inside_mask = box_utils.is_point_inside_any_box(expanded_points, corner_points)
            inside_indices = np.any(inside_mask, axis=1)
            all_outside_points = points[~inside_indices]

            return all_outside_points

    return IntermediateFusionDataset


