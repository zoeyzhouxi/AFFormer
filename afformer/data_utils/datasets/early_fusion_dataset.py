# early fusion dataset
import torch
import numpy as np
from afformer.utils.pcd_utils import downsample_lidar_minimum
import math
from collections import OrderedDict

from afformer.utils import box_utils
from afformer.utils.common_utils import merge_features_to_dict
from afformer.data_utils.post_processor import build_postprocessor
from afformer.data_utils.pre_processor import build_preprocessor
from afformer.hypes_yaml.yaml_utils import load_yaml
from afformer.utils.pcd_utils import \
    mask_points_by_range, mask_ego_points, shuffle_points, \
    downsample_lidar_minimum
from afformer.utils.transformation_utils import x1_to_x2
from afformer.utils.heter_utils import AgentSelector


def getEarlyFusionDataset(cls):
    class EarlyFusionDataset(cls):
        """
        This dataset is used for early fusion, where each CAV transmit the raw
        point cloud to the ego vehicle.
        """
        def __init__(self, params, visualize, train=True, mode=None):
            super(EarlyFusionDataset, self).__init__(params, visualize, train, mode)
            self.supervise_single = True if ('supervise_single' in params['model']['args'] and params['model']['args']['supervise_single']) \
                                        else False
            assert self.supervise_single is False

            self.proj_first = False if 'proj_first' not in params['fusion']['args'] \
                                        else params['fusion']['args']['proj_first']
            # self.anchor_box = self.post_processor.generate_anchor_box()
            # self.anchor_box_torch = torch.from_numpy(self.anchor_box)

            self.heterogeneous = False
            if 'heter' in params:
                self.heterogeneous = True
                self.selector = AgentSelector(params['heter'], self.max_cav)

        def __getitem__(self, idx):
            base_data_dict = self.retrieve_base_data(idx) # ['4740', '-1', '4722']
            

            processed_data_dict = OrderedDict()
            processed_data_dict['ego'] = {}

            ego_id = -1
            k_ego_lidar_pose = [] 
            # first find the ego vehicle's lidar pose
            for cav_id, cav_content in base_data_dict.items(): # ['ego', 'k_frames']
                if cav_content['ego']:
                    ego_id = cav_id
                    assert cav_id == list(base_data_dict.keys())[0], "The first element in the OrderedDict must be ego"
                    assert ego_id != -1
                
                    for i, content in cav_content['k_frames'].items(): # [0, 1, 2]
                        k_ego_lidar_pose.append(content['params']['lidar_pose']) # content['timestamp', 'params', 'lidar_np']
                        assert len(content['params']['lidar_pose']) > 0
                    break
            # content['params'].keys(): ['RSU', 'ego_speed', 'lidar_pose', 'plan_trajectory', 'predicted_ego_pos', 'true_ego_pos', 'vehicles']

            k_projected_lidar_stack = []
            object_stack = []
            object_id_stack = []

            # loop over all CAVs to process information
            for cav_id, selected_cav_base in base_data_dict.items():
                distance = \
                    math.sqrt((selected_cav_base['k_frames'][0]['params']['lidar_pose'][0] -
                            k_ego_lidar_pose[0][0]) ** 2 + (
                                    selected_cav_base['k_frames'][0]['params'][
                                        'lidar_pose'][1] - k_ego_lidar_pose[0][
                                        1]) ** 2)
                # if distance is too far, we will just skip this agent
                if distance > self.params['comm_range']:
                    continue

                selected_cav_processed = \
                self.get_item_single_car(selected_cav_base, k_ego_lidar_pose)
                # ['object_bbx_center', 'object_ids', 'k_projected_lidar']

                # all these lidar and object coordinates are projected to ego already.
                k_projected_lidar_stack.append(
                    selected_cav_processed['k_projected_lidar'])

                object_stack.append(selected_cav_processed['object_bbx_center'])
                object_id_stack += selected_cav_processed['object_ids']

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

            
            # # convert list to numpy array, (N, K, 4)
            # k_projected_lidar_stack = np.vstack(k_projected_lidar_stack)
            
            lidar_dict = OrderedDict()
            for i in range(self.k):
                k_projected_lidar_list = []
                for j in range(len(k_projected_lidar_stack)):
                    k_projected_lidar_list.append(k_projected_lidar_stack[j][i])
                k_projected_lidar_list = np.vstack(k_projected_lidar_list)
                # data augmentation
                if self.mode == 'train':
                    k_projected_lidar_list, object_bbx_center, mask = \
                        self.augment(k_projected_lidar_list, object_bbx_center, mask)
                # we do lidar filtering in the stacked lidar
                k_projected_lidar_list = mask_points_by_range(k_projected_lidar_list,
                                                        self.params['preprocess'][
                                                            'cav_lidar_range'])
                # pre-process the lidar to voxel/bev/downsampled lidar
                lidar_dict[i] = self.pre_processor.preprocess(k_projected_lidar_list)
            
            # augmentation may remove some of the bbx out of range
            if self.mode == 'train':
                object_bbx_center_valid = object_bbx_center[mask == 1]
                object_bbx_center_valid, range_mask = \
                box_utils.mask_boxes_outside_range_numpy(object_bbx_center_valid,
                                                        self.params['preprocess'][
                                                            'cav_lidar_range'],
                                                        self.params['postprocess'][
                                                            'order'],
                                                        return_mask=True
                                                        )
                mask[object_bbx_center_valid.shape[0]:] = 0
                object_bbx_center[:object_bbx_center_valid.shape[0]] = \
                    object_bbx_center_valid
                object_bbx_center[object_bbx_center_valid.shape[0]:] = 0
                unique_indices = list(np.array(unique_indices)[range_mask])

            # generate the anchor boxes
            anchor_box = self.post_processor.generate_anchor_box()

            # generate targets label
            label_dict = \
                self.post_processor.generate_label(
                    gt_box_center=object_bbx_center,
                    anchors=anchor_box,
                    mask=mask)

            processed_data_dict['ego'].update(
                {'object_bbx_center': object_bbx_center,
                'object_bbx_mask': mask,
                'object_ids': [object_id_stack[i] for i in unique_indices],
                'anchor_box': anchor_box,
                'processed_lidar': lidar_dict,
                'label_dict': label_dict})

            if self.visualize:
                processed_data_dict['ego'].update({'origin_lidar':
                                                    k_projected_lidar_stack[0][0]})
            
            return processed_data_dict #['object_bbx_center', 'object_bbx_mask', 'object_ids', 'anchor_box', 'processed_lidar', 'label_dict']

        def get_item_single_car(self, selected_cav_base, k_ego_lidar_pose):
            """
            Project the lidar and bbx to ego space first, and then do clipping.

            Parameters
            ----------
            selected_cav_base : dict
                The dictionary contains a single CAV's raw information. ['ego', 'k_frames']
            k_ego_lidar_pose : list 
                The ego vehicle lidar pose of k frames under world coordinate.

            Returns
            -------
            selected_cav_processed : dict
                The dictionary contains the cav's processed information.
            """
            selected_cav_processed = {}

            # currently retrieve objects under ego coordinates
            object_bbx_center, object_bbx_mask, object_ids = \
                self.generate_object_center([selected_cav_base], k_ego_lidar_pose[0])

            k_projected_lidar = []
            for i in range(self.k):
                # calculate the transformation matrix
                transformation_matrix_i = \
                    x1_to_x2(selected_cav_base['k_frames'][i]['params']['lidar_pose'],
                        k_ego_lidar_pose[i]) # T_ego_cav
                # lidar data
                lidar_np_i = selected_cav_base['k_frames'][i]['lidar_np']
                lidar_np_i = shuffle_points(lidar_np_i)
                # remove points that hit itself
                lidar_np_i = mask_ego_points(lidar_np_i)
                # project the lidar to ego space
                lidar_np_i[:, :3] = \
                box_utils.project_points_by_matrix_torch(lidar_np_i[:, :3],
                                                        transformation_matrix_i)
                k_projected_lidar.append(lidar_np_i)
        
            selected_cav_processed.update(
                {'object_bbx_center': object_bbx_center[object_bbx_mask == 1],
                'object_ids': object_ids,
                'k_projected_lidar': k_projected_lidar})

            return selected_cav_processed

        def collate_batch_test(self, batch):
            """
            Customized collate function for pytorch dataloader during testing
            for late fusion dataset.

            Parameters
            ----------
            batch : dict

            Returns
            -------
            batch : dict
                Reformatted batch.
            """
            # currently, we only support batch size of 1 during testing
            assert len(batch) <= 1, "Batch size 1 is required during testing!"

            batch = batch[0] # only ego

            output_dict = {}

            for cav_id, cav_content in batch.items():
                output_dict.update({cav_id: {}})
                # shape: (1, max_num, 7)
                object_bbx_center = \
                    torch.from_numpy(np.array([cav_content['object_bbx_center']]))
                object_bbx_mask = \
                    torch.from_numpy(np.array([cav_content['object_bbx_mask']]))
                object_ids = cav_content['object_ids']

                # the anchor box is the same for all bounding boxes usually, thus
                # we don't need the batch dimension.
                if cav_content['anchor_box'] is not None:
                    output_dict[cav_id].update({'anchor_box':
                        torch.from_numpy(np.array(
                            cav_content[
                                'anchor_box']))})
                if self.visualize:
                    origin_lidar = [cav_content['origin_lidar']]

                # processed lidar dictionary
                # processed_lidar_torch_dict = \
                #     self.pre_processor.collate_batch(
                #         [cav_content['processed_lidar']])

                processed_lidar_torch_dict = {}
                for i in range(self.k):
                    processed_lidar_torch_dict[i] = self.pre_processor.collate_batch(
                        [cav_content['processed_lidar'][i]])
                # label dictionary
                label_torch_dict = \
                    self.post_processor.collate_batch([cav_content['label_dict']])

                # save the transformation matrix (4, 4) to ego vehicle
                transformation_matrix_torch = \
                    torch.from_numpy(np.identity(4)).float()
                transformation_matrix_clean_torch = \
                    torch.from_numpy(np.identity(4)).float()
               
                output_dict[cav_id].update({'object_bbx_center': object_bbx_center,
                                            'object_bbx_mask': object_bbx_mask,
                                            'processed_lidar': processed_lidar_torch_dict,
                                            'label_dict': label_torch_dict,
                                            'object_ids': object_ids,
                                            'transformation_matrix': transformation_matrix_torch,
                                            'transformation_matrix_clean': transformation_matrix_clean_torch})

                if self.visualize:
                    origin_lidar = \
                        np.array(
                            downsample_lidar_minimum(pcd_np_list=origin_lidar))
                    origin_lidar = torch.from_numpy(origin_lidar)
                    output_dict[cav_id].update({'origin_lidar': origin_lidar})

            return output_dict
        
        def collate_batch_train(self, batch):
            # Intermediate fusion is different the other two
            output_dict = {'ego': {}}

            object_bbx_center = []
            object_bbx_mask = []
            object_ids = []
            processed_lidar_list = []
            image_inputs_list = []
            # used to record different scenario
            label_dict_list = []
            origin_lidar = []
            
            # heterogeneous
            lidar_agent_list = []
            
            ### 2022.10.10 single gt ####
            if self.supervise_single:
                pos_equal_one_single = []
                neg_equal_one_single = []
                targets_single = []

            for i in range(len(batch)):
                ego_dict = batch[i]['ego']
                object_bbx_center.append(ego_dict['object_bbx_center'])
                object_bbx_mask.append(ego_dict['object_bbx_mask'])
                object_ids.append(ego_dict['object_ids'])
                if self.load_lidar_file:
                    processed_lidar_list.append(ego_dict['processed_lidar'])
                if self.load_camera_file:
                    image_inputs_list.append(ego_dict['image_inputs']) # different cav_num, ego_dict['image_inputs'] is dict.
                
                label_dict_list.append(ego_dict['label_dict'])

                if self.visualize:
                    origin_lidar.append(ego_dict['origin_lidar'])

                ### 2022.10.10 single gt ####
                if self.supervise_single:
                    pos_equal_one_single.append(ego_dict['single_label_dict_torch']['pos_equal_one'])
                    neg_equal_one_single.append(ego_dict['single_label_dict_torch']['neg_equal_one'])
                    targets_single.append(ego_dict['single_label_dict_torch']['targets'])

                # heterogeneous
                if self.heterogeneous:
                    lidar_agent_list.append(ego_dict['lidar_agent'])

            # convert to numpy, (B, max_num, 7)
            object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
            object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

            if self.load_lidar_file:
                merged_feature_dict = merge_features_to_dict(processed_lidar_list)
               
                for i in range(self.k):
                    merged_feature_dict[i] = merge_features_to_dict(merged_feature_dict[i])
                    merged_feature_dict[i] = self.pre_processor.collate_batch(merged_feature_dict[i])
                output_dict['ego'].update({'processed_lidar': merged_feature_dict})

            if self.load_camera_file:
                merged_image_inputs_dict = merge_features_to_dict(image_inputs_list, merge='cat')

                if self.heterogeneous:
                    camera_agent = 1 - lidar_agent
                    camera_agent_idx = camera_agent.nonzero()[0].tolist()
                    if sum(camera_agent) != 0:
                        for k, v in merged_image_inputs_dict.items(): # 'imgs' 'rots' 'trans' ...
                            merged_image_inputs_dict[k] = torch.stack([v[index] for index in camera_agent_idx])
                            
                if not self.heterogeneous or (self.heterogeneous and sum(camera_agent) != 0):
                    output_dict['ego'].update({'image_inputs': merged_image_inputs_dict})
            
            label_torch_dict = \
                self.post_processor.collate_batch(label_dict_list)

            # for centerpoint
            label_torch_dict.update({'object_bbx_center': object_bbx_center,
                                    'object_bbx_mask': object_bbx_mask})

            # add pairwise_t_matrix to label dict

            # object id is only used during inference, where batch size is 1.
            # so here we only get the first element.
            output_dict['ego'].update({'object_bbx_center': object_bbx_center,
                                    'object_bbx_mask': object_bbx_mask,
                                    'label_dict': label_torch_dict,
                                    'object_ids': object_ids[0]})


            if self.visualize:
                origin_lidar = \
                    np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
                origin_lidar = torch.from_numpy(origin_lidar)
                output_dict['ego'].update({'origin_lidar': origin_lidar})

            if self.supervise_single:
                output_dict['ego'].update({
                    "label_dict_single" : 
                        {"pos_equal_one": torch.cat(pos_equal_one_single, dim=0),
                        "neg_equal_one": torch.cat(neg_equal_one_single, dim=0),
                        "targets": torch.cat(targets_single, dim=0)}
                })

            if self.heterogeneous:
                output_dict['ego'].update({
                    "lidar_agent_record": torch.from_numpy(np.concatenate(lidar_agent_list)) # [0,1,1,0,1...]
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

    return EarlyFusionDataset

