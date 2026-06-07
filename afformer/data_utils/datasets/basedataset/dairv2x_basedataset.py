import os
from collections import OrderedDict
import cv2
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import random
import afformer.utils.pcd_utils as pcd_utils
from afformer.data_utils.augmentor.data_augmentor import DataAugmentor
from afformer.hypes_yaml.yaml_utils import load_yaml
from afformer.utils.pcd_utils import downsample_lidar_minimum
from afformer.utils.camera_utils import load_camera_data, load_intrinsic_DAIR_V2X
from afformer.utils.common_utils import read_json
from afformer.utils.transformation_utils import tfm_to_pose, rot_and_trans_to_trasnformation_matrix
from afformer.utils.transformation_utils import veh_side_rot_and_trans_to_trasnformation_matrix
from afformer.utils.transformation_utils import inf_side_rot_and_trans_to_trasnformation_matrix
from afformer.data_utils.pre_processor import build_preprocessor
from afformer.data_utils.post_processor import build_postprocessor

def id_to_str(id, digits=6):
    result = ""
    for i in range(digits):
        result = str(id % 10) + result
        id //= 10
    return result

class DAIRV2XBaseDataset(Dataset):
    def __init__(self, params, visualize, train=True, mode=None):
        self.params = params
        self.visualize = visualize
        self.train = train
        self.mode = mode

        self.pre_processor = build_preprocessor(params["preprocess"], train)
        self.post_processor = build_postprocessor(params["postprocess"], train)
        self.post_processor.generate_gt_bbx = self.post_processor.generate_gt_bbx_by_iou
        self.data_augmentor = DataAugmentor(params['data_augment'],
                                            train)
        # number of input frames
        if 'num_sweep_frames' in params:
            self.k = params['num_sweep_frames']
        else:
            self.k = 1

        if 'clip_pc' in params['fusion']['args'] and params['fusion']['args']['clip_pc']:
            self.clip_pc = True
        else:
            self.clip_pc = False

        if 'train_params' not in params or 'max_cav' not in params['train_params']:
            self.max_cav = 2
        else:
            self.max_cav = params['train_params']['max_cav']

        self.load_lidar_file = True if 'lidar' in params['input_source'] or self.visualize else False
        self.load_camera_file = True if 'camera' in params['input_source'] else False
        self.load_depth_file = True if 'depth' in params['input_source'] else False

        assert self.load_depth_file is False

        self.label_type = params['label_type'] # 'lidar' or 'camera'
        self.generate_object_center = self.generate_object_center_lidar if self.label_type == "lidar" \
                                                    else self.generate_object_center_camera

        if self.load_camera_file:
            self.data_aug_conf = params["fusion"]["args"]["data_aug_conf"]

        self.root_dir = params['data_dir']

        # 读取cooperative/data_info.json文件，返回一个list
        co_data_info = read_json(os.path.join(self.root_dir, 'cooperative/data_info.json'))
        '''
        co_data_info的格式如下：
        {
            "infrastructure_image_path": "infrastructure-side/image/infrastructure_image/000000.jpg",
            "infrastructure_pointcloud_path": "infrastructure-side/pointcloud/infrastructure_pointcloud/000000.pcd",
            "vehicle_image_path": "vehicle-side/image/vehicle_image/000000.jpg",
            "vehicle_pointcloud_path": "vehicle-side/pointcloud/vehicle_pointcloud/000000.pcd",
            "cooperative_label_path": "vehicle-side/label/cooperative/000000.json"
        }
        {}
        '''
        # 遍历co_data_info中的每个元素，将time step作为co_data的key，frame_info作为value
        # 这里co_data_info中的每个元素是一个字典，每个字典的key是vehicle_image_path..，value是对应的path
        self.co_id2info = OrderedDict()
        self.infid2vehid = OrderedDict()
        for frame_info in co_data_info:
            veh_frame_id = frame_info['vehicle_pointcloud_path'].split("/")[-1].replace(".pcd", "") # time step
            self.co_id2info[veh_frame_id] = frame_info
            inf_frame_id = frame_info['infrastructure_pointcloud_path'].split("/")[-1].replace(".pcd", "")
            self.infid2vehid[inf_frame_id] = veh_frame_id
        '''
        self.co_id2info的格式如下：
        {
            "000000": {
                "infrastructure_image_path": "infrastructure-side/image/infrastructure_image/000000.jpg",
                "infrastructure_pointcloud_path": "infrastructure-side/pointcloud/infrastructure_pointcloud/000000.pcd",
                "vehicle_image_path": "vehicle-side/image/vehicle_image/000000.jpg",
                "vehicle_pointcloud_path": "vehicle-side/pointcloud/vehicle_pointcloud/000000.pcd",
                "cooperative_label_path": "vehicle-side/label/cooperative/000000.json"
            }
        }
        '''
        # import pdb; pdb.set_trace()
        inf_data_info = read_json(os.path.join(self.root_dir, 'infrastructure-side/data_info.json'))
        self.inf_id2info = OrderedDict()
        for frame_info in inf_data_info:
            inf_frame_id = frame_info['pointcloud_path'].split("/")[-1].replace(".pcd", "")
            self.inf_id2info[inf_frame_id] = frame_info
            # import pdb; pdb.set_trace()

        vehicle_data_info = read_json(os.path.join(self.root_dir, 'vehicle-side/data_info.json'))
        self.veh_id2info = OrderedDict()
        for frame_info in vehicle_data_info:
            veh_frame_id = frame_info['pointcloud_path'].split("/")[-1].replace(".pcd", "")
            self.veh_id2info[veh_frame_id] = frame_info

        # 这里split_dirr是一个json文件！--> 代表一个split
        if self.train:
            split_dir = params['root_dir']
        else:
            split_dir = params['validate_dir']

        self.split_info = read_json(split_dir) # 读取split_dir中的json文件，返回一个list
        self.valid_veh_frame_ids = []
        for veh_frame_id in self.split_info:
            if self.is_valid_id(veh_frame_id):
                self.valid_veh_frame_ids.append(veh_frame_id)

        


    def reinitialize(self):
        pass

    def is_valid_id(self, veh_frame_id):
        """
        Given veh_frame_id, determine whether there is a corresponding inf_frame that meets the "k-1 historical frames" requirement.

        Parameters
        ----------
        veh_frame_id : 05d
            Vehicle frame id

        Returns
        -------
        bool valud
            True means there is a corresponding road-side frame.
        """
    
        frame_info = self.co_id2info[veh_frame_id]
        inf_frame_id = frame_info['infrastructure_pointcloud_path'].split("/")[-1].replace(".pcd", "")
        # cur_inf_info = self.inf_id2info[inf_frame_id]
        # if (int(inf_frame_id) - self.k + 1 < int(cur_inf_info["batch_start_id"])):
        #     return False
        for i in range(self.k): 
            past_veh_frame_id = id_to_str(int(veh_frame_id) - i)
            if past_veh_frame_id not in self.co_id2info.keys():
                return False
            past_inf_frame_id = id_to_str(int(inf_frame_id) - i) 
            if past_inf_frame_id not in self.infid2vehid.keys():
                return False

        return True

    def get_vehicle_trans(self, veh_frame_id):
        lidar_to_novatel = read_json(os.path.join(self.root_dir,'vehicle-side/calib/lidar_to_novatel/'+str(veh_frame_id)+'.json'))
        novatel_to_world = read_json(os.path.join(self.root_dir,'vehicle-side/calib/novatel_to_world/'+str(veh_frame_id)+'.json'))
        transformation_matrix = veh_side_rot_and_trans_to_trasnformation_matrix(lidar_to_novatel, novatel_to_world)
        return tfm_to_pose(transformation_matrix)

    def get_inf_trans(self, inf_frame_id, system_error_offset):
        virtuallidar_to_world = read_json(os.path.join(self.root_dir,'infrastructure-side/calib/virtuallidar_to_world/'+str(inf_frame_id)+'.json'))
        transformation_matrix = inf_side_rot_and_trans_to_trasnformation_matrix(virtuallidar_to_world, system_error_offset)
        return tfm_to_pose(transformation_matrix)

    def retrieve_base_data(self, idx):
        """
        Given the index, return the corresponding data.
        NOTICE!
        It is different from Intermediate Fusion and Early Fusion
        Label is not cooperative and loaded for both veh side and inf side.
        Parameters
        ----------
        idx : int
            Index given by dataloader.
        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
        """
        curr_veh_frame_id = self.valid_veh_frame_ids[idx]
        frame_info = self.co_id2info[curr_veh_frame_id]
        curr_inf_frame_id = frame_info['infrastructure_pointcloud_path'].split("/")[-1].replace(".pcd", "")
        system_error_offset = frame_info["system_error_offset"]
        data = OrderedDict()
        # 这里data[0]是vehicle side，data[1]是infrastructure side
        for i in range(2):
            data[i] = OrderedDict()
            data[i]['ego'] = True if i == 0 else False
            data[i]['scenario_info'] = {'scenario_type': 'V2I-LOS', 'V2I_K_dB': random.randint(5, 9), 'V2V_K_dB': None}
            data[i]['k_frames'] = OrderedDict()
            for j in range(self.k):
                data[i]['k_frames'][j] = {}
                k_veh_frame_id = id_to_str(int(curr_veh_frame_id)-j)
                k_inf_frame_id = id_to_str(int(curr_inf_frame_id)-j)
                data[i]['k_frames'][j]['frame_id'] = k_veh_frame_id if i == 0 else k_inf_frame_id
                data[i]['k_frames'][j]['timestamp'] = self.veh_id2info[k_veh_frame_id]['pointcloud_timestamp'] if i == 0 else self.inf_id2info[k_inf_frame_id]['pointcloud_timestamp']

                data[i]['k_frames'][j]['params'] = OrderedDict()
                data[i]['k_frames'][j]['params']['vehicles']=  read_json(os.path.join(self.root_dir, self.co_id2info[k_veh_frame_id]['cooperative_label_path'])) if i == 0 else []
                data[i]['k_frames'][j]['params']['lidar_pose'] = self.get_vehicle_trans(k_veh_frame_id) if i == 0 else self.get_inf_trans(k_inf_frame_id, system_error_offset)
    
                if self.load_camera_file:
                    data[i]['k_frames'][j]['camera_data'] = load_camera_data([os.path.join(self.root_dir, self.co_id2info[k_veh_frame_id]["vehicle_image_path"])]) if i == 0 else load_camera_data([os.path.join(self.root_dir,self.co_id2info[k_veh_frame_id]["infrastructure_image_path"])])

                    data[i]['k_frames'][j]['params']['camera0'] = OrderedDict()
                    data[i]['k_frames'][j]['params']['camera0']['extrinsic'] = rot_and_trans_to_trasnformation_matrix( \
                                                    read_json(os.path.join(self.root_dir, 'vehicle-side/calib/lidar_to_camera/'+str(k_veh_frame_id)+'.json'))) if i == 0 \
                                                    else rot_and_trans_to_trasnformation_matrix( \
                                                    read_json(os.path.join(self.root_dir, 'infrastructure-side/calib/virtuallidar_to_camera/'+str(k_inf_frame_id)+'.json')))

                    data[i]['k_frames'][j]['params']['camera0']['intrinsic'] = load_intrinsic_DAIR_V2X( \
                                                    read_json(os.path.join(self.root_dir, 'vehicle-side/calib/camera_intrinsic/'+str(k_veh_frame_id)+'.json'))) if i == 0 \
                                                    else load_intrinsic_DAIR_V2X( \
                                                    read_json(os.path.join(self.root_dir, 'infrastructure-side/calib/camera_intrinsic/'+str(k_inf_frame_id)+'.json')))


                if self.load_lidar_file or self.visualize:
                    data[i]['k_frames'][j]['lidar_np'], _ = pcd_utils.read_pcd(os.path.join(self.root_dir,self.co_id2info[k_veh_frame_id]["vehicle_pointcloud_path"])) if i == 0 \
                                                            else pcd_utils.read_pcd(os.path.join(self.root_dir,self.co_id2info[k_veh_frame_id]["infrastructure_pointcloud_path"]))

        return data


    def __len__(self):
        return len(self.valid_veh_frame_ids)

    def __getitem__(self, idx):
        pass


    def generate_object_center_lidar(self,
                               cav_contents,
                               reference_lidar_pose):
        """
        reference lidar 's coordinate 
        """
        # for cav_content in cav_contents:
        #     cav_content['params']['vehicles'] = cav_content['params']['vehicles_all']
        return self.post_processor.generate_object_center_dairv2x(cav_contents,
                                                        reference_lidar_pose)

    def generate_object_center_camera(self,
                               cav_contents,
                               reference_lidar_pose):
        """
        reference lidar 's coordinate 
        """
        for cav_content in cav_contents:
            cav_content['params']['vehicles'] = cav_content['params']['vehicles_front']
        return self.post_processor.generate_object_center_dairv2x(cav_contents,
                                                        reference_lidar_pose)
                                                        
    ### Add new func for single side
    def generate_object_center_single(self,
                               cav_contents,
                               reference_lidar_pose,
                               **kwargs):
        """
        veh or inf 's coordinate
        """
        suffix = "_single"
        for cav_content in cav_contents:
            cav_content['params']['vehicles_single'] = \
                    cav_content['params']['vehicles_single_front'] if self.label_type == 'camera' else \
                    cav_content['params']['vehicles_single_all']
        return self.post_processor.generate_object_center_dairv2x_single(cav_contents, suffix)

    def get_ext_int(self, params, camera_id):
        lidar_to_camera = params["camera%d" % camera_id]['extrinsic'].astype(np.float32) # R_cw
        camera_to_lidar = np.linalg.inv(lidar_to_camera) # R_wc
        camera_intrinsic = params["camera%d" % camera_id]['intrinsic'].astype(np.float32
        )
        return camera_to_lidar, camera_intrinsic

    def augment(self, lidar_np, object_bbx_center, object_bbx_mask):
        """
        Given the raw point cloud, augment by flipping and rotation.
        Parameters
        ----------
        lidar_np : np.ndarray
            (n, 4) shape
        object_bbx_center : np.ndarray
            (n, 7) shape to represent bbx's x, y, z, h, w, l, yaw
        object_bbx_mask : np.ndarray
            Indicate which elements in object_bbx_center are padded.
        """
        tmp_dict = {'lidar_np': lidar_np,
                    'object_bbx_center': object_bbx_center,
                    'object_bbx_mask': object_bbx_mask}
        tmp_dict = self.data_augmentor.forward(tmp_dict)

        lidar_np = tmp_dict['lidar_np']
        object_bbx_center = tmp_dict['object_bbx_center']
        object_bbx_mask = tmp_dict['object_bbx_mask']

        return lidar_np, object_bbx_center, object_bbx_mask