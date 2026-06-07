
import os
from collections import OrderedDict
import cv2
import h5py
import random
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import json
import afformer.utils.pcd_utils as pcd_utils
from afformer.data_utils.augmentor.data_augmentor import DataAugmentor
from afformer.hypes_yaml.yaml_utils import load_yaml
from afformer.utils.camera_utils import load_camera_data
from afformer.utils.transformation_utils import x1_to_x2
from afformer.data_utils.pre_processor import build_preprocessor
from afformer.data_utils.post_processor import build_postprocessor
from afformer.utils.pose_utils import get_angle_diff

class OPV2VBaseDataset(Dataset):
    def __init__(self, params, visualize, train=True, mode=None):
        self.params = params
        self.visualize = visualize
        self.train = train
        self.mode = mode
        self.pre_processor = build_preprocessor(params["preprocess"], train)
        self.post_processor = build_postprocessor(params["postprocess"], train)
        self.data_augmentor = DataAugmentor(params['data_augment'],
                                            train)

        # number of input frames
        if 'num_sweep_frames' in params:
            self.k = params['num_sweep_frames']
        else:
            self.k = 1

        # get the root directory of the dataset according to the mode
        if self.mode == 'train':
            root_dir = params['root_dir']
        elif self.mode == 'validate':
            root_dir = params['validate_dir']
        else:
            root_dir = params['test_dir']

        self.root_dir = root_dir 
        
        print("Dataset dir:", root_dir)

        if 'train_params' not in params or \
                'max_cav' not in params['train_params']:
            self.max_cav = 5
        else:
            self.max_cav = params['train_params']['max_cav']

        self.load_lidar_file = True if 'lidar' in params['input_source'] or self.visualize else False
        self.load_camera_file = True if 'camera' in params['input_source'] else False
        self.load_depth_file = True if 'depth' in params['input_source'] else False

        self.label_type = params['label_type'] # 'lidar' or 'camera'
        self.generate_object_center = self.generate_object_center_lidar if self.label_type == "lidar" \
                                            else self.generate_object_center_camera
        self.generate_object_center_single = self.generate_object_center # will it follows 'self.generate_object_center' when 'self.generate_object_center' change? Yes

        if self.load_camera_file:
            self.data_aug_conf = params["fusion"]["args"]["data_aug_conf"]

        # by default, we load lidar, camera and metadata. But users may
        # define additional inputs/tasks
        self.add_data_extension = \
            params['add_data_extension'] if 'add_data_extension' \
                                            in params else []

        if "noise_setting" not in self.params:
            self.params['noise_setting'] = OrderedDict()
            self.params['noise_setting']['add_noise'] = False

        # first load all paths of different scenarios
        self.scenario_folders = sorted([os.path.join(root_dir, x)
                                   for x in os.listdir(root_dir) if
                                   os.path.isdir(os.path.join(root_dir, x))])

        self.reinitialize()
   
    def reinitialize(self):
        # Structure: {scenario_id : {cav_1 : {timestamp1 : {yaml: path,
        # lidar: path, cameras:list of path}}}}
        self.scenario_database = OrderedDict()
        self.len_record = []

        # loop over all scenarios
        for (i, scenario_folder) in enumerate(self.scenario_folders):
            self.scenario_database.update({i: OrderedDict()})

            # at least 1 cav should show up
            cav_list = sorted([x for x in os.listdir(scenario_folder)
                                   if os.path.isdir(
                        os.path.join(scenario_folder, x))])
            assert len(cav_list) > 0

            # roadside unit data's id is always negative, so here we want to
            # make sure they will be in the end of the list as they shouldn't
            # be ego vehicle.
            if int(cav_list[0]) < 0:
                cav_list = cav_list[1:] + [cav_list[0]]

            # get the timestamps of the scenario through the first cav
            ego_path = os.path.join(scenario_folder, cav_list[0])
            # use the frame number as key, the full path as the values
            yaml_files = \
                sorted([os.path.join(ego_path, x)
                        for x in os.listdir(ego_path) if
                        x.endswith('.yaml')])
            timestamps = self.extract_timestamps(yaml_files)
            self.scenario_database[i]['timestamps'] = timestamps

            # classify the scenario type
            data_protocol_yaml_file = os.path.join(scenario_folder,
                                                    'data_protocol.yaml')
            data_protocol = load_yaml(data_protocol_yaml_file)
            scenario_type, V2I_K_dB, V2V_K_dB = self.classify_v2x_scenario_final(data_protocol)
     
            # loop over all CAV data
            for (j, cav_id) in enumerate(cav_list):
                if j > self.max_cav - 1:
                    print('too many cavs reinitialize')
                    break
                self.scenario_database[i][cav_id] = OrderedDict()

                # save all yaml files to the dictionary
                cav_path = os.path.join(scenario_folder, cav_id)

                for timestamp in timestamps:
                    self.scenario_database[i][cav_id][timestamp] = \
                        OrderedDict()
                    yaml_file = os.path.join(cav_path,
                                             timestamp + '.yaml')
                    lidar_file = os.path.join(cav_path,
                                              timestamp + '.pcd')
                    camera_files = self.find_camera_files(cav_path, 
                                                timestamp)
                    depth_files = self.find_camera_files(cav_path, 
                                                timestamp,sensor="depth")

                    self.scenario_database[i][cav_id][timestamp]['yaml'] = \
                        yaml_file
                    self.scenario_database[i][cav_id][timestamp]['lidar'] = \
                        lidar_file
                    self.scenario_database[i][cav_id][timestamp]['cameras'] = \
                        camera_files
                    self.scenario_database[i][cav_id][timestamp]['depths'] = \
                        depth_files

                   # load extra data
                    for file_extension in self.add_data_extension:
                        file_name = \
                            os.path.join(cav_path,
                                         timestamp + '_' + file_extension)

                        self.scenario_database[i][cav_id][timestamp][
                            file_extension] = file_name                  

                # Assume all cavs will have the same timestamps length. Thus
                # we only need to calculate for the first vehicle in the 
                # scene.
                if j == 0:
                    # we regard the agent with the minimum id as the ego
                    self.scenario_database[i][cav_id]['ego'] = True
                    slice_num = len(timestamps) - self.k + 1
                    if not self.len_record:
                        self.len_record.append(slice_num)
                    else:
                        prev_last = self.len_record[-1]
                        self.len_record.append(prev_last + slice_num)
                else:
                    self.scenario_database[i][cav_id]['ego'] = False
                    self.scenario_database[i][cav_id]['scenario_info'] = {'scenario_type': scenario_type, 'V2I_K_dB': V2I_K_dB, 'V2V_K_dB': V2V_K_dB}
                    
            
                
    def classify_v2x_scenario_final(self, data):

        scenario = data.get('scenario', {})
        cav_list = scenario.get('single_cav_list', [])
        rsu_list = scenario.get('rsu_list', [])
        tm_config = data.get('carla_traffic_manager', {})

        # 背景车辆忽略红绿灯比例
        ignore_lights_percentage = tm_config['ignore_lights_percentage']
        global_speed_perc = tm_config['global_speed_perc']

        # 提取平均最大速度
        avg_max_speed = sum(c['behavior']['max_speed'] for c in cav_list) / len(cav_list)
        
        # 提取所有车的偏航角
        yaws = [c['spawn_position'][4] for c in cav_list]
        yaw_diff = get_angle_diff(max(yaws), min(yaws))

        dist_x = [abs(c['spawn_position'][0] - c['destination'][0]) for c in cav_list]
        dist_y = [abs(c['spawn_position'][1] - c['destination'][1]) for c in cav_list]
           
        
        # 提取目的地坐标并去重 (判断是否汇聚于一点)
        dests = set([(round(c['destination'][0], 1), round(c['destination'][1], 1)) for c in cav_list])
        is_converging = len(dests) == 1
        
        
        # --- 2. 判定逻辑 ---  
        # if  ignore_lights_percentage < 0.5 and (global_speed_perc >= 0) \
        #     or any(abs(get_angle_diff(y1, y2) - 90) < 15 for i, y1 in enumerate(yaws) for y2 in yaws[i+1:]) \
        #         or global_distance <3 \
        #         or (ignore_lights_percentage==0 and (is_converging or any(x > 20 and y > 20 for x, y in zip(dist_x, dist_y)))):
        #     return "NLOS"
        # else:
        #     return "LOS"

        if len(rsu_list) > 0:
            if  ignore_lights_percentage < 0.5 and (global_speed_perc >= 0) \
                or any(abs(get_angle_diff(y1, y2) - 90) < 15 for i, y1 in enumerate(yaws) for y2 in yaws[i+1:]):
                return "NLOS", random.randint(5, 9), None # Urban_Intersection
            else:
                return 'LOS', random.randint(12, 18), random.randint(7, 11) # Highway Ramp Merging
        else:
            if avg_max_speed > 40 and yaw_diff < 5:
                return "LOS", None,random.randint(10, 15) # Straight_Freeway
            elif 0 < ignore_lights_percentage < 20:
                return "LOS", None,random.randint(7, 11) # General_Suburban_Road
            elif any(abs(get_angle_diff(y1, y2) - 90) < 15 for i, y1 in enumerate(yaws) for y2 in yaws[i+1:]):
                if is_converging:
                    return "LOS", None, random.randint(6, 10) # Highway_Curve
                # elif ignore_lights_percentage==0:
                #     return "NLOS", None, None # Urban_Junction
                else:
                    return "NLOS", None, None # Highway_Junction
            elif ignore_lights_percentage==100 or avg_max_speed > 25:
                return "LOS", None, random.randint(9, 14) # General_Highway
            elif ignore_lights_percentage==0 \
                and (is_converging or any(x > 20 and y > 20 for x, y in zip(dist_x, dist_y))):
                return "NLOS", None, None # Urban_Junction
            else:
                return 'NLOS', None, None # General_Urban_Road
        
            

    def retrieve_base_data(self, idx):
        """
        Given the index, return the corresponding data.

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
        # we loop the accumulated length list to see get the scenario index
        scenario_index = 0
        for i, ele in enumerate(self.len_record):
            if idx < ele:
                scenario_index = i
                break
        scenario_database = self.scenario_database[scenario_index]

        # check the timestamp index
        timestamp_index = idx if scenario_index == 0 else \
            idx - self.len_record[scenario_index - 1]
        timestamp_index = timestamp_index + self.k - 1

        # retrieve the corresponding timestamp key
        timestamps = scenario_database['timestamps']
        # timestamp_key = timestamps[timestamp_index]


        data = OrderedDict()
        # load files for all CAVs
        for cav_id, cav_content in scenario_database.items():

            if cav_id == 'timestamps':
                continue

            data[cav_id] = OrderedDict()
            data[cav_id]['ego'] = cav_content['ego']

            if cav_content['ego'] is False:
                data[cav_id]['scenario_info'] = cav_content['scenario_info']          
            
            data[cav_id]['k_frames'] = OrderedDict()

            for i in range(self.k):
                data[cav_id]['k_frames'][i] = {}

                # check the timestamp index
                past_i_idx = timestamp_index - i 
                timestamp_key_i = timestamps[past_i_idx]

                data[cav_id]['k_frames'][i]['timestamp'] = timestamp_key_i

                # load param file: json is faster than yaml
                json_file = cav_content[timestamp_key_i]['yaml'].replace("yaml", "json")
                if os.path.exists(json_file):
                    with open(json_file, "r") as f:
                        data[cav_id]['k_frames'][i]['params'] = json.load(f)
                else:
                    data[cav_id]['k_frames'][i]['params'] = \
                        load_yaml(cav_content[timestamp_key_i]['yaml'])

                # load camera file: hdf5 is faster than png
                hdf5_file = cav_content[timestamp_key_i]['cameras'][0].replace("camera0.png", "imgs.hdf5")
                if os.path.exists(hdf5_file):
                    with h5py.File(hdf5_file, "r") as f:
                        data[cav_id]['k_frames'][i]['camera_data'] = []
                        data[cav_id]['k_frames'][i]['depth_data'] = []
                        for i in range(4):
                            data[cav_id]['k_frames'][i]['camera_data'].append(Image.fromarray(f[f'camera{i}'][()]))
                            data[cav_id]['k_frames'][i]['depth_data'].append(Image.fromarray(f[f'depth{i}'][()]))
                else:
                    if self.load_camera_file:
                        data[cav_id]['k_frames'][i]['camera_data'] = \
                            load_camera_data(cav_content[timestamp_key_i]['cameras'])
                    if self.load_depth_file:
                        data[cav_id]['k_frames'][i]['depth_data'] = \
                            load_camera_data(cav_content[timestamp_key_i]['depths']) 

                # load lidar file
                if self.load_lidar_file or self.visualize:
                    data[cav_id]['k_frames'][i]['lidar_np'] = \
                        pcd_utils.pcd_to_np(cav_content[timestamp_key_i]['lidar'])

                for file_extension in self.add_data_extension:
                    # if not find in the current directory
                    # go to additional folder
                    if not os.path.exists(cav_content[timestamp_key_i][file_extension]):
                        cav_content[timestamp_key_i][file_extension] = cav_content[timestamp_key_i][file_extension].replace("train","additional/train")
                        cav_content[timestamp_key_i][file_extension] = cav_content[timestamp_key_i][file_extension].replace("validate","additional/validate")
                        cav_content[timestamp_key_i][file_extension] = cav_content[timestamp_key_i][file_extension].replace("test","additional/test")
                        
                    if '.yaml' in file_extension:
                        data[cav_id]['k_frames'][i][file_extension] = \
                            load_yaml(cav_content[timestamp_key_i][file_extension])
                    else:
                        data[cav_id]['k_frames'][i][file_extension] = \
                            cv2.imread(cav_content[timestamp_key_i][file_extension])


        return data

    def __len__(self):
        return self.len_record[-1]

    def __getitem__(self, idx):
        """
        Abstract method, needs to be define by the children class.
        """
        pass

    @staticmethod
    def extract_timestamps(yaml_files):
        """
        Given the list of the yaml files, extract the mocked timestamps.

        Parameters
        ----------
        yaml_files : list
            The full path of all yaml files of ego vehicle

        Returns
        -------
        timestamps : list
            The list containing timestamps only.
        """
        timestamps = []

        for file in yaml_files:
            res = file.split('/')[-1]

            timestamp = res.replace('.yaml', '')
            timestamps.append(timestamp)

        return timestamps


    @staticmethod
    def find_camera_files(cav_path, timestamp, sensor="camera"):
        """
        Retrieve the paths to all camera files.

        Parameters
        ----------
        cav_path : str
            The full file path of current cav.

        timestamp : str
            Current timestamp

        sensor : str
            "camera" or "depth" 

        Returns
        -------
        camera_files : list
            The list containing all camera png file paths.
        """
        camera0_file = os.path.join(cav_path,
                                    timestamp + f'_{sensor}0.png')
        camera1_file = os.path.join(cav_path,
                                    timestamp + f'_{sensor}1.png')
        camera2_file = os.path.join(cav_path,
                                    timestamp + f'_{sensor}2.png')
        camera3_file = os.path.join(cav_path,
                                    timestamp + f'_{sensor}3.png')
        return [camera0_file, camera1_file, camera2_file, camera3_file]


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


    def generate_object_center_lidar(self,
                               cav_contents,
                               reference_lidar_pose):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.
        The object_bbx_center is in ego coordinate.

        Notice: it is a wrap of postprocessor

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.
            in fact it is used in get_item_single_car, so the list length is 1

        reference_lidar_pose : list
            The final target lidar pose with length 6.

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 7).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """
        return self.post_processor.generate_object_center(cav_contents,
                                                        reference_lidar_pose)

    def generate_object_center_camera(self, 
                                cav_contents, 
                                reference_lidar_pose):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.
        The object_bbx_center is in ego coordinate.

        Notice: it is a wrap of postprocessor

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.
            in fact it is used in get_item_single_car, so the list length is 1

        reference_lidar_pose : list
            The final target lidar pose with length 6.
        
        visibility_map : np.ndarray
            for OPV2V, its 256*256 resolution. 0.39m per pixel. heading up.

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 7).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """
        return self.post_processor.generate_visible_object_center(
            cav_contents, reference_lidar_pose
        )

    def get_ext_int(self, params, camera_id):
        camera_coords = np.array(params["camera%d" % camera_id]["cords"]).astype(np.float32)
        camera_to_lidar = x1_to_x2(
            camera_coords, params["lidar_pose_clean"]
        ).astype(np.float32)  # T_LiDAR_camera
        camera_to_lidar = camera_to_lidar @ np.array(
            [[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
            dtype=np.float32)  # UE4 coord to opencv coord
        camera_intrinsic = np.array(params["camera%d" % camera_id]["intrinsic"]).astype(np.float32)
        
        return camera_to_lidar, camera_intrinsic