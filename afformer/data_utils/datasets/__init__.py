from afformer.data_utils.datasets.late_fusion_dataset import getLateFusionDataset
from afformer.data_utils.datasets.early_fusion_dataset import getEarlyFusionDataset
from afformer.data_utils.datasets.intermediate_fusion_dataset import getIntermediateFusionDataset
from afformer.data_utils.datasets.basedataset.dairv2x_basedataset import DAIRV2XBaseDataset
from afformer.data_utils.datasets.basedataset.v2xset_basedataset import V2XSETBaseDataset


def build_dataset(dataset_cfg, visualize=False, train=True, mode=None):
    fusion_name = dataset_cfg['fusion']['core_method']
    dataset_name = dataset_cfg['fusion']['dataset']

    assert fusion_name in ['late', 'intermediate', 'early']
    assert dataset_name in ['dairv2x', 'v2xset']

    fusion_dataset_func = "get" + fusion_name.capitalize() + "FusionDataset"
    base_dataset_cls = dataset_name.upper() + "BaseDataset"

    fusion_dataset_func = eval(fusion_dataset_func)
    base_dataset_cls = eval(base_dataset_cls)

    dataset = fusion_dataset_func(base_dataset_cls)(
        params=dataset_cfg,
        visualize=visualize,
        train=train,
        mode=mode
    )

    return dataset