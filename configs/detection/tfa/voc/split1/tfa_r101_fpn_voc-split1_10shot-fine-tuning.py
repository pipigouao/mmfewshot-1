_base_ = [
    '../../../_base_/datasets/fine_tune_based/few_shot_voc.py',
    '../../../_base_/schedules/schedule.py', '../../tfa_r101_fpn.py',
    '../../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.
data = dict(
    samples_per_gpu=5,  # 原本是2*8，现在设成5*3
    workers_per_gpu=2,
    train=dict(
        type='FewShotVOCDefaultDataset',
        ann_cfg=[dict(method='TFA', setting='SPLIT1_10SHOT')],
        num_novel_shots=10,  # 此处是随机抽样吗？
        num_base_shots=10,
        classes='ALL_CLASSES_SPLIT1'),
    val=dict(classes='ALL_CLASSES_SPLIT1'),
    test=dict(classes='ALL_CLASSES_SPLIT1'))
evaluation = dict(
    interval=40000,
    class_splits=['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1'])
checkpoint_config = dict(interval=40000)
optimizer = dict(lr=0.001)
lr_config = dict(
    warmup_iters=10, step=[
        36000,
    ])
runner = dict(max_iters=40000)
# base model needs to be initialized with following script:
#   tools/detection/misc/initialize_bbox_head.py
# please refer to configs/detection/tfa/README.md for more details.
# load_from = ('work_dirs/tfa_r101_fpn_voc-split1_base-training/'
#              'base_model_random_init_bbox_head.pth')
load_from = ('/data/jyq/project/VFA/mmfewshot/ckpt/tfa/voc/split1_step2/tfa_r101_fpn_voc-split1_base-training_20211031_114821_random-init-bbox-head-1e681852.pth')
