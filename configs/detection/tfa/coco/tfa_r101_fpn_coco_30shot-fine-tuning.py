_base_ = [
    '../../_base_/datasets/fine_tune_based/few_shot_coco.py',
    '../../_base_/schedules/schedule.py', '../tfa_r101_fpn.py',
    '../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotCocoDataset
# FewShotCocoDefaultDataset predefine ann_cfg for model reproducibility
data = dict(
    samples_per_gpu=5,  # 原本是2*8，现在设成5*3
    workers_per_gpu=2,
    train=dict(
        type='FewShotCocoDefaultDataset',
        ann_cfg=[dict(method='TFA', setting='30SHOT')],
        num_novel_shots=30,
        num_base_shots=30))
evaluation = dict(interval=120000)
checkpoint_config = dict(interval=120000)
optimizer = dict(lr=0.001)
lr_config = dict(
    warmup_iters=10, step=[
        216000,
    ])
runner = dict(max_iters=240000)
model = dict(roi_head=dict(bbox_head=dict(num_classes=80)))
# base model needs to be initialized with following script:
#   tools/detection/misc/initialize_bbox_head.py
# please refer to configs/detection/tfa/README.md for more details.
load_from = ('/data/jyq/project/VFA/mmfewshot/ckpt/tfa/coco/tfa_r101_fpn_coco_base-training_20211102_030413_random-init-bbox-head-ea1c2981.pth')
