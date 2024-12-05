# 多卡train(finetune)
# CUDA_VISIBLE_DEVICES=3 \
# bash ./tools/detection/dist_train.sh \
# /data/jyq/project/VFA/mmfewshot/mycfg/tfa_r101_fpn_coco_10shot-fine-tuning.py \
# 1

CUDA_VISIBLE_DEVICES=3 python ./tools/detection/train.py /data/jyq/project/VFA/mmfewshot/configs/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_5shot-fine-tuning.p
