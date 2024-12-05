# python ./tools/detection/test.py \
# configs/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_1shot-fine-tuning.py \
# /data/jyq/project/VFA/mmfewshot/work_dirs/tfa_r101_fpn_voc-split1_10shot-fine-tuning/latest.pth \
# --eval mAP

# 多卡test
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash ./tools/detection/dist_test.sh \
configs/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_1shot-fine-tuning.py \
/data/jyq/project/VFA/mmfewshot/work_dirs/tfa_r101_fpn_voc-split1_10shot-fine-tuning/latest.pth \
4 \
--eval mAP