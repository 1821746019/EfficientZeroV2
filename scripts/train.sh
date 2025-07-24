#!/bin/bash
set -ex
#export OMP_NUM_THREADS=1
#export CUDA_DEVICE_ORDER='PCI_BUS_ID'
#export CUDA_VISIBLE_DEVICES=0,1
#export CUDA_VISIBLE_DEVICES=2,3
export CUDA_VISIBLE_DEVICES=6,7
#export CUDA_VISIBLE_DEVICES=6,7
export NCCL_IGNORE_DISABLED_P2P=1
export HYDRA_FULL_ERROR=1
export MASTER_PORT='12399'
# 不用在无GPU的环境MUJOCO才不会报错 export MUJOCO_GL=osmesa #egl
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1        # 如果装了 MKL
# python ez/train.py exp_config=ez/config/exp/atari.yaml #> profile.txt
# python ez/train.py exp_config=ez/config/exp/dmc_image.yaml #> profile.txt
python ez/train.py exp_config=ez/config/exp/dmc_state.yaml #> profile.txt
# gym的没完全适配，缺少相应的Agent类
# python ez/train.py exp_config=ez/config/exp/state.yaml #> profile.txt 
