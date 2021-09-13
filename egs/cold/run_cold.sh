#!/bin/bash
#SBATCH -p gpu
#SBATCH -x sls-titan-[0-2]
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=48000
#SBATCH --job-name="ast-esc50"
#SBATCH --output=./log_%j.txt

set -x
# comment this line if not running on sls cluster
#. /data/sls/scratch/share-201907/slstoolchainrc
#source ../../venvast/bin/activate
export TORCH_HOME=../../pretrained_models

model=ast
dataset=cold
imagenetpretrain=True
audiosetpretrain=True
bal=none
if [ $audiosetpretrain == True ]
then
  lr=1e-5
else
  lr=1e-4
fi
freqm=24
timem=96
mixup=0
epoch=25
batch_size=16
fstride=10
tstride=10
base_exp_dir=./exp/test-${dataset}-f$fstride-t$tstride-imp$imagenetpretrain-asp$audiosetpretrain-b$batch_size-lr${lr}

#python ./prep_esc50.py

if [ -d $base_exp_dir ]; then
  echo 'exp exist'
  rmdir -r $exp_dir
  #exit
fi
mkdir -p $exp_dir


exp_dir=${base_exp_dir}

train_data=./data/datafiles/${dataset}_train_data.json
val_data=./data/datafiles/${dataset}_dev_data.json
eval_data=./data/datafiles/${dataset}_test_data.json

CUDA_CACHE_DISABLE=1 CUDA_DEVICE_ORDER="PCE_BUS_ID" CUDA_VISIBLE_DEVICES="0,1,2,3" python -W ignore ../../src/run.py --model ${model} --dataset ${dataset} \
--data-train ${train_data} --data-val ${val_data} --data-eval ${eval_data} --exp-dir $exp_dir \
--label-csv ./data/${dataset}_class_label_indices.csv --n_class 2 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --imagenet_pretrain $imagenetpretrain --audioset_pretrain $audiosetpretrain

#python ./get_esc_result.py --exp_path ${base_exp_dir}