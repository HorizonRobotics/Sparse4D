export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:./

gpus=(${CUDA_VISIBLE_DEVICES//,/ })
gpu_num=${#gpus[@]}
echo "number of gpus: "${gpu_num}

config=projects/configs/$1.py

if [ ${gpu_num} -gt 1 ]
then
    bash ./tools/dist_train.sh \
        ${config} \
        ${gpu_num} \
        --work-dir=work_dirs/$1
else
    python ./tools/train.py \
        ${config}
fi
