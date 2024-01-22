export PYTHONPATH=$PYTHONPATH:./
export CUDA_VISIBLE_DEVICES=3
export PORT=29532

gpus=(${CUDA_VISIBLE_DEVICES//,/ })
gpu_num=${#gpus[@]}

config=projects/configs/$1.py
checkpoint=$2

echo "number of gpus: "${gpu_num}
echo "config file: "${config}
echo "checkpoint: "${checkpoint}

if [ ${gpu_num} -gt 1 ]
then
    bash ./tools/dist_test.sh \
        ${config} \
        ${checkpoint} \
        ${gpu_num} \
        --eval bbox \
        $@
else
    python ./tools/test.py \
        ${config} \
        ${checkpoint} \
        --eval bbox \
        $@
fi
