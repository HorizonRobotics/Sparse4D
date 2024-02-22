# Quick Start

### Set up a new virtual environment
```bash
virtualenv mm_sparse4d --python=python3.8
source mm_sparse4d/bin/activate
```

### Install packpages using pip3
```bash
sparse4d_path="path/to/sparse4d"
cd ${sparse4d_path}
pip3 install --upgrade pip
pip3 install -r requirement.txt
```

### Compile the deformable_aggregation CUDA op
```bash
cd projects/mmdet3d_plugin/ops
python3 setup.py develop
cd ../../../
```

### Prepare the data
Download the [NuScenes dataset](https://www.nuscenes.org/nuscenes#download) and create symbolic links.
```bash
cd ${sparse4d_path}
mkdir data
ln -s path/to/nuscenes ./data/nuscenes
```

Pack the meta-information and labels of the dataset, and generate the required .pkl files.
```bash
pkl_path="data/nuscenes_anno_pkls"
mkdir -p ${pkl_path}
python3 tools/nuscenes_converter.py --version v1.0-mini --info_prefix ${pkl_path}/nuscenes-mini
python3 tools/nuscenes_converter.py --version v1.0-trainval,v1.0-test --info_prefix ${pkl_path}/nuscenes
```

### Generate anchors by K-means
```bash
python3 tools/anchor_generator.py --ann_file ${pkl_path}/nuscenes_infos_train.pkl
```

### Download pre-trained weights
Download the required backbone [pre-trained weights](https://download.pytorch.org/models/resnet50-19c8e357.pth).
```bash
mkdir ckpt
wget https://download.pytorch.org/models/resnet50-19c8e357.pth -O ckpt/resnet50-19c8e357.pth
```

### Commence training and testing
```bash
# train
bash local_train.sh sparse4dv3_temporal_r50_1x8_bs6_256x704

# test
bash local_test.sh sparse4dv3_temporal_r50_1x8_bs6_256x704  path/to/checkpoint
```

For inference-related guidelines, please refer to the [tutorial/tutorial.ipynb](../tutorial/tutorial.ipynb).
