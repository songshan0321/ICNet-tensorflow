# ICNet_tensorflow
ICNet Tensorflow for ADE20K Indoor Images

Changes:

- Added [relabel.py](relabel.py) to do class relabeling.
- Added graph plotting during training. 
- Modified code and README to cater for ADE20K transfer training.



This repo is modified based on [hellochick/ICNet-tensorflow](https://github.com/hellochick/ICNet-tensorflow) which provides a TensorFlow-based implementation of paper "[ICNet for Real-Time Semantic Segmentation on High-Resolution Images](https://arxiv.org/abs/1704.08545)," by Hengshuang Zhao, and et. al. (ECCV'18).



## Table Of Contents
- [Environment Setup](#environment)
- [Download Weights](#download-weights)
- [Download Dataset](#download-dataset)
  + [ade20k](#download-ade20k)
  + [cityscapes](#download-cityscapes)
- [Get Started!](#get-started)
  + [Inference on your own image](#inference)
  + [Evaluate on cityscapes/ade20k dataset](#evaluation)
  + [Training on your own dataset](#training)
  
## Environment Setup <a name="environment"></a>
```
pip install tensorflow-gpu opencv-python jupyter matplotlib tqdm
```

## Download Weights <a name="download-weights"></a>
Download pre-trained weights for [cityscapes](https://www.cityscapes-dataset.com/) dataset.

```
python script/download_weights.py --dataset cityscapes
```

## Download Dataset (Optional) <a name="download-dataset"></a>
### ADE20k dataset <a name="download-ade20k"></a>
Simply run following command:

```
bash script/download_ADE20k.sh
```

## Get started! <a name="get-started"></a>
### Inference on your own image<a name="inference"></a>
[demo.ipynb](./demo.ipynb) to run semantic segmnetation on your own image. 

In the end of [demo.ipynb](./demo.ipynb), you can test the speed of ICNet.



### Evaluate on your trained dataset <a name="evaluation"></a>
To get the results, you need to follow the steps mentioned above to download dataset first.  
Then you need to change the `data_dir` path in [config.py](./utils/config.py#L6).

```python
ADE20K_DATA_DIR = './data/ADEChallengeData2016/'
```

Run following command to get evaluation results,
```
python evaluate.py --dataset=others --filter-scale=1 --model=others
```

## Training on your own dataset <a name="training"></a>
This implementation is different from the details descibed in ICNet paper, since I did not re-produce model compression part. Instead, we **train on the half kernels directly**.  

In orignal paper, the authod trained the model in full kernels and then performed model-pruning techique to kill half kernels. Here **we use --filter-scale to denote whether pruning or not**. 

For example, `--filter-scale=1` <-> `[h, w, 32]` and `--filter-scale=2` <-> `[h, w, 64]`. 

### Step by Step
**1. Change the configurations** in [utils/config.py](./utils/config.py).

```python
others_param = {'name': 'lb_ade20k',
                    'num_classes': 5,
                    'ignore_label': 255,
                    'eval_size': [480, 640],
                    'eval_steps': 2000,
                    'eval_list': ADE20K_eval_list,
                    'train_list': ADE20K_train_list,
                    'data_dir': LB_ADE20K_DATA_DIR}
```

**2. Set Hyperparameters** in `train.py`, 

```python
class TrainConfig(Config):
    def __init__(self, dataset, is_training,  filter_scale=1, random_scale=None, random_mirror=None):
        Config.__init__(self, dataset, is_training, filter_scale, random_scale, random_mirror)

    # --------------- Set pre-trained weights here ----------------
    ## To train on pretrained weight on cityscapes
    model_weight = './model/cityscapes/icnet_cityscapes_trainval_90k_bnnomerge.npy'
    ## or To further train on your own checkpoint
    # model_weight = './snapshots/model.ckpt-1000'
    # -------------------------------------------------------------
    
    # Set hyperparameters here, you can get much more setting in Config Class, see 'utils/config.py' for details.
    LAMBDA1 = 0.4
    LAMBDA2 = 0.6
    LAMBDA3 = 1.0
    BATCH_SIZE = 8
    LEARNING_RATE = 5e-4
```

**3.** Run following command and **decide whether to update mean/var or train beta/gamma variable**.
```
python train.py --update-mean-var --train-beta-gamma --random-scale --random-mirror --dataset others --filter-scale 1
```

**Note: Be careful to use `--update-mean-var`!** Use this flag means you will update the moving mean and moving variance in batch normalization layer. This **need large batch size**, otherwise it will lead bad results. 



## Citation
    @article{zhao2017icnet,
      author = {Hengshuang Zhao and
                Xiaojuan Qi and
                Xiaoyong Shen and
                Jianping Shi and
                Jiaya Jia},
      title = {ICNet for Real-Time Semantic Segmentation on High-Resolution Images},
      journal={arXiv preprint arXiv:1704.08545},
      year = {2017}
    }
    
    @inproceedings{zhou2017scene,
        title={Scene Parsing through ADE20K Dataset},
        author={Zhou, Bolei and Zhao, Hang and Puig, Xavier and Fidler, Sanja and Barriuso, Adela and Torralba, Antonio},
        booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
        year={2017}
    }
    
    @article{zhou2016semantic,
      title={Semantic understanding of scenes through the ade20k dataset},
      author={Zhou, Bolei and Zhao, Hang and Puig, Xavier and Fidler, Sanja and Barriuso, Adela and Torralba, Antonio},
      journal={arXiv preprint arXiv:1608.05442},
      year={2016}
    }
