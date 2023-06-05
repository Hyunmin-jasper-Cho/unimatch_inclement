<p align="center">
  <h1 align="center">Unifying Flow, Stereo and Depth Estimation - Fine tuning with Driving stereo</h1>
  <a> Original Authors and Projects Links</a>  
  <hr>
  <p align="center">
    <a href="https://haofeixu.github.io/">Haofei Xu</a>
    ·
    <a href="https://scholar.google.com/citations?user=9jH5v74AAAAJ">Jing Zhang</a>
    ·
    <a href="https://jianfei-cai.github.io/">Jianfei Cai</a>
    ·
    <a href="https://scholar.google.com/citations?user=VxAuxMwAAAAJ">Hamid Rezatofighi</a>
    ·
    <a href="https://www.yf.io/">Fisher Yu</a>
    ·
    <a href="https://scholar.google.com/citations?user=RwlJNLcAAAAJ">Dacheng Tao</a>
    ·
    <a href="http://www.cvlibs.net/">Andreas Geiger</a> 
  </p>
  <h3 align="center"><a href="https://arxiv.org/abs/2211.05783">Paper</a> | <a href="https://haofeixu.github.io/slides/20221228_synced_unimatch.pdf">Slides</a> | <a href="https://haofeixu.github.io/unimatch/">Project Page</a> | <a href="https://colab.research.google.com/drive/1r5m-xVy3Kw60U-m5VB-aQ98oqqg_6cab?usp=sharing">Colab</a> | <a href="https://huggingface.co/spaces/haofeixu/unimatch">Demo</a> </h3>
  <div align="center"></div>
</p>
<hr>
<a> Fine tuned with </a> <a href="https://drivingstereo-dataset.github.io">Driving Stereo Dataset</a>  
<p> Fine tuned result images </p>
 
![2018-08-17-09-45-58_2018-08-17-10-32-05-242_disp](https://github.com/Hyunmin-jasper-Cho/unimatch_inclement/assets/71583831/2386bcd0-79b8-4cfc-8bc1-1a13ad7e8037)

<p> Input left image</p> 

![2018-08-17-09-45-58_2018-08-17-10-32-05-242](https://github.com/Hyunmin-jasper-Cho/unimatch_inclement/assets/71583831/b48ad8b8-dc5e-4dc6-938c-b6e004f2239d)

<p> Input right image</p> 

![2018-08-17-09-45-58_2018-08-17-10-32-05-242](https://github.com/Hyunmin-jasper-Cho/unimatch_inclement/assets/71583831/71fc82f8-c232-4fc5-ae0a-a58c37266961)

## Installation

Our code follows the original code instllation. 

We recommend using [conda](https://www.anaconda.com/distribution/) for installation:

```
conda env create -f conda_environment.yml
conda activate unimatch
```

Alternatively, we also support installing with pip:

```
bash pip_install.sh
```



## Model Zoo

Our trained model for transfer learned can be download [here](https://drive.google.com/drive/folders/1IuNFBaZhZDLP9yROTSkzWi9Om6tq_9fO?usp=share_link) 


## Datasets

The datasets used to train and evaluate our models for all three tasks are given in [Here](https://drivingstereo-dataset.github.io)


## Training & Evaluation log 
You can refer the [colab](https://colab.research.google.com/drive/1659K51jWSBy04uYmi3TlGeSl68wqUT4i?usp=share_link) link. 


## Evaluation - only for stereo dataset 

```
!python main_tf_stereo.py --resume ./pretrained/step_037000.pth --inference_dir_left ./inference/left --inference_dir_right ./inference/right
```


## Training - only for stereo dataset  

```
!python main_tf_stereo.py --resume ./pretrained/unimatch_kitty.pth --freeze 100 --transfer_learning --no_resume_optimizer
```
<img width="373" alt="스크린샷 2023-06-03 21 37 19" src="https://github.com/Hyunmin-jasper-Cho/unimatch_inclement/assets/71583831/64e32336-0366-4dc4-86da-102b09adcced">


## Code modification 

### Trainer modification (main_tf_stereo.py)
```
parser.add_argument('--transfer_learning', action='store_true')
parser.add_argument('--freeze', default=10, type=int)
```

```
# model
model = UniMatch(feature_channels=args.feature_channels,
                 num_scales=args.num_scales,
                 upsample_factor=args.upsample_factor,
                 num_head=args.num_head,
                 ffn_dim_expansion=args.ffn_dim_expansion,
                 num_transformer_layers=args.num_transformer_layers,
                 reg_refine=args.reg_refine,
                 task=args.task).to(device)
    
if print_info:
    print(model)
    
# If we are doing transfer learning, freeze some layers
if args.transfer_learning:
    f = args.freeze
    _i = 0
    for _, param in model.named_parameters():
        # if 'last_layer_name' not in name:  # replace 'last_layer_name' with your actual last layer's name
        if _i < f:
            param.requires_grad = False
            _i += 1
    print("=> Performing transfer learning. Some layers are frozen.")
```

```
if args.resume or args.transfer_learning:
    print("=> Load checkpoint: %s" % args.resume)

    loc = 'cuda:{}'.format(args.local_rank) if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(args.resume, map_location=loc)

    model_without_ddp.load_state_dict(checkpoint['model'], strict=args.strict_resume)

    if 'optimizer' in checkpoint and 'step' in checkpoint and 'epoch' in checkpoint and not \
            args.no_resume_optimizer:
        print('Load optimizer')
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_step = checkpoint['step']
        start_epoch = checkpoint['epoch']

    if print_info:
        print('start_epoch: %d, start_step: %d' % (start_epoch, start_step))
```

```
if args.resume or args.transfer_learning:
    print("=> Load checkpoint: %s" % args.resume)

    loc = 'cuda:{}'.format(args.local_rank) if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(args.resume, map_location=loc)

    model_without_ddp.load_state_dict(checkpoint['model'], strict=args.strict_resume)

    if 'optimizer' in checkpoint and 'step' in checkpoint and 'epoch' in checkpoint and not \
            args.no_resume_optimizer:
        print('Load optimizer')
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_step = checkpoint['step']
        start_epoch = checkpoint['epoch']

    if print_info:
        print('start_epoch: %d, start_step: %d' % (start_epoch, start_step))
```

```
elif args.stage == 'inclement':
        train_transform_list = [transforms.RandomScale(crop_width=args.img_width),
                                transforms.RandomCrop(args.img_height, args.img_width),
                                transforms.RandomColor(),
                                transforms.RandomVerticalFlip(),
                                transforms.ToTensor(no_normalize=args.raft_stereo),
                                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                                ]

        train_transform = transforms.Compose(train_transform_list)

        train_dataset = INCLEMENT(transform=train_transform)

        return train_dataset
```

```
# validation
if total_steps % args.val_freq == 0:
    val_results = {}
    if 'inclement' in args.val_dataset:
        results_dict = validate_inclement(model_without_ddp,
                                    max_disp=args.max_disp,
                                    padding_factor=args.padding_factor,
                                    inference_size=args.inference_size,
                                    attn_type=args.attn_type,
                                    attn_splits_list=args.attn_splits_list,
                                    corr_radius_list=args.corr_radius_list,
                                    prop_radius_list=args.prop_radius_list,
                                    num_reg_refine=args.num_reg_refine,
                                    )

        if args.local_rank == 0:
            val_results.update(results_dict)
```

```
@torch.no_grad()
def validate_inclement(model,
                       max_disp=400,
                       padding_factor=16,
                       inference_size=None,
                       attn_type=None,
                       num_iters_per_scale=None,
                       attn_splits_list=None,
                       corr_radius_list=None,
                       prop_radius_list=None,
                       num_reg_refine=1,
                       ):
    model.eval()
    results = {}

    val_transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                          ]

    val_transform = transforms.Compose(val_transform_list)

    val_dataset = INCLEMENT(mode='testing', transform=val_transform)

    num_samples = len(val_dataset)
    print('=> %d samples found in the validation set' % num_samples)

    val_epe = 0
    val_d1 = 0

    valid_samples = 0

    for i, sample in enumerate(val_dataset):
        if i % 48 == 0:
            print('=> Validating %d/%d' % (i, num_samples))

        left = sample['left'].to(device).unsqueeze(0)  # [1, 3, H, W]
        right = sample['right'].to(device).unsqueeze(0)  # [1, 3, H, W]
        gt_disp = sample['disp'].to(device)  # [H, W]

        if inference_size is None:
            padder = InputPadder(left.shape, padding_factor=padding_factor)
            left, right = padder.pad(left, right)
        else:
            ori_size = left.shape[-2:]
            left = F.interpolate(left, size=inference_size, mode='bilinear',
                                 align_corners=True)
            right = F.interpolate(right, size=inference_size, mode='bilinear',
                                  align_corners=True)

        mask = (gt_disp > 0) & (gt_disp < max_disp)

        if not mask.any():
            continue

        valid_samples += 1

        with torch.no_grad():
            pred_disp = model(left, right,
                              attn_type=attn_type,
                              num_iters_per_scale=num_iters_per_scale,
                              attn_splits_list=attn_splits_list,
                              corr_radius_list=corr_radius_list,
                              prop_radius_list=prop_radius_list,
                              num_reg_refine=num_reg_refine,
                              task='stereo',
                              )['flow_preds'][-1]  # [1, H, W]

        # remove padding
        if inference_size is None:
            pred_disp = padder.unpad(pred_disp)[0]  # [H, W]
        else:
            # resize back
            pred_disp = F.interpolate(pred_disp.unsqueeze(1), size=ori_size, mode='bilinear',
                                      align_corners=True).squeeze(1)[0]  # [H, W]
            pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])

        epe = F.l1_loss(gt_disp[mask], pred_disp[mask], reduction='mean')
        d1 = d1_metric(pred_disp, gt_disp, mask)

        val_epe += epe.item()
        val_d1 += d1.item()

    mean_epe = val_epe / valid_samples
    mean_d1 = val_d1 / valid_samples

    print('Validation inclement EPE: %.3f, D1: %.4f' % (
        mean_epe, mean_d1))

    results['inclement_epe'] = mean_epe
    results['inclement_d1'] = mean_d1

    return results
```

### DataLoader modification 
```
# stage = train_dataset: We use our own custom dataset: inclement
parser.add_argument('--stage', default='inclement', type=str,
                    help='training stage on different datasets')
```

```
class INCLEMENT(StereoDataset):
    def __init__(self,
                 data_dir='dataloader/stereo/datasets/inclement',
                 mode='training',
                 transform=None,
                 save_filename=False,
                 ):
        super(INCLEMENT, self).__init__(transform=transform)

        assert mode in ['training', 'testing']

        self.save_filename = save_filename

        left_files = sorted(glob(data_dir + '/' + mode + '/left-image-half-size/*.jpg'))
        right_files = sorted(glob(data_dir + '/' + mode + '/right-image-half-size/*.jpg'))
        disparity_files = sorted(glob(data_dir + '/' + mode + '/disparity-map-half-size/*.png'))
        
        assert len(left_files) == len(right_files) == len(disparity_files), \
            f'len(left_files): {len(left_files)}, len(right_files): {len(right_files)}, len(disparity_files): {len(disparity_files)}'

        for i in range(len(left_files)):
            sample = dict()
            sample['left'] = left_files[i]
            sample['right'] = right_files[i]
            sample['disp'] = disparity_files[i]

            if self.save_filename:
                sample['left_name'] = os.path.basename(left_files[i])

            self.samples.append(sample)

        print(f'The length of the {mode} dataset is {len(self.samples)}')
```

## Citation

```
@article{xu2022unifying,
  title={Unifying Flow, Stereo and Depth Estimation},
  author={Xu, Haofei and Zhang, Jing and Cai, Jianfei and Rezatofighi, Hamid and Yu, Fisher and Tao, Dacheng and Geiger, Andreas},
  journal={arXiv preprint arXiv:2211.05783},
  year={2022}
}
```

This work is a substantial extension of our previous conference paper [GMFlow (CVPR 2022, Oral)](https://arxiv.org/abs/2111.13680), please consider citing GMFlow as well if you found this work useful in your research.

```
@inproceedings{xu2022gmflow,
  title={GMFlow: Learning Optical Flow via Global Matching},
  author={Xu, Haofei and Zhang, Jing and Cai, Jianfei and Rezatofighi, Hamid and Tao, Dacheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8121-8130},
  year={2022}
}
```



## Acknowledgements

This project would not have been possible without relying on some awesome repos: [RAFT](https://github.com/princeton-vl/RAFT), [LoFTR](https://github.com/zju3dv/LoFTR), [DETR](https://github.com/facebookresearch/detr), [Swin](https://github.com/microsoft/Swin-Transformer), [mmdetection](https://github.com/open-mmlab/mmdetection) and [Detectron2](https://github.com/facebookresearch/detectron2/blob/main/projects/TridentNet/tridentnet/trident_conv.py). We thank the original authors for their excellent work.







