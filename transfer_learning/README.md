## Acknowledgment

This entire folder has been copied, with minor modifications to fit our usage, from the official [MoCo](https://github.com/facebookresearch/moco/tree/main/detection) github repository. 

## CorInfoMax: Transferring to Detection

The `train_net.py` script reproduces the object detection and instance segmentation experiments on COCO.

### Instruction

1. Install [detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).

1. Convert a pre-trained CorInfoMax model resnet50 backbone to detectron2's format:
   ```
   python convert.py /path/to/corinfomax/resnet50/checkpoint.pth corinfomax_weights.pth
   ```

1. Put dataset under "./datasets" directory,
   following the [directory structure](https://github.com/facebookresearch/detectron2/tree/master/datasets)
	 requried by detectron2.

1. Run training:
   ```
   python train_net.py --config-file configs/coco_R_50_C4_2x_moco.yaml \
	--num-gpus 8 MODEL.WEIGHTS ./corinfomax_weights.pth
   ```

