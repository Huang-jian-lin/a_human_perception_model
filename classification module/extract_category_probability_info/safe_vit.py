import mmcv
import time
import os
import os.path as osp
from argparse import ArgumentParser
from mmcls.apis import inference_model, init_model, show_result_pyplot, train_model
from mmcv import Config
from mmcls.apis import set_random_seed
from mmcls.datasets import build_dataset
from mmcls.models import build_classifier


def main():
    parser = ArgumentParser()
    parser.add_argument('--img_file_path', default='img/', help='Image file Path')
    parser.add_argument('--config', default='configs/vision_transformer/vit-base-p16_ft-64xb64_in1k-384.py', help='Config file')
    parser.add_argument('--checkpoint', default='checkpoints/safe/safe.pth', help='Checkpoint file')
    parser.add_argument('--save_result_path', default='category_probability_info_file/safe_category_probability_info.txt', help='Path used to save result')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    checkpoint_file = args.checkpoint
    cfg.model.backbone.init_cfg = dict(type='Pretrained', checkpoint=checkpoint_file, prefix='backbone')

    # modify the config
    # Path for dataset
    cfg.data.train.data_prefix = 'data/safe/train'
    cfg.data.train.classes = 'data/safe/classes.txt'

    cfg.data.val.data_prefix = 'data/safe/val'
    cfg.data.val.ann_file = 'data/safe/val.txt'
    cfg.data.val.classes = 'data/safe/classes.txt'

    cfg.data.test.data_prefix = 'data/safe/test'
    cfg.data.test.ann_file = 'data/safe/test.txt'
    cfg.data.test.classes = 'data/safe/classes.txt'

    normalize_cfg = dict(type='Normalize', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)
    cfg.data.train.pipeline[3] = normalize_cfg
    cfg.data.val.pipeline[3] = normalize_cfg
    cfg.data.test.pipeline[3] = normalize_cfg

    cfg.model.head.num_classes = 10
    cfg.model.head.topk = (1, 5)
    cfg.data.samples_per_gpu = 16
    cfg.data.workers_per_gpu = 2

    cfg.evaluation['metric_options']={'topk': (1, 5)}
    cfg.optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
    cfg.optimizer_config = dict(grad_clip=None)
    cfg.lr_config = dict(
        policy='CosineAnnealing',
        min_lr=0,
        warmup='linear',
        warmup_iters=1000,
        warmup_ratio=1e-4)
    cfg.runner = dict(type='EpochBasedRunner', max_epochs=10)

    cfg.work_dir = 'work_dirs/safe_dataset_10e'
    cfg.seed = 0
    set_random_seed(0, deterministic=True)

    cfg.gpu_ids = range(1)
    device = 'cuda:0'
    model = init_model(cfg, checkpoint_file, device=device)
    model.cfg = cfg

    save_to_file = open(args.save_result_path, mode='w')
    path = args.img_file_path
    img_files = os.listdir(path)
    for img_file in img_files:
      img_file_path = path + img_file
      img = mmcv.imread(img_file_path)
      result, score = inference_model(model, img)
      save_to_file.write(img_file + "\t" + str(score[0][0]) + "\t" + str(score[0][1]) + "\t" + str(score[0][2]) + "\t" + \
       str(score[0][3]) + "\t" + str(score[0][4]) + "\t" + str(score[0][5]) + "\t" + \
       str(score[0][6]) + "\t" + str(score[0][7]) + "\t" + str(score[0][8]) + "\t" + str(score[0][9]) + "\n")

    save_to_file.close()


if __name__ == '__main__':
    main()