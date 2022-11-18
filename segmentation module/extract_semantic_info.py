from argparse import ArgumentParser
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import numpy as np
from collections import Counter
import os
from PIL import Image
import mmcv

def main():
    parser = ArgumentParser()
    parser.add_argument('--img_file_path', default='img/', help='Image file Path')
    parser.add_argument('--config', default='local_configs/segformer/B1/segformer.b1.1024x1024.city.160k.py', help='Config file')
    parser.add_argument('--checkpoint', default='checkpoints/segformer.b1.1024x1024.city.160k.pth', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    parser.add_argument('--save_seg_path', default='segmented_image/',
        help='Path used to save semantic segmentation images')
    parser.add_argument('--save_result_path', default='semantic_info_file/semantic_info.txt',
        help='Path used to save result')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    # save the semantic info to txt file
    save_to_file = open(args.save_result_path,'w')
    imgs = os.listdir(args.img_file_path)
    for img in imgs:
      if os.path.splitext(img)[-1] in ['.jpg', '.png']:
        img_path = args.img_file_path + img
        result = inference_segmentor(model, img_path)
        result_to_list = list(np.array(result[0]).flatten())

        IMG = Image.open(img_path)
        IMG_width = IMG.width
        IMG_height = IMG.height
        IMG.close()
        sum_pixel = IMG_width*IMG_height

        count_dict = lsit_to_dict(result_to_list, sum_pixel)

        #save to file
        save_to_file.write(str(img)+'\t'+str(count_dict[0])+'\t'+str(count_dict[1])+'\t'+str(count_dict[2])+'\t'\
          +str(count_dict[3])+'\t'+str(count_dict[4])+'\t'+str(count_dict[5])+'\t'\
          +str(count_dict[6])+'\t'+str(count_dict[7])+'\t'+str(count_dict[8])+'\t'\
          +str(count_dict[9])+'\t'+str(count_dict[10])+'\t'+str(count_dict[11])+'\t'\
          +str(count_dict[12])+'\t'+str(count_dict[13])+'\t'+str(count_dict[14])+'\t'\
          +str(count_dict[15])+'\t'+str(count_dict[16])+'\t'+str(count_dict[17])+'\t'+str(count_dict[18])+'\n')
        
        # output and save the seg_img
        seg_img = seg_result(model, img_path, result, get_palette(args.palette))
        seg_img.save(args.save_seg_path + img)

    save_to_file.close()

def lsit_to_dict(result_to_list, sum_pixel):
  count_dict = {}
  for i in range(19):
    count_dict[i] = round((result_to_list.count(i))/sum_pixel, 6)
  
  return count_dict

def seg_result(model, img, result, palette=None, fig_size=(15, 10)):
  if hasattr(model, 'module'):
    model = model.module
  img = model.show_result(img, result, palette=palette, show=False)
  img = mmcv.bgr2rgb(img)
  img = Image.fromarray(img)
  return img

if __name__ == '__main__':
    main()
      

