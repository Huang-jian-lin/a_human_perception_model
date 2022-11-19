####  Introduction

1. Folder *classification module* is used to extract category probability feature, folder *segmentation module* is used to extract semantic feature, and folder *regression module* is used to predict perceptual scores.

2. We use the VIT model trained on the perception dataset to extract the category probability feature. For the use of the VIT model, we employ the repository [mmclassification](https://github.com/open-mmlab/mmclassification), so you need to put it in folder *classification module*. We provide the model file [*checkpoints* ](https://drive.google.com/drive/folders/1--WZYmruXgXxwNe47MbWuCASrjMcxsxC?usp=share_link)trained on the perception dataset you can download and put it in the above folder. For example, use the code we provide to extract category probability feature for the perceptual dimension *beautiful*:

   `python extract_category_probability_info/beautiful_vit.py`

3. We use the Segformer model trained on the cityscapes dataset to extract semantic feature. For the use of the Segformer model, we employ the repository [SegFormer](https://github.com/NVlabs/SegFormer), so you need to put it in folder *segmentation module*. And you can download the model file [*checkpoints* ](https://drive.google.com/drive/folders/1-Dtns8rlcgpNkTw4f0Nvlt16qu5t5G9P?usp=share_link)into the above folder. Use the code we provide to extract the semantic information:

   `python extract_semantic_info.py`

4. We use the RF model trained on the perception dataset to predict the scores of each dimension. We provide the model files trained on the six perceptual dimensions in the folder *regression module/model_weights*. For example, you can predict the perceptual scores of the dimension *beautiful*:

   `python RF_prediction.py`

#### references

[1] [open-mmlab/mmclassification](https://github.com/open-mmlab/mmclassification)

[2] [NVlabs/SegFormer](https://github.com/NVlabs/SegFormer)

