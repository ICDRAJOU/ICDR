# <p align=center>ICDR: Image Restoration Framework for Composite Degradation following Human Instructions</p>


<div align="center">
 
[![Hugging Face Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/Siwon123/ICDR)

</div>

---
## Dependencies

* Python == 3.10
* Pytorch == 2.4.0
* mmcv == 2.2.0

requirements.txt allows you to install the environment required to execute the project. If this is not possible, please contact "kimsiw42@ajou.ac.kr"

## Dataset

You could find the dataset we used in the project at following:

CDD-11: https://github.com/gy65896/OneRestore/tree/main?tab=readme-ov-file

## Demo

You could download the pre-trained model from [Google Drive](https://drive.google.com/file/d/13jCADhncLHCtEt0Ad8lsBskHo67-hxtO/view?usp=sharing) Remember to put the pre-trained model into ckpt/

Support demo using gradio. You can run demo in huggingface space.
[![Hugging Face Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/Siwon123/ICDR)


However, in huggingface space, it can take a long time to operate in cpu environment. (30 minutes per sheet)

In this case, you can run app.py directly to test demo in a local environment.

```bash
python app.py
```

## Training

If you want to train our model, you need to first put the Dataset to in data/CDD-11_test_100 and data/CDD-11_train_100

```bash
python train_text.py
```

## Citation

If you find ICDR useful in your research, please consider citing:

```
@inproceedings{ICDR,
author = 
title = 
booktitle = 
year = 
address = 
month = 
}
```

## Acknowledgement

This repo is built upon the framework of [AIRNET](https://github.com/XLearning-SCU/2022-CVPR-AirNet), thanks for their excellent work!

