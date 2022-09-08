# TFCNs (ICANN 2022 Oral)
This repository includes the official project of TFCNs, presented in our paper:  TFCNs: A CNN-Transformer Hybrid Network for Medical Image Segmentation
, which is accepted by ICANN 2022 (International Conference on Artificial Neural Networks).
![image](https://github.com/HUANGLIZI/TFCNs/blob/main/imgs/TFCNs.jpg)

paper link: https://arxiv.org/abs/2207.03450 or https://doi.org/10.1007/978-3-031-15937-4_65

Email: dihanli@stu.xmu.edu.cn

Please contact dihan or me if you need the further help.


# Usage

model/ : save for the model you have train

networks/ : all the component that construct our TFCNs

preprocess.py : simple data augumentation

train_utils.py : some tools used for training

utils.py : some tools used for testing

you can run the train.py and test.py for training and testing.

# Environment

Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

# Citation

```bash
@article{li2022TFCNs,
  title={TFCNs: A CNN-Transformer Hybrid Network for Medical Image Segmentation},
  author={Li, Zihan and Li, Dihan and Xu, Cangbai and Wang, Weice and Hong, Qingqi and Li, Qingde and Tian, Jie},
  journal={ICANN2022},
  year={2022}
}
```
