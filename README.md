# jncsnlp/QAIE-ABSA
QAIE-ABSA
# [QAIE: LLM-based Quantity Augmentation and Information Enhancement for few-shot Aspect-Based Sentiment Analysis](https://www.sciencedirect.com/science/article/pii/S0306457324002760?dgcid=author)

## Installation

Create a conda environment with pytorch and scikit-learn :
```
conda create --name QAIE python=3.8
source activate QAIE
conda install --file requirements.txt -c pytorch
```

## Train
```
python3 main.py \
    --task at \
    --dataset rest15 \
    --model_name_or_path t5-model/base \
    --n_gpu 0 \
    --do_train \
    --train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --eval_batch_size 16 \
    --learning_rate 3e-4 \
    --num_train_epochs 20
```

## Test
```
python3 main.py \
    --task at \
    --dataset rest15 \
    --model_name_or_path t5-model/base \
    --n_gpu 0 \
    --do_inference \
    --train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --eval_batch_size 16 \
    --learning_rate 3e-4 \
    --num_train_epochs 20
```

## Acknowledgement

We refer to the code of QAIE. Thanks for their great contributions!

## Cite Information

Lu H, Liu T, Cong R, et al. QAIE: LLM-based Quantity Augmentation and Information Enhancement for few-shot Aspect-Based Sentiment Analysis[J]. Information Processing & Management, 2025, 62(1): 103917.
```
@article{lu2025qaie,
  title={QAIE: LLM-based Quantity Augmentation and Information Enhancement for few-shot Aspect-Based Sentiment Analysis},
  author={Lu, Heng-yang and Liu, Tian-ci and Cong, Rui and Yang, Jun and Gan, Qiang and Fang, Wei and Wu, Xiao-jun},
  journal={Information Processing \& Management},
  volume={62},
  number={1},
  pages={103917},
  year={2025},
  publisher={Elsevier}
}
```
Contact: liutianci@stu.jiangnan.edu.cn


