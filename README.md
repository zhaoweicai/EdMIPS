## EdMIPS: Rethinking Differentiable Search for Mixed-Precision Neural Networks

by [Zhaowei Cai](https://zhaoweicai.github.io/), and [Nuno Vasconcelos](http://www.svcl.ucsd.edu/~nuno/).

This implementation is written by Zhaowei Cai at UC San Diego.

### Introduction

EdMIPS is an efficient algorithm to search the optimal mixed-precision neural network directly without proxy task on ImageNet given computation budgets. It can be applied to many popular network architectures, including ResNet, GoogLeNet, and Inception-V3. More details can be found in the [paper](https://arxiv.org/abs/2004.05795).

### Citation

If you use our code/model/data, please cite our paper:

    @inproceedings{cai20edmips,
      author = {Zhaowei Cai and Nuno Vasconcelos},
      Title = {Rethinking Differentiable Search for Mixed-Precision Neural Networks},
      booktitle = {CVPR},
      Year  = {2020}
    }

### Installation

1. Install PyTorch and ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet).

2. Clone the EdMIPS repository, and we'll call the directory that you cloned EdMIPS into `EdMIPS_ROOT`
    ```Shell
    git clone https://github.com/zhaoweicai/EdMIPS.git
    cd EdMIPS_ROOT/
    ```

### Searching the Mixed-precision Network with EdMIPS

You can start training EdMIPS. Take ResNet-18 for example. 
```
python search.py \
  -a mixres18_w1234a234 --epochs 25 --step-epoch 10 --lr 0.1 --lra 0.01 --cd 0.00335 -j 16 \
  [your imagenet-folder with train and val folders]
```
    
The other network architectures are also available, including ResNet-50, GoogLeNet and Inception-V3.

### Training the Searched Mixed-precision Network

After the EdMIPS searching is finished, with the checkpoint `arch_checkpoint.pth.tar`, you can start to train the classification model with the learned bit allocation. 
```
python main.py \
  -a quantres18_cfg --epochs 95 --step-epoch 30 -j 16 \
  --ac arch_checkpoint.pth.tar \
  [your imagenet-folder with train and val folders]
```

### Results

The results are shown as following:
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">network</th>
<th valign="bottom">precision</th>
<th valign="bottom">bit</th>
<th valign="bottom">--cd</th>
<th valign="bottom">top-1/5 acc.</th>
<th valign="bottom">model</th>
<!-- TABLE BODY -->
<tr><td align="left">ResNet-18</td>
<td align="center">uniform</td>
<td align="center">2.0</td>
<td align="center"></td>
<td align="center">65.1/86.2</td>
<td align="center">download</td>
</tr>
<tr><td align="left">ResNet-18</td>
<td align="center">mixed</td>
<td align="center">1.992</td>
<td align="center">0.00335</td>
<td align="center">65.9/86.5</td>
<td align="center">download</td>
</tr>
<tr><td align="left">ResNet-50</td>
<td align="center">uniform</td>
<td align="center">2.0</td>
<td align="center"></td>
<td align="center">70.6/89.8</td>
<td align="center">download</td>
</tr>
<tr><td align="left">ResNet-50</td>
<td align="center">mixed</td>
<td align="center">2.007</td>
<td align="center">0.00015</td>
<td align="center">72.1/90.6</td>
<td align="center">download</td>
</tr>
<tr><td align="left">GoogleNet</td>
<td align="center">uniform</td>
<td align="center">2.0</td>
<td align="center"></td>
<td align="center">64.8/86.3</td>
<td align="center">download</td>
</tr>
<tr><td align="left">GoogleNet</td>
<td align="center">mixed</td>
<td align="center">1.994</td>
<td align="center">0.00045</td>
<td align="center">67.8/88.0</td>
<td align="center">download</td>
</tr>
<tr><td align="left">Inception-V3</td>
<td align="center">uniform</td>
<td align="center">2.0</td>
<td align="center"></td>
<td align="center">71.0/89.9</td>
<td align="center">download</td>
</tr>
<tr><td align="left">Inception-V3</td>
<td align="center">mixed</td>
<td align="center">1.982</td>
<td align="center">0.0015</td>
<td align="center">72.4/90.7</td>
<td align="center">download</td>
</tr>
</tbody></table>


### Disclaimer

1. The training of EdMIPS has some variance. Tune ``--cd`` a little bit to get the optimal bit allocation you want.

2. The BitOps are counted only on the quantized layers. They are normalized to the bit space as in the above table. 

3. Since some changes have been made after the paper submission, you may get slightly worse performances (0.1~0.2 points) than those in the paper. 

If you encounter any issue when using our code/model, please let me know.