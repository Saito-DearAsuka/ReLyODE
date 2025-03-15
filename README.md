
## Downloading Datasets

1. [Turbulent Radiative Layer - 2D](https://polymathic-ai.org/the_well/datasets/turbulent_radiative_layer_2D/)
2. [Rayleigh-Bénard convection](https://polymathic-ai.org/the_well/datasets/rayleigh_benard/)
3. [Bouncing Ball](https://drive.google.com/drive/folders/1bPM1ld3KEk_3hZK64MzJuNHGWWz8X5PP)


## Training

To train a model on specific dataset, run:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --phase train --dataset bball
```

All arguments used for this project are described in the function "get_opt()" in ```main.py```. There are a lot of options to train our network on a wide range of datasets and also to evaluate various architectures for writing the paper. However, just for the purpose of executing the proposed method, the number of arguments that you need to change would be very limited.
The following options will be what you need to concern:


```--dataset``` : Specify the dataset to train, select among [kth, penn, mgif, hurricane].<br>
```--extrap``` : If you toggle this option, you can train the extrapolation model.<br>
```--irregular``` : If you toggle this option, you can train the model with irregularly sampled frames.<br>


## Evaluation

We evaluate our model using Structural Similarity (SSIM), Peak Signal-to-Noise Ratio (PSNR), and Learned Perceptual Image Patch Similarity (LPIPS). To evaluate a model on specific dataset, run:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --phase test_met --dataset kth --test_dir CHECKPOINT_DIR
```


## Citation
If you find this work useful for your research, please cite our [paper](https://arxiv.org/abs/2010.08188):

```
@article{park2020vid,
  title={Vid-ODE: Continuous-Time Video Generation with Neural Ordinary Differential Equation},
  author={Park, Sunghyun and Kim, Kangyeol and Lee, Junsoo and Choo, Jaegul and Lee, Joonseok and Kim, Sookyung and Choi, Edward},
  journal={arXiv preprint arXiv:2010.08188},
  booktitle={The Thirty-Fifth {AAAI} Conference on Artificial Intelligence, {AAAI} 2021},
  pages={online},
  publisher={{AAAI} Press},
  year={2021},
}
```
