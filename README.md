
## Downloading Datasets

1. [Turbulent Radiative Layer - 2D](https://polymathic-ai.org/the_well/datasets/turbulent_radiative_layer_2D/)
2. [Rayleigh-Bénard convection](https://polymathic-ai.org/the_well/datasets/rayleigh_benard/)
3. [Bouncing Ball](https://drive.google.com/drive/folders/1bPM1ld3KEk_3hZK64MzJuNHGWWz8X5PP)


## Training

To train a model on specific dataset, run:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --phase train --dataset bball
```


## Evaluation

We evaluate our model using Structural Similarity (SSIM), Peak Signal-to-Noise Ratio (PSNR), and Learned Perceptual Image Patch Similarity (LPIPS). To evaluate a model on specific dataset, run:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --phase test_met --dataset bball --test_dir CHECKPOINT_DIR
```




