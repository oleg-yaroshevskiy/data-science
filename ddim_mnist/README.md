# MNIST with DDIM (2020), in plain PyTorch

A small learning example: train on MNIST digits **on CPU**, then generate new
28×28 digits from Gaussian noise. No diffusion library or trainer framework.
The images are unconditional: digit labels are not used.

This follows the noise-prediction objective and deterministic (`eta = 0`)
sampler from [Denoising Diffusion Implicit Models (2020)](https://arxiv.org/abs/2010.02502).
DDIM uses the same training objective as DDPM; its sampling rule lets us skip
training timesteps. The small U-Net here is for learning, not a reproduction
of the paper's full architecture or results.

## Run

From this folder (Python 3.10+):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python ddim.py train
```

MNIST downloads automatically into `data/` on the first run. Training uses
all 60,000 training images for 10 epochs by default. After **every epoch**, it
writes a grid of 64 generated images and a checkpoint into `outputs/`.
The first grids may look noisy; quality improves with training. Runtime depends
on your CPU. The console prints elapsed time so you can judge it after one epoch.

For a shorter first experiment:

```bash
python ddim.py train --limit 10000 --epochs 5
```

Generate more images after training:

```bash
python ddim.py sample --seed 456 --steps 50
```

The result is `outputs/generated.png`. Change `--seed` for different digits;
change `--steps` to compare 10, 50, and 100 denoising steps. Training always uses
1,000 timesteps. Sampling uses 50 by default. The same checkpoint works with all
these sampling step counts. With the same seed and step count, sampling is
repeatable on the same setup.

`--threads` goes before the command, e.g. `python ddim.py --threads 2 train`.
CPU threads default to 4; more threads are not always faster for a tiny network.
Each training command starts a fresh model and overwrites `outputs/model.pt`;
it does not resume a previous run. Downloads, local environments and outputs
are ignored by Git.

## Read the code in this order

1. `make_schedule`: linear beta values from 0.0001 to 0.02, then
   `alpha_bar = cumprod(1 - beta)`.
2. `add_noise`: create `x_t = sqrt(alpha_bar[t]) * x_0 +
   sqrt(1 - alpha_bar[t]) * noise`, with pixels normalized to [-1, 1].
3. `SmallUNet`: given a noisy image and timestep, predict the added noise.
   Sine/cosine embeddings tell the network how noisy the input is. Skip
   connections join matching resolutions in the down/up paths.
4. `train`: randomly choose timesteps, add known noise, and minimize the
   mean squared error between predicted and actual noise.
5. `sample`: start with random noise, predict the clean image, and move to
   an earlier timestep. At `eta = 0`, the DDIM update is:

   ```text
   predicted_x0 = (x_t - sqrt(1 - a_t) * predicted_noise) / sqrt(a_t)
   x_previous   = sqrt(a_previous) * predicted_x0
                + sqrt(1 - a_previous) * predicted_noise
   ```

   Here `a_t` means `alpha_bar[t]`. The final clean endpoint has `a_previous = 1`.
   We clip the clean prediction to the valid pixel range [-1, 1].
   No fresh noise is added during sampling.

## Included examples

The `examples/` folder contains an actual CPU-generated grid from this script.
See its README for the training settings. Trained checkpoints are kept locally
and excluded from Git; train first, then use the sampling command above.
