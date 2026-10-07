"""MNIST diffusion: noise-prediction training and deterministic DDIM sampling."""
import argparse
import math
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parent


def time_embedding(t, size=32):
    """Encode each integer timestep with sine/cosine waves."""
    frequencies = torch.exp(-math.log(10000) * torch.arange(size // 2) / (size // 2 - 1))
    angles = t.float()[:, None] * frequencies[None, :]
    return torch.cat([angles.sin(), angles.cos()], dim=1)


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time = nn.Linear(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(4, out_channels)
        self.norm2 = nn.GroupNorm(4, out_channels)

    def forward(self, x, embedding):
        x = F.silu(self.norm1(self.conv1(x)))
        x = x + self.time(embedding)[:, :, None, None]
        return F.silu(self.norm2(self.conv2(x)))


class SmallUNet(nn.Module):
    """28 -> 14 -> 7 -> 14 -> 28; skip connections keep spatial details."""
    def __init__(self):
        super().__init__()
        self.time = nn.Sequential(nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 32))
        self.down1 = Block(1, 16)
        self.down2 = Block(16, 32)
        self.middle = Block(32, 32)
        self.up2 = Block(64, 16)
        self.up1 = Block(32, 16)
        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, x, t):
        embedding = self.time(time_embedding(t))
        skip1 = self.down1(x, embedding)
        skip2 = self.down2(F.avg_pool2d(skip1, 2), embedding)
        x = self.middle(F.avg_pool2d(skip2, 2), embedding)
        x = F.interpolate(x, size=skip2.shape[-2:], mode='nearest')
        x = self.up2(torch.cat([x, skip2], dim=1), embedding)
        x = F.interpolate(x, size=skip1.shape[-2:], mode='nearest')
        x = self.up1(torch.cat([x, skip1], dim=1), embedding)
        return self.out(x)  # Predict noise, not a digit label.


def make_schedule(steps=1000):
    """Original linear beta schedule; alpha_bar is cumulative signal power."""
    beta = torch.linspace(0.0001, 0.02, steps)
    return torch.cumprod(1 - beta, dim=0)


def add_noise(clean, t, noise, alpha_bar):
    """q(x_t | x_0): mix the clean image with known Gaussian noise."""
    a = alpha_bar[t][:, None, None, None]
    return a.sqrt() * clean + (1 - a).sqrt() * noise


@torch.no_grad()
def sample(model, alpha_bar, count=64, steps=50, seed=123):
    """DDIM with eta=0: only the initial image is random; later steps are deterministic."""
    if not 1 <= steps <= len(alpha_bar):
        raise ValueError('sampling steps must be between 1 and the training steps')
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(count, 1, 28, 28, generator=generator)
    # Visit a subset of training timesteps, from most noisy to least noisy.
    timesteps = torch.linspace(len(alpha_bar) - 1, 0, steps).long().tolist()
    for i, t in enumerate(timesteps):
        previous_t = timesteps[i + 1] if i + 1 < steps else -1
        a = alpha_bar[t]
        # -1 means the clean endpoint, with no remaining noise.
        previous_a = alpha_bar[previous_t] if previous_t >= 0 else torch.tensor(1.0)
        predicted_noise = model(x, torch.full((count,), t, dtype=torch.long))
        predicted_clean = (x - (1 - a).sqrt() * predicted_noise) / a.sqrt()
        predicted_clean = predicted_clean.clamp(-1, 1)
        x = previous_a.sqrt() * predicted_clean + (1 - previous_a).sqrt() * predicted_noise
    return (x.clamp(-1, 1) + 1) / 2  # Back to [0, 1] for saving.


def save_grid(images, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(images, path, nrow=8, padding=2)


def train(args):
    torch.manual_seed(args.seed)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST(ROOT / 'data', train=True, download=True, transform=transform)
    if args.limit:
        # A seeded random subset, rather than just the first digits.
        indices = torch.randperm(len(dataset))[:args.limit].tolist()
        dataset = Subset(dataset, indices)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = SmallUNet()  # Everything stays on CPU.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    alpha_bar = make_schedule()
    output = ROOT / 'outputs'
    output.mkdir(exist_ok=True)
    started = time.perf_counter()
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for clean, _ in loader:  # Labels are unused: generation is unconditional.
            t = torch.randint(len(alpha_bar), (len(clean),))
            noise = torch.randn_like(clean)
            noisy = add_noise(clean, t, noise, alpha_bar)
            loss = F.mse_loss(model(noisy, t), noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(clean)
            seen += len(clean)
        mean_loss = total_loss / seen
        history.append(mean_loss)
        save_grid(sample(model, alpha_bar, steps=args.sample_steps), output / f'epoch_{epoch:03d}.png')
        torch.save({'model': model.state_dict(), 'alpha_bar': alpha_bar,
                    'epochs': epoch, 'seed': args.seed, 'examples': len(dataset),
                    'loss_history': history}, output / 'model.pt')
        print(f'Epoch {epoch}/{args.epochs}: loss={mean_loss:.4f}, elapsed={time.perf_counter() - started:.1f}s', flush=True)
    print(f'Checkpoint and image grids: {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=4, help='CPU threads; try 1, 2 or 4')
    commands = parser.add_subparsers(dest='command', required=True)
    training = commands.add_parser('train')
    training.add_argument('--epochs', type=int, default=10)
    training.add_argument('--batch-size', type=int, default=128)
    training.add_argument('--limit', type=int, default=0, help='0: full MNIST; e.g. 10000: quick experiment')
    training.add_argument('--seed', type=int, default=42)
    training.add_argument('--sample-steps', type=int, default=50)
    sampling = commands.add_parser('sample')
    sampling.add_argument('--checkpoint', type=Path, default=ROOT / 'outputs/model.pt')
    sampling.add_argument('--output', type=Path, default=ROOT / 'outputs/generated.png')
    sampling.add_argument('--steps', type=int, default=50)
    sampling.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    if args.command == 'train':
        if args.epochs < 1 or args.batch_size < 1 or args.limit < 0 or not 1 <= args.sample_steps <= 1000:
            parser.error('use positive epochs/batch size, a nonnegative limit and 1..1000 sample steps')
        train(args)
    else:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        model = SmallUNet()
        model.load_state_dict(checkpoint['model'])
        save_grid(sample(model, checkpoint['alpha_bar'], steps=args.steps, seed=args.seed), args.output)
        print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
