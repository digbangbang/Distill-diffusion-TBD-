import argparse
import torch
import os
import numpy as np
import yaml
from models.diffusion import Model
from PIL import Image
from tqdm import tqdm

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--config", type=str, default="cifar10_sparse.yml", help="Path to the config file"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="CUDA DEVICE"
    )
    parser.add_argument(
        "--time_step", type=int, default=8, help="timesteps of back process, choose from [8,32,250,500]"
    )
    parser.add_argument("--use_pretrained", action="store_true")
    # parser.add_argument(
    #     "--ckpt", type=str, default="output/logs/CIFAR10_pd_dis_to_4/distill_ckpt_16_step.pth", help="model .pth path"
    # )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="eta used to control the variances of sigma",
    )
    parser.add_argument(
        "--beta_start",
        type=float,
        default=0.0001,
        help="mask prune rate",
    )
    parser.add_argument(
        "--beta_end",
        type=float,
        default=0.02,
        help="mask prune rate",
    )
    parser.add_argument(
        "--num_diffusion_timesteps", type=int, default=1000, help="number of steps involved"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    return args, new_config

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def generalized_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in tqdm(zip(reversed(seq), reversed(seq_next)), desc="Sampling",total=len(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            et = model(xt, t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds

def main():
    args, config = parse_args_and_config()
    if not args.use_pretrained:
        ckpt = f"output/logs/CIFAR10_pd_dis_to_4/distill_ckpt_{args.time_step}_step.pth"
        states = torch.load(ckpt, map_location=args.device)
        model = Model(config)
        model = model.to(args.device)
        model.load_state_dict(states[0],strict=True)
        time_step = states[3]
    else:
        ckpt = f"pre_trained_model/model-790000.ckpt"
        model = Model(config)
        model.load_state_dict(torch.load(ckpt, map_location=args.device))
        model.to(args.device)
        time_step=np.arange(0,1000)[::int(1000/args.time_step)]

    betas = get_beta_schedule(
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        num_diffusion_timesteps=args.num_diffusion_timesteps,
    )
    betas = torch.from_numpy(betas).float().to(args.device)

    x = torch.randn(
        8,
        config.data.channels,
        config.data.image_size,
        config.data.image_size,
        device=args.device,
    )

    xs, x0 = generalized_steps(x, time_step, model, betas, eta=args.eta)
    x = xs[-1]
    image = (x / 2 + 0.5).clamp(0, 1)

    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    if not args.use_pretrained:
        for num, im in enumerate(pil_images):
            im.save(f"sample/distill_time_step{args.time_step}_{num}.png")
    else:
        for num, im in enumerate(pil_images):
            im.save(f"sample/prtrain_time_step{args.time_step}_{num}.png")

if __name__ == "__main__":
    main()



