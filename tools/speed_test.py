# https://github.com/zhijian-liu/torchprofile
import time
import wandb
import torch
from torchprofile import profile_macs

from fgir_backbones.model_utils.build_model import build_model
from fgir_backbones.other_utils.build_args import parse_train_args


def count_params(model):
    return sum([p.numel() for p in model.parameters()])


def count_params_trainable(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def main():

    args = parse_train_args()

    args.num_classes = 200

    if not args.debugging and not args.offline:
        wandb.init(config=args, project=args.project_name)
        wandb.run.name = args.run_name

    model = build_model(args)
    model.eval()

    x = torch.rand(1, 3, args.image_size, args.image_size).to(args.device)

    # summary stats
    macs = profile_macs(model, x) / 1e9
    max_memory = torch.cuda.max_memory_reserved() / (1024 ** 3)
    no_params = count_params(model) / 1e6
    no_params_trainable = count_params_trainable(model) / 1e6

    # batched inference
    start = time.time()

    for _ in range(args.test_multiple):
        x = torch.rand(args.batch_size, 3, args.image_size, args.image_size).to(args.device, non_blocking=True)
        with torch.no_grad():
            model(x)

    torch.cuda.synchronize()
    time = time.time() - start

    throughput = (args.batch_size * args.test_multiple) / time

    print(f'''{args.model_name}\n
          GMACs: {macs}\n
          TP (BS={args.batch_size}): {throughput}\n

          ''')

    if not args.vis_errors and not args.offline:
        wandb.run.summary['throughput'] = throughput
        wandb.run.summary['gmac'] = macs
        wandb.run.summary['max_memory'] = max_memory
        wandb.run.summary['no_params'] = no_params
        wandb.run.summary['no_params_trainable'] = no_params_trainable

        wandb.finish()


if __name__ == "__main__":
    main()
