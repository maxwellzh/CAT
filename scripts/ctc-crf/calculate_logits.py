"""
Copyright 2021 Tsinghua University
Apache 2.0.
Author: Hongyu Xiang, Keyu An, Zheng Huahuan
"""

import json
import utils
import argparse
import kaldi_io
import numpy as np
from tqdm import tqdm
from train import build_model
from dataset import InferDataset

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def main(args):
    if not torch.cuda.is_available():
        utils.highlight_msg("Using CPU.")
        single_worker('cpu', args.nj, args)
        return None

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
    print(f"> Global number of GPUs: {args.world_size}")
    num_jobs = args.nj
    if num_jobs <= ngpus_per_node:
        utils.highlight_msg(
            f"Number of jobs (--nj={num_jobs}) is too small.\nUse only one GPU for avoiding errors.")
        single_worker("cuda:0", num_jobs, args)
        return None

    inferset = InferDataset(args.input_scp)
    res = len(inferset) % args.world_size

    if res == 0:
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args, num_jobs))
        return None
    else:
        # This is a hack for non-divisible length of data to number of GPUs
        utils.highlight_msg("Using hack to deal with undivisible data length.")
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args, num_jobs-1))
        single_worker("cuda:0", 1, args, len(inferset)-res)


def main_worker(gpu, ngpus_per_node, args, num_jobs):
    args.gpu = gpu

    args.rank = args.rank * ngpus_per_node + gpu
    print(f"  Use GPU: local[{args.gpu}] | global[{args.rank}]")
    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    world_size = dist.get_world_size()
    local_writers = [open(f"{args.output_dir}/decode.{i+1}.ark", "wb")
                     for i in range(args.rank, num_jobs, world_size)]

    inferset = InferDataset(args.input_scp)
    res = len(inferset) % args.world_size
    if res > 0:
        inferset.dataset = inferset.dataset[:-res]

    dist_sampler = DistributedSampler(inferset)
    dist_sampler.set_epoch(1)

    testloader = DataLoader(
        inferset, batch_size=1, shuffle=(dist_sampler is None),
        num_workers=args.workers, pin_memory=True,
        sampler=dist_sampler)

    with open(args.config, 'r') as fi:
        configures = json.load(fi)

    model = build_model(args, configures, train=False)

    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    model.load_state_dict(torch.load(
        args.resume, map_location=f"cuda:{args.gpu}"))
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.gpu])
    model.eval()

    if args.rank == 0:
        print("> Model built.")
        print("  Model size:{:.2f}M".format(
            utils.count_parameters(model)/1e6))

    cal_logit(model, testloader, args.gpu, local_writers)


def single_worker(device, num_jobs, args, idx_beg=0):

    if idx_beg > 0 and num_jobs == 1:
        local_writers = [open(f"{args.output_dir}/decode.{args.nj}.ark", 'wb')]
    else:
        local_writers = [open(f"{args.output_dir}/decode.{i+1}.ark", 'wb')
                         for i in range(num_jobs)]

    inferset = InferDataset(args.input_scp)
    inferset.dataset = inferset.dataset[idx_beg:]

    testloader = DataLoader(
        inferset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    with open(args.config, 'r') as fi:
        configures = json.load(fi)

    model = build_model(args, configures, train=False)

    model = model.to(device)
    model.load_state_dict(torch.load(
        args.resume, map_location=device))
    model.eval()

    print("> Model built.")
    print("  Model size:{:.2f}M".format(
        utils.count_parameters(model)/1e6))

    cal_logit(model, testloader, device, local_writers)


def cal_logit(model, testloader, device, local_writers):
    results = []
    with torch.no_grad():
        for batch in tqdm(testloader):
            key, x, x_lens = batch
            x_lens = x_lens.flatten()
            x = x.to(device, non_blocking=True)
            netout, _ = model.forward(x, x_lens)

            r = netout.cpu().data.numpy()
            r[r == -np.inf] = -1e16
            r = r[0]
            results.append((r, key[0]))

    num_local_writers = len(local_writers)
    for i, (r, utt) in enumerate(results):
        kaldi_io.write_mat(
            local_writers[i % num_local_writers], r, key=utt)

    for write in local_writers:
        write.close()
    return None


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Logit calculation")
    parser.add_argument("--input_scp", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--config", type=str, default=None, metavar='PATH',
                        help="Path to configuration file of backbone.")

    parser.add_argument("--nj", type=int)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to location of checkpoint.")

    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 1)')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:12947', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')

    args = parser.parse_args()

    main(args)
