#!/usr/bin/env python
import argparse

from Local.server import LocalServer

#####
# Communication-Efficient Learning of Deep Networks from Decentralized Data
from FedAvg.server import AvgServer
# Federated Optimization in Heterogeneous Networks
from FedProx.server import ProxServer
# 
from FedBABU.server import BABUServer
#
from FedRep.server import RepServer
#
from FedAMP.server import AMPServer
# FedGen
from FedGen.server import GenServer
#
from pFedPR.server import pPRServer

#####
from FedProto.server import ProtoServer
from FedPR.server import PRServer

import torch, numpy as np, random


def main(args):
    print("\n\n         [ Start training ]           \n\n")
    if args.device == 'cuda':
        torch.cuda.set_device(0)
        torch.cuda.manual_seed(114)
    torch.manual_seed(114)
    np.random.seed(114)
    random.seed(114)
    if args.algorithm == "Local":
        server = LocalServer(args)
    elif args.algorithm == "FedProto":
        server = ProtoServer(args)
    elif args.algorithm == "FedRep":
        server = RepServer(args)
    elif args.algorithm == "FedAvg" or args.algorithm == "pFedAvg":
       server = AvgServer(args)
    elif args.algorithm == "FedBABU":
        server = BABUServer(args)
    elif args.algorithm == "FedProx":
        server = ProxServer(args)
    elif args.algorithm == "FedGen":
        server = GenServer(args)
    elif args.algorithm == "FedAMP":
        server = AMPServer(args)
    elif args.algorithm == "FedPR":
        server = PRServer(args)
    elif args.algorithm == "pFedPR+":
        server = pPRServer(args)
    mertics = server.proceed()
    print("\n\n         [ Finished training ]           \n\n")
    best_iter = np.argmax(mertics["global_avg_acc"])
    best_global = mertics["global_avg_acc"][best_iter]
    print("global_avg_acc: %.2f" % best_global)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--dataset", type=str, default="mnist")
    # 数据分布，可选：pathological、latent-distribution、practical、natural
    parser.add_argument("--sampling", type=str, default="pathological")
    # pathological: num of shards per user
    parser.add_argument("--n", type=int, default=2)
    # 算法
    parser.add_argument("--algorithm", type=str, default="Local")
    parser.add_argument("--gamma", type=float, default=0.0)
    # 本地模型
    parser.add_argument("--model_name", type=str, default='mlp', help="Local model name")
    # 本地模型的初始学习率
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Local learning rate")
    # 生成模型的学习率
    parser.add_argument("--generator_lr", type=float, default=0.001, help="Generator learning rate")
    # 1 次实验 进行 num_glob_iters 轮通信
    parser.add_argument("--num_glob_iters", type=int, default=100)
    # 共有 num_users 个客户端
    parser.add_argument("--num_users", type=int, default=100, help="Number of Users")
    # 1 轮通信 选择 per_users 个客户端
    parser.add_argument("--per_users", type=int, default=100, help="Number of Users per round")
    # 1 轮通信 本地模型 训练 local_ep 次
    parser.add_argument("--local_ep", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=50)
    # 
    parser.add_argument("--num_partial_work", type=int, default=0)
    # 决策层初始化
    parser.add_argument("--param_init", type=str, default='uniform')

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # args.device = 'cpu'
    if args.dataset=='mnist' or args.dataset=='fashion' or args.dataset=='femnist' or args.dataset=='cifar10':
        args.num_classes = 10
    elif args.dataset.startswith('synthetic'):
        args.num_classes = 4

    main(args)

    
