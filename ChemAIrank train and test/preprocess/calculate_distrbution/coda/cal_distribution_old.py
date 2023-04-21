import sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import math, random
import numpy as np
import argparse
import pickle as pickle
from fast_jtnn import *
import rdkit
from tqdm import tqdm
import os


def main_vae_train(train,
             vocab,
             model_path,
             save_data,
             load_epoch=0,
             hidden_size=100,
             batch_size=8,
             latent_size=64,
             depthT=20,
             depthG=3,
             lr=1e-3,
             clip_norm=50.0,
             beta=0.0,
             step_beta=0.002,
             max_beta=1.0,
             warmup=40000,
             epoch=20,
             anneal_rate=0.9,
             anneal_iter=40000,
             kl_anneal_iter=2000,
             print_iter=50,
             save_iter=5000):
    
    vocab = [x.strip("\r\n ") for x in open(vocab)] 
    vocab = Vocab(vocab)

    model = JTNNVAE(vocab, int(hidden_size), int(latent_size), int(depthT), int(depthG)).cuda()
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    else:
        for param in model.parameters():
            if param.dim() == 1:
                nn.init.constant(param, 0)
            else:
                nn.init.xavier_normal(param)
    print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # exponentially decay the learning rate学习率呈指数衰减
    scheduler = lr_scheduler.ExponentialLR(optimizer, anneal_rate)
    #scheduler.step()

    # 先不管这部分
    param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
    grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))
    if os.path.isdir(save_data) is False:
        os.makedirs(save_data)
    total_step = load_epoch
    beta = beta
    mol_tree_mean = []
    train_processed_list = []
    mol_count = 0  # 添加一个分子计数器
    with torch.no_grad():
        for epoch in tqdm(range(epoch)):
            loader = MolTreeFolder(train, vocab, batch_size)  # , num_workers=4)
            for batch in loader:
                mol_count += len(batch[0])  # 更新分子计数器
                total_step += 1
                # mol_tree_mean = model.forward_dis(batch)
                # train_processed_list.append([mol_tree_mean])
                # mol_tree_mean = np.stack(mol_tree_mean)
                try:
                    mol_tree_mean = model.forward_dis(batch)
                    mol_tree_mean = mol_tree_mean.cpu().detach().numpy()
                    train_processed_list.append(mol_tree_mean)
                except Exception as e:
                    print(e)
                    print("出错的分子在源数据中的索引:", mol_count - len(batch[0]))
                    continue


    train_processed_list = np.vstack(train_processed_list)

    with open(os.path.join(save_data, 'seprated_chembl_val_mean.pkl'), 'wb') as f:
        pickle.dump(train_processed_list, f)
    return model
def setup(seed):
    # frozen random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



if __name__ == '__main__':
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)
    setup(seed=727956)
    parser = argparse.ArgumentParser()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


    parser.add_argument('--train', type=str, default="/home/dell/wangzhen/RealQED(2.17)/test/data/docking/affinity_large_than_8_processed")
    parser.add_argument('--vocab', type=str, default="/home/dell/wangzhen/RealQED(2.17)/data/vocab/all_data_vocab.txt")
    parser.add_argument('--model_path', type=str, default="/home/dell/wangzhen/RealQED(2.17)/data/save_model/pre_zinc250_mix_rand1_processed_model1206/model.epoch-19")
    parser.add_argument('--save_data', type=str, default="/home/dell/wangzhen/RealQED(2.17)/test/data/docking/affinity_large_than_8_mean")

    parser.add_argument('--load_epoch', type=int, default=0)

    parser.add_argument('--hidden_size', type=int, default=450)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--latent_size', type=int, default=64)
    parser.add_argument('--depthT', type=int, default=20)
    parser.add_argument('--depthG', type=int, default=3)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--clip_norm', type=float, default=50.0)
    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--step_beta', type=float, default=0.002)
    parser.add_argument('--max_beta', type=float, default=1.0)
    parser.add_argument('--warmup', type=int, default=40000, help='20000, 40000')

    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--anneal_rate', type=float, default=0.9)
    parser.add_argument('--anneal_iter', type=int, default=40000)
    parser.add_argument('--kl_anneal_iter', type=int, default=1000)
    parser.add_argument('--print_iter', type=int, default=50)
    parser.add_argument('--save_iter', type=int, default=5000)


    args = parser.parse_args()
    print(args)

    main_vae_train(args.train,
                   args.vocab,
                   args.model_path,
             args.save_data,
             args.load_epoch,
             args.hidden_size,
             args.batch_size,
             args.latent_size,
             args.depthT,
             args.depthG,
             args.lr,
             args.clip_norm,
             args.beta,
             args.step_beta,
             args.max_beta,
             args.warmup,
             args.epoch,
             args.anneal_rate,
             args.anneal_iter,
             args.kl_anneal_iter,
             args.print_iter,
             args.save_iter)

