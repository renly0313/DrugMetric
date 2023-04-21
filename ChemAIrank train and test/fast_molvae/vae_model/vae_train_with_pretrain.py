import sys
sys.path.append('../../')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable

import math, random, sys
import numpy as np
import argparse
from collections import deque
import pickle as pickle

from fast_jtnn import *
import rdkit
from tqdm import tqdm
import os

# def make_print_to_file(path='./log/'):
#     '''
#     path， it is a path for save your log about fuction print
#     example:
#     use  make_print_to_file()   and the   all the information of funtion print , will be write in to a log file
#     :return:
#     '''
#     class Logger(object):
#         def __init__(self, filename="Default.log", path="./log/"):
#             self.terminal = sys.stdout
#             self.log = open(os.path.join(path, filename), "a", encoding='utf8', )
#
#         def write(self, message):
#             self.terminal.write(message)
#             self.log.write(message)
#
#         def flush(self):
#             pass
#
#     fileName = ( 'pre_zinc250_vocab784_e3.log' )
#     sys.stdout = Logger(fileName + '.log', path=path)

    #############################################################
    # 这里输出之后的所有的输出的print 内容即将写入日志
    #############################################################
 #   print(fileName.center(60, '*'))

def main_vae_train(train,
             vocab,
             model_path,
             save_dir,
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
    #print(model)
    # for all multi-dimensional parameters, initialize them using xavier initialization
    # for one-dinimensional parameters, initialize them to 0
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    else:
        for param in model.parameters():
            if param.dim() == 1:
                nn.init.constant(param, 0)
            else:
                nn.init.xavier_normal(param)

    # for param in model.parameters():
    #     if param.dim() == 1:
    #         nn.init.constant_(param, 0)
    #     else:
    #         nn.init.xavier_normal_(param)
    #
    if os.path.isdir(save_dir) is False:
        os.makedirs(save_dir)
    
    if load_epoch > 0:
        model.load_state_dict(torch.load(save_dir + "/model.epoch-" + str(load_epoch)))

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
    train_processed_list = []
    meters = np.zeros(5)

    for epoch in tqdm(range(epoch)):
        loader = MolTreeFolder(train, vocab, batch_size)#, num_workers=4)
        for batch in loader:
            total_step += 1
            try:
                model.zero_grad()
                loss, kl_div, wacc, tacc, sacc, mol_tree_mean, mol_tree_std= model(batch, beta)
                loss.backward()
                # implement gradient clipping 实现梯度裁剪
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
            except Exception as e:
                print(e)
                continue
            loss1 = loss.detach().cpu().numpy()
            meters = meters + np.array([kl_div, loss1, wacc * 100, tacc * 100, sacc * 100])
            if total_step % print_iter == 0:
                meters /= print_iter
                # make_print_to_file(path='../data/log/')
                print("[%d] Beta: %.3f, KL: %.2f, loss: %.2f, Word: %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f" % (total_step, beta, meters[0], meters[1], meters[2], meters[3], meters[4], param_norm(model), grad_norm(model)))

                # torch.save(mol_tree_mean, "./gbd_mol_tree_mean.pt")
                # torch.save(mol_tree_std, "./gbd_mol_tree_std.pt")
                train_processed_list.append((mol_tree_mean, mol_tree_std))



                # y = torch.load("./myTensor.pt")#读取tensor
                # print(y)

                # final_mean.append(mol_tree_mean.detach().cpu().numpy())
                #
                # final_std.append(mol_tree_std.detach().cpu().numpy())

                sys.stdout.flush()
                meters *= 0

            if total_step % save_iter == 0:
                torch.save(model.state_dict(), save_dir + "/model.iter-" + str(total_step))

            if total_step % anneal_iter == 0:
                scheduler.step()
                print("learning rate: %.6f" % scheduler.get_lr()[0])

            if total_step % kl_anneal_iter == 0 and total_step >= warmup:
                beta = min(max_beta, beta + step_beta)

        scheduler.step()
#         torch.save(model.state_dict(), save_dir + "/model.epoch-" + str(epoch))
    torch.save(model.state_dict(), save_dir + "/model.epoch-" + str(epoch))

    with open(os.path.join(save_data, 'pre_zinc250_zinc_rand3_total_mean_std.pkl'), 'wb') as f:
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
    setup(seed=811048)
    parser = argparse.ArgumentParser()
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    # parser.add_argument('--train', required=True)
    # parser.add_argument('--vocab', required=True)
    # parser.add_argument('--save_dir', required=True)

    parser.add_argument('--train', type=str, default="../data/datasets/zinc/new_sample/zinc_rand3_processed")
    parser.add_argument('--vocab', type=str, default="../data/vocab/all_data_vocab.txt")
    parser.add_argument('--model_path', type=str, default="../data/save_model/pre_zinc250_model/model.epoch-2")
    parser.add_argument('--save_dir', type=str, default="../data/save_model/pre_zinc250_zinc_rand3_processed_model")
    parser.add_argument('--save_data', type=str, default="../data/latent_space_result/pre_zinc250_zinc_rand3")

    parser.add_argument('--load_epoch', type=int, default=0)

    parser.add_argument('--hidden_size', type=int, default=450)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--latent_size', type=int, default=64)
    parser.add_argument('--depthT', type=int, default=20)
    parser.add_argument('--depthG', type=int, default=3)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--clip_norm', type=float, default=50.0)
    parser.add_argument('--beta', type=float, default=0.001)
    parser.add_argument('--step_beta', type=float, default=0.002)
    parser.add_argument('--max_beta', type=float, default=1.0)
    parser.add_argument('--warmup', type=int, default=40000, help='20000, 40000')

    parser.add_argument('--epoch', type=int, default=20)
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
             args.save_dir,
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
    
