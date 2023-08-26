# -*- coding: utf-8 -*-
import os
import argparse
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

from models.classification_heads import ClassificationHead
from models.protonet_embedding_5l import ProtoNetEmbedding
from models.graph_match_head import SimGNN

from utils import set_gpu, Timer, count_accuracy, check_dir, log

def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.
        
    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """
    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1,index,1)
    
    return encoded_indicies

def get_model(options):
    # Choose the embedding network
    if options.network == 'ProtoNet':
        network = ProtoNetEmbedding().cuda()
        fea_dim = 256
    else:
        print ("Cannot recognize the network type")
        assert(False)
        
    # Choose the classification head
    if options.head == 'GraphMatchNet':
        cls_head = SimGNN().cuda()
    else:
        print ("Cannot recognize the dataset type")
        assert(False)
    return (network, cls_head)


def get_dataset(options):
    # Choose the embedding network
    if options.dataset == 'UCML':
        from data.ucml import UCML, FewShotDataloader
        dataset_train = UCML(phase='train')
        dataset_val = UCML(phase='val')
        dataset_test = UCML(phase='test')
        data_loader = FewShotDataloader
    else:
        print("Cannot recognize the dataset type")
        assert (False)

    return (dataset_train, dataset_val, dataset_test, data_loader)

def weight_mse_loss(logits, target):
    logits = logits.reshape(-1)
    target = target.reshape(-1)
    index_one = torch.gt(target, 0.5)
    index_zeros = torch.lt(target, 0.5)
    logits_ones = torch.masked_select(logits, index_one)
    logits_zeros = torch.masked_select(logits, index_zeros)
    target_ones = torch.masked_select(target, index_one)
    target_zeros = torch.masked_select(target, index_zeros)

    # mse loss
    Ej = F.mse_loss(logits_ones, target_ones)
    Em = F.mse_loss(logits_zeros, target_zeros)
    loss = Ej+Em

    return loss

def seed_torch(seed=21):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.deterministic = True  #cpu/gpu结果一致
    torch.backends.cudnn.benchmark = False   #训练集变化不大时使训练加速

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-epoch', type=int, default=60,
                            help='number of training epochs')
    parser.add_argument('--save-epoch', type=int, default=10,
                            help='frequency of model saving')
    parser.add_argument('--train-shot', type=int, default=1,
                            help='number of support examples per training class')
    parser.add_argument('--val-shot', type=int, default=1,
                            help='number of support examples per validation class')
    parser.add_argument('--train-query', type=int, default=15,
                            help='number of query examples per training class')
    parser.add_argument('--val-episode', type=int, default=600,
                            help='number of episodes per validation')
    parser.add_argument('--val-query', type=int, default=15,
                            help='number of query examples per validation class')
    parser.add_argument('--train-way', type=int, default=5,
                            help='number of classes in one training episode')
    parser.add_argument('--test-way', type=int, default=5,
                            help='number of classes in one test (or validation) episode')
    parser.add_argument('--save-path', default='./experiments/exp_1')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--network', type=str, default='ProtoNet',
                            help='choose which embedding network to use. ProtoNet, R2D2, ResNet')
    parser.add_argument('--head', type=str, default='GraphMatchNet',
                            help='choose which classification head to use. ProtoNet, Ridge, R2D2, SVM')
    parser.add_argument('--dataset', type=str, default='UCML',
                            help='choose which classification head to use. miniImageNet, tieredImageNet, CIFAR_FS, FC100')
    parser.add_argument('--episodes-per-batch', type=int, default=8,
                            help='number of episodes per batch')
    parser.add_argument('--eps', type=float, default=0.0,
                            help='epsilon of label smoothing')

    opt = parser.parse_args()
    seed_torch(0)

    (dataset_train, dataset_val, dataset_test, data_loader) = get_dataset(opt)

    # Dataloader of Gidaris & Komodakis (CVPR 2018)
    dloader_train = data_loader(
        dataset=dataset_train,
        nKnovel=opt.train_way,
        nKbase=0,
        nExemplars=opt.train_shot, # num training examples per novel category
        nTestNovel=opt.train_way * opt.train_query, # num test examples for all the novel categories
        nTestBase=0, # num test examples for all the base categories
        batch_size=opt.episodes_per_batch,
        num_workers=4,
        epoch_size=opt.episodes_per_batch * 10, # num of batches per epoch
    )

    dloader_val = data_loader(
        dataset=dataset_val,
        nKnovel=opt.test_way,
        nKbase=0,
        nExemplars=opt.val_shot, # num training examples per novel category
        nTestNovel=opt.val_query * opt.test_way, # num test examples for all the novel categories
        nTestBase=0, # num test examples for all the base categories
        batch_size=1,
        num_workers=0,
        epoch_size=1 * opt.val_episode, # num of batches per epoch
    )

    dloader_test = data_loader(
        dataset=dataset_test,
        nKnovel=opt.test_way,
        nKbase=0,
        nExemplars=opt.val_shot, # num training examples per novel category
        nTestNovel=opt.val_query * opt.test_way, # num test examples for all the novel categories
        nTestBase=0, # num test examples for all the base categories
        batch_size=1,
        num_workers=0,
        epoch_size=1 * opt.val_episode, # num of batches per epoch
    )

    set_gpu(opt.gpu)
    check_dir('./experiments/')
    check_dir(opt.save_path)
    torch.backends.cudnn.enabled = False

    log_file_path = os.path.join(opt.save_path, "train_log.txt")
    log(log_file_path, str(vars(opt)))

    (embedding_net, cls_head) = get_model(opt)
    saved_models = torch.load(os.path.join(opt.save_path, 'ucml_best_dist_rotate_pretrain_model_resnet_{}_1s.pth'.format(opt.network)))
    # saved_models = torch.load(os.path.join(opt.save_path, 'ucml_epoch_60_pretrain_5s.pth'))
    embedding_net.load_state_dict(saved_models['embedding'])
    embedding_net.eval()

    optimizer = torch.optim.Adam([{'params': cls_head.parameters()}], lr=0.00001, weight_decay=5e-4)

    lambda_epoch = lambda e: 1.0 if e < 60 else (0.1 if e < 40 else 0.01 if e < 50 else (0.001))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch, last_epoch=-1)

    max_val_acc = 0.0
    max_test_acc = 0.0

    timer = Timer()
    x_entropy = torch.nn.CrossEntropyLoss()

    for epoch in range(1, opt.num_epoch + 1):
        # Train on the training split
        lr_scheduler.step()

        # Fetch the current epoch's learning rate
        epoch_learning_rate = 0.1
        for param_group in optimizer.param_groups:
            epoch_learning_rate = param_group['lr']

        log(log_file_path, 'Train Epoch: {}\tLearning Rate: {:.4f}'.format(
                            epoch, epoch_learning_rate))

        _,  = [x.train() for x in (cls_head, )]

        train_accuracies = []
        train_losses = []

        for i, batch in enumerate(tqdm(dloader_train(epoch)), 1):
            data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in batch]

            train_n_support = opt.train_way * opt.train_shot
            train_n_query = opt.train_way * opt.train_query

            with torch.no_grad():
                emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])), use_pool=False)
                emb_support = emb_support.reshape([opt.episodes_per_batch, train_n_support] + list(emb_support.shape[-3:]))
                data_query = data_query.reshape([-1] + list(data_query.shape[-3:]))
                data_query_1 = data_query[:int(data_query.shape[0]//2)]
                data_query_2 = data_query[int(data_query.shape[0] // 2):]
                emb_query_1 = embedding_net(data_query_1, use_pool=False)
                emb_query_2 = embedding_net(data_query_2, use_pool=False)
                emb_query = torch.cat([emb_query_1, emb_query_2], dim=0)
                emb_query = emb_query.reshape([opt.episodes_per_batch, train_n_query] + list(emb_query.shape[-3:]))
            info_query, pos_neg_label, logit_query = cls_head(F.normalize(emb_support, p=2, dim=2),
                                                              F.normalize(emb_query, p=2, dim=2),
                                                              labels_support, labels_query, opt.train_way)

            info_query = info_query.view(-1)
            pos_neg_label = pos_neg_label.view(-1)

            smoothed_one_hot = pos_neg_label * (1 - opt.eps) + (1 - pos_neg_label) * opt.eps / (opt.train_way - 1)
            loss = weight_mse_loss(logits=info_query, target=smoothed_one_hot)

            acc = count_accuracy(logit_query.reshape(-1, opt.train_way), labels_query.reshape(-1))

            train_accuracies.append(acc.item())
            train_losses.append(loss.item())

            if (i % 10 == 0):
                train_acc_avg = np.mean(np.array(train_accuracies))
                log(log_file_path, 'Train Epoch: {}\tBatch: [{}]\tLoss: {}\tAccuracy: {} % ({} %)'.format(
                            epoch, i, loss.item(), train_acc_avg, acc))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Evaluate on the validation split
        _, _ = [x.eval() for x in (embedding_net, cls_head)]

        val_accuracies = []
        val_losses = []

        for i, batch in enumerate(tqdm(dloader_val(epoch)), 1):
            data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in batch]

            test_n_support = opt.test_way * opt.val_shot
            test_n_query = opt.test_way * opt.val_query
            with torch.no_grad():
                emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])), use_pool=False)
                emb_support = emb_support.reshape(
                    [1, test_n_support] + list(emb_support.shape[-3:]))

                emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])), use_pool=False)
                emb_query = emb_query.reshape([1, test_n_query] + list(emb_query.shape[-3:]))
                info_query, pos_neg_label, logit_query = cls_head(F.normalize(emb_support, p=2, dim=2),
                                                                  F.normalize(emb_query, p=2, dim=2),
                                                                  labels_support, labels_query,
                                                                  opt.test_way)
            info_query = info_query.view(-1)
            pos_neg_label = pos_neg_label.view(-1)

            smoothed_one_hot = pos_neg_label * (1 - opt.eps) + (1 - pos_neg_label) * opt.eps / (opt.train_way - 1)
            loss = weight_mse_loss(logits=info_query, target=smoothed_one_hot)
            # loss = x_entropy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))
            acc = count_accuracy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))

            val_accuracies.append(acc.item())
            val_losses.append(loss.item())

        val_acc_avg = np.mean(np.array(val_accuracies))
        val_acc_ci95 = 1.96 * np.std(np.array(val_accuracies)) / np.sqrt(opt.val_episode)

        val_loss_avg = np.mean(np.array(val_losses))

        if val_acc_avg > max_val_acc:
            max_val_acc = val_acc_avg
            torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict()},\
                       os.path.join(opt.save_path, 'ucml_best_graph_match_{}s_model.pth'.format(opt.val_shot)))
            log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} % (Best)'\
                  .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95))
        else:
            log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} %'\
                  .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95))

        torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict()}\
                   , os.path.join(opt.save_path, 'ucml_last_graph_match_{}s_epoch.pth'.format(opt.val_shot)))

        if epoch % opt.save_epoch == 0:
            torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict()}\
                       , os.path.join(opt.save_path, 'ucml_epoch_graph_match_{}s_{}.pth'.format(opt.val_shot, epoch)))

