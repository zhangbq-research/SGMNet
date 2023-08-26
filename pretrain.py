# -*- coding: utf-8 -*-
import os
import argparse
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

from models.classification_heads import ClassificationHead
from models.R2D2_embedding import R2D2Embedding
from models.protonet_embedding_5l import ProtoNetEmbedding
from models.resnet12_2 import resnet12
from models.classifier import LinearClassifier, NNClassifier, distLinear
from models.PredTrainHead import LinearRotateHead, DCLHead, DistRotateHead, EucDistRotateHead, ArcMarginProduct, AddMarginProduct, SphereProduct
from models.resnet_ import resnet18, resnet34, resnet50
from models.wrn28 import Wide_ResNet

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


    n_classes=10
    if options.pre_head == 'LinearRotateNet':
        pre_head = LinearRotateHead(in_dim=fea_dim, n_classes=n_classes).cuda()
    else:
        print("Cannot recognize the dataset type")
        assert (False)

    # Choose the classification head
    if options.head == 'CosineNet':
        cls_head = ClassificationHead(base_learner='CosineNet').cuda()
    else:
        print ("Cannot recognize the dataset type")
        assert(False)
    # info_max_layer = InfoMaxLayer().cuda()
    return (network, pre_head, cls_head)

def get_dataset(options):
    # Choose the embedding network
    if options.dataset == 'UCML':
        from data.ucml import UCML, FewShotDataloader
        dataset_train = UCML(phase='train')
        dataset_val = UCML(phase='val')
        dataset_test = UCML(phase='test')
        data_loader = FewShotDataloader
    else:
        print ("Cannot recognize the dataset type")
        assert(False)
        
    return (dataset_train, dataset_val, dataset_test, data_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-epoch', type=int, default=100,
                            help='number of training epochs')
    parser.add_argument('--save-epoch', type=int, default=20,
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
                            help='choose which embedding network to use. ProtoNet, R2D2, ResNet, WRN28')
    parser.add_argument('--head', type=str, default='CosineNet',
                            help='choose which classification head to use. ProtoNet, Ridge, R2D2, SVM')
    parser.add_argument('--pre_head', type=str, default='LinearRotateNet',
                            help='choose which classification head to use. ProtoNet, Ridge, R2D2, SVM')
    parser.add_argument('--dataset', type=str, default='UCML',
                            help='choose which classification head to use. miniImageNet, tieredImageNet, CIFAR_FS, FC100')
    parser.add_argument('--episodes-per-batch', type=int, default=8,
                            help='number of episodes per batch')
    parser.add_argument('--eps', type=float, default=0.0,
                            help='epsilon of label smoothing')

    opt = parser.parse_args()
    
    (dataset_train, dataset_val, dataset_test, data_loader) = get_dataset(opt)

    data_loader_pre = torch.utils.data.DataLoader
    # Dataloader of Gidaris & Komodakis (CVPR 2018)

    train_way = 10
    dloader_train = data_loader_pre(
        dataset=dataset_train,
        batch_size=16,
        shuffle=True,
        num_workers=4
    )
    dloader_val = data_loader(
        dataset=dataset_val,
        nKnovel=opt.test_way,
        nKbase=0,
        nExemplars=opt.val_shot,  # num training examples per novel category
        nTestNovel=opt.val_query * opt.test_way,  # num test examples for all the novel categories
        nTestBase=0,  # num test examples for all the base categories
        batch_size=1,
        num_workers=0,
        epoch_size=1 * opt.val_episode,  # num of batches per epoch
    )

    dloader_test = data_loader(
        dataset=dataset_test,
        nKnovel=opt.test_way,
        nKbase=0,
        nExemplars=opt.val_shot,  # num training examples per novel category
        nTestNovel=opt.val_query * opt.test_way,  # num test examples for all the novel categories
        nTestBase=0,  # num test examples for all the base categories
        batch_size=1,
        num_workers=0,
        epoch_size=1 * opt.val_episode,  # num of batches per epoch
    )

    set_gpu(opt.gpu)
    check_dir('./experiments/')
    check_dir(opt.save_path)

    log_file_path = os.path.join(opt.save_path, "train_log.txt")
    log(log_file_path, str(vars(opt)))

    (embedding_net, pre_head, cls_head) = get_model(opt)

    optimizer = torch.optim.SGD([{'params': embedding_net.parameters()},
                                 {'params': pre_head.parameters()}], lr=0.1, momentum=0.9, \
                                weight_decay=5e-4, nesterov=True)

    # lambda_epoch = lambda e: 1.0 if e < 60 else (0.1 if e < 80 else 0.01 if e < 90 else (0.001))
    lambda_epoch = lambda e: 1.0 if e < 10 else (0.1 if e < 15 else 0.01 if e < 20 else (0.001))
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

        _, _, _ = [x.train() for x in (embedding_net, pre_head, cls_head)]

        train_accuracies = []
        train_losses = []

        for i, batch in enumerate(tqdm(dloader_train), 1):
            data, labels = [x.cuda() for x in batch]

            if opt.pre_head == 'LinearNet' or opt.pre_head == 'CosineNet':
                emb = embedding_net(data)
                logit = pre_head(emb)
                smoothed_one_hot = one_hot(labels.reshape(-1), train_way)
                smoothed_one_hot = smoothed_one_hot * (1 - opt.eps) + (1 - smoothed_one_hot) * opt.eps / (train_way - 1)

                log_prb = F.log_softmax(logit.reshape(-1, train_way), dim=1)
                loss = -(smoothed_one_hot * log_prb).sum(dim=1)
                loss = loss.mean()
                acc = count_accuracy(logit.reshape(-1, train_way), labels.reshape(-1))
            elif opt.pre_head == 'LinearRotateNet' or opt.pre_head == 'DistRotateNet' or opt.pre_head == 'eucdistRotateNet' or opt.pre_head == 'distRotateNet':
                x_ = []
                y_ = []
                a_ = []
                for j in range(data.shape[0]):
                    x90 = data[j].transpose(2, 1).flip(1)
                    x180 = x90.transpose(2, 1).flip(1)
                    x270 = x180.transpose(2, 1).flip(1)
                    x_ += [data[j], x90, x180, x270]
                    y_ += [labels[j] for _ in range(4)]
                    a_ += [torch.tensor(0), torch.tensor(1), torch.tensor(2), torch.tensor(3)]

                x_ = Variable(torch.stack(x_, 0)).cuda()
                y_ = Variable(torch.stack(y_, 0)).cuda()
                a_ = Variable(torch.stack(a_, 0)).cuda()
                emb = embedding_net(x_)
                # print(emb.shape)
                logit = pre_head(emb, use_cls=True)
                logit_rotate = pre_head(emb, use_cls=False)
                smoothed_one_hot = one_hot(y_.reshape(-1), train_way)
                smoothed_one_hot = smoothed_one_hot * (1 - opt.eps) + (1 - smoothed_one_hot) * opt.eps / (train_way - 1)

                log_prb = F.log_softmax(logit.reshape(-1, train_way), dim=1)
                loss = -(smoothed_one_hot * log_prb).sum(dim=1)
                loss = loss.mean()
                rloss = F.cross_entropy(input=logit_rotate, target=a_)
                loss = 0.5 * loss + 0.5 * rloss
                acc = count_accuracy(logit.reshape(-1, train_way), y_.reshape(-1))
            else:
                emb = embedding_net(data)
                logit = pre_head(emb, labels)

                smoothed_one_hot = one_hot(labels.reshape(-1), train_way)
                smoothed_one_hot = smoothed_one_hot * (1 - opt.eps) + (1 - smoothed_one_hot) * opt.eps / (train_way - 1)

                log_prb = F.log_softmax(logit.reshape(-1, train_way), dim=1)
                loss = -(smoothed_one_hot * log_prb).sum(dim=1)
                loss = loss.mean()
                acc = count_accuracy(logit.reshape(-1, train_way), labels.reshape(-1))
            train_accuracies.append(acc.item())
            train_losses.append(loss.item())

            if (i % 5 == 0):
                train_acc_avg = np.mean(np.array(train_accuracies))
                log(log_file_path, 'Train Epoch: {}\tBatch: [{}]\tLoss: {}\tAccuracy: {} % ({} %)'.format(
                    epoch, i, loss.item(), train_acc_avg, acc))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate on the validation split
        _, _, _ = [x.eval() for x in (embedding_net, pre_head, cls_head)]

        val_accuracies = []
        val_losses = []

        for i, batch in enumerate(tqdm(dloader_val(epoch)), 1):
            data_support, labels_support, \
            data_query, labels_query, _, _ = [
                x.cuda() for x in batch]

            test_n_support = opt.test_way * opt.val_shot
            test_n_query = opt.test_way * opt.val_query

            with torch.no_grad():
                emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
                emb_support = emb_support.reshape(1, test_n_support, -1)

                emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
                emb_query = emb_query.reshape(1, test_n_query, -1)

            logit_query = cls_head(emb_query, emb_support, labels_support, opt.test_way, opt.val_shot)

            loss = x_entropy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))
            acc = count_accuracy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))

            val_accuracies.append(acc.item())
            val_losses.append(loss.item())

        val_acc_avg = np.mean(np.array(val_accuracies))
        val_acc_ci95 = 1.96 * np.std(np.array(val_accuracies)) / np.sqrt(opt.val_episode)

        val_loss_avg = np.mean(np.array(val_losses))

        if val_acc_avg > max_val_acc:
            max_val_acc = val_acc_avg
            torch.save({'embedding': embedding_net.state_dict(),
                        'pre_head': pre_head.state_dict(), 'cls_head': cls_head.state_dict()}, \
                       os.path.join(opt.save_path, 'ucml_best_pretrain_model_{}_{}s.pth'.format(opt.network, opt.val_shot)))
            log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} % (Best)' \
                .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95))
        else:
            log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} %' \
                .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95))

        torch.save({'embedding': embedding_net.state_dict(),
                    'pre_head': pre_head.state_dict(), 'cls_head': cls_head.state_dict()} \
                   , os.path.join(opt.save_path, 'ucml_last_pretrain_epoch_{}_{}s.pth'.format(opt.network, opt.val_shot)))

        if epoch % opt.save_epoch == 0:
            torch.save({'embedding': embedding_net.state_dict(),
                        'pre_head': pre_head.state_dict(), 'cls_head': cls_head.state_dict()} \
                       , os.path.join(opt.save_path, 'ucml_epoch_{}_pretrain_{}_{}s.pth'.format(epoch, opt.network, opt.val_shot)))
