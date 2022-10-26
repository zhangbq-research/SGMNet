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
        dataset_test = UCML(phase='test')
        data_loader = FewShotDataloader
    else:
        print("Cannot recognize the dataset type")
        assert (False)

    return (dataset_test, data_loader)

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
    parser.add_argument('--val-shot', type=int, default=1,
                            help='number of support examples per validation class')
    parser.add_argument('--val-episode', type=int, default=600,
                            help='number of episodes per validation')
    parser.add_argument('--val-query', type=int, default=15,
                            help='number of query examples per validation class')
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
    parser.add_argument('--eps', type=float, default=0.0,
                            help='epsilon of label smoothing')

    opt = parser.parse_args()
    seed_torch(0)

    (dataset_test, data_loader) = get_dataset(opt)

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
    torch.backends.cudnn.enabled = False

    (embedding_net, cls_head) = get_model(opt)
    saved_models = torch.load(os.path.join(opt.save_path, 'ucml_best_meta_training_model_{}_{}s.pth'.format(opt.network, opt.val_shot)))
    embedding_net.load_state_dict(saved_models['embedding'])
    embedding_net.eval()
    cls_head.load_state_dict(saved_models['cls_head'])
    cls_head.eval()

    max_test_acc = 0.0

    timer = Timer()
    x_entropy = torch.nn.CrossEntropyLoss()

    # Evaluate on the validation split
    _, _ = [x.eval() for x in (embedding_net, cls_head)]

    test_accuracies = []
    test_losses = []

    for i, batch in enumerate(tqdm(dloader_test()), 1):
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

        acc = count_accuracy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))

        test_accuracies.append(acc.item())

    test_acc_avg = np.mean(np.array(test_accuracies))
    test_acc_ci95 = 1.96 * np.std(np.array(test_accuracies)) / np.sqrt(opt.val_episode)

    print('Test Accuracy: {:.2f} ± {:.2f} %'.format(test_acc_avg, test_acc_ci95))
