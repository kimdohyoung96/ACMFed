from validation import epochVal_metrics_test
from options import args_parser
import os
import sys
import logging
import random
import numpy as np
import copy
import datetime
from FedAvg import FedAvg, model_dist
import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn
from networks.models import ModelFedCon
from dataloaders import dataset
from local_supervised import SupervisedLocalUpdate
from local_unsupervised import UnsupervisedLocalUpdate
from tqdm import trange
from cifar_load import get_dataloader, partition_data, partition_data_allnoniid
from torch.utils.tensorboard import SummaryWriter


def split(dataset, num_users):
    # randomly split dataset into equally num_users parties
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def select_samlple(x,y,n_classes):
    class_n = torch.zeros(n_classes)
    for i in range(n_classes):
        class_n[i] = (y==i).float().sum()
    X_new = []
    Y_new = []
    select_n = torch.ones(n_classes)*min(class_n)
    select_list = random.shuffle([i for i in range(len(y))])
    print(select_list)
    #print(y.shape, select_list[i], y[select_list[i]])
    for i in range(len(x)):
        if select_n[y[select_list[i]]] > 0:
            X_new.append(x[select_list[i]])
            Y_new.append(y[select_list[i]])
            select_n[y[select_list[i]]] = select_n[y[select_list[i]]] - 1

def test(epoch, checkpoint, data_test, label_test, n_classes):

    #231220
    #CBAFed method

    #if args.model == 'Res18':
    #    net = torch_models.resnet18(pretrained=args.Pretrained)
    #    net.fc = nn.Linear(net.fc.weight.shape[1], n_classes)

    net = ModelFedCon(args.model, args.out_dim, n_classes=n_classes)
    #if len(args.gpu.split(',')) > 1:
    #    net = torch.nn.DataParallel(net, device_ids=[i for i in range(round(len(args.gpu) / 2))])
    model = net.cuda()
    model.load_state_dict(checkpoint)

    if args.dataset == 'SVHN' or args.dataset == 'cifar100' or args.dataset == 'pneumonia' or args.dataset == 'cifar10':
        test_dl, test_ds = get_dataloader(args, data_test, label_test,
                                          args.dataset, args.datadir, args.batch_size,
                                          is_labeled=True, is_testing=True)
    elif args.dataset == 'skin' or args.dataset == 'STL10' or args.dataset == 'fmnist':
        test_dl, test_ds = get_dataloader(args, data_test, label_test,
                                          args.dataset, args.datadir, args.batch_size,
                                          is_labeled=True, is_testing=True, pre_sz=args.pre_sz, input_sz=args.input_sz)

    # 오류 1. 같은 cpu상에 있어야 함
    #thresh = 임계값
    #AUROCs, Accus = epochVal_metrics_test(model, test_dl, args.model, thresh=0.4, n_classes=n_classes)

    # 오류 2. 압축을 풀기에 값이 너무 많음
    #AUROCs, Accus = epochVal_metrics_test(model, test_dl, args.model, n_classes=n_classes)
    
    # 오류 3. TypeError: epochVal_metrics_test() got an unexpected keyword argument 'thresh'
    #AUROCs, Accus = epochVal_metrics_test(model, test_dl, args.model, thresh=0.1, n_classes=n_classes)
    
    # 오류 4. TypeError: epochVal_metrics_test() got an unexpected keyword argument 'thresh'
    # local_unsupervised.py 수정 후 실행
    # AUROCs, Accus = epochVal_metrics_test(model, test_dl, args.model, thresh=0.4, n_classes=n_classes)
    
    # 시도 5. TypeError: epochVal_metrics_test() got an unexpected keyword argument 'threshold'
    # thresh -> threshold로 수정 후 실행
    #
    #AUROCs, Accus = epochVal_metrics_test(model, test_dl, args.model, threshold=0.4, n_classes=n_classes)

    # 시도 6. TypeError: epochVal_metrics_test() got an unexpected keyword argument 'threshold'
    #AUROCs, Accus = epochVal_metrics_test(model, test_dl, args.model, threshold=0.4, n_classes=n_classes)

    # 시도 7. TypeError: epochVal_metrics_test() got an unexpected keyword argument 'thresh'
    # validation 수정, thresh
    #AUROCs, Accus = epochVal_metrics_test(model, test_dl, args.model, thresh=0.4, n_classes=n_classes)

    # 시도 8. TypeError: epochVal_metrics_test() got an unexpected keyword argument 'thresh'
    # validation 수정, thresh, pip install -U imbalanced-learn
    #AUROCs, Accus = epochVal_metrics_test(model, test_dl, args.model, thresh=0.4, n_classes=n_classes)
    
    # 시도 9. TypeError: epochVal_metrics_test() got an unexpected keyword argument 'thresh'
    # validation 수정, thresh, pip install -U imbalanced-learn, conda install -c conda-forge imbalanced-learn
    #AUROCs, Accus = epochVal_metrics_test(model, test_dl, args.model, thresh=0.4, n_classes=n_classes)

    # 시도 10. ValueError: too many values to unpack (expected 2)
    # validation 수정, thresh, pip install -U imbalanced-learn, conda install -c conda-forge imbalanced-learn,
    #AUROCs, Accus = epochVal_metrics_test(model, test_dl, args.model, n_classes=n_classes)

    # 시도 11. NameError: name 'thresh' is not defined
    # validation 수정, thresh, pip install -U imbalanced-learn, conda install -c conda-forge imbalanced-learn, thresh=thresh
    #AUROCs, Accus = epochVal_metrics_test(model, test_dl, args.model, thresh=thresh, n_classes=n_classes)

    # 시도 12. TypeError: epochVal_metrics_test() got an unexpected keyword argument 'thresh'
    # validation 수정, thresh, pip install -U imbalanced-learn, conda install -c conda-forge imbalanced-learn, thresh 변수 추가
    #thresh = 0.4
    #AUROCs, Accus = epochVal_metrics_test(model, test_dl, args.model, thresh=thresh, n_classes=n_classes)

    # 시도 13. 
    # validation 수정, thresh, pip install -U imbalanced-learn, conda install -c conda-forge imbalanced-learn, thresh 변수 추가
    #thresh = 0.4
    #AUROCs, Accus = epochVal_metrics_test(model, test_dl, args.model, thresh, n_classes=n_classes)

    # 시도 13. TypeError: epochVal_metrics_test() takes 4 positional arguments but 5 were given
    # validation 수정, thresh, pip install -U imbalanced-learn, conda install -c conda-forge imbalanced-learn, thresh 변수 추가
    #thresh = 0.4
    #AUROCs, Accus = epochVal_metrics_test(model, test_dl, args.model, thresh, n_classes)
    
    # 시도 14.ValueError: too many values to unpack (expected 2)
    # validation 수정, thresh, pip install -U imbalanced-learn, conda install -c conda-forge imbalanced-learn
    #AUROCs, Accus = epochVal_metrics_test(model, test_dl, args.model, n_classes)

    # 시도 15.ValueError: too many values to unpack (expected 2)
    # validation 수정, thresh, pip install -U imbalanced-learn, conda install -c conda-forge imbalanced-learn
    #result, _ = epochVal_metrics_test(model, test_dl, args.model, n_classes)
    #AUROCs, Accus = result
    
    # 시도 16.ValueError: too many values to unpack (expected 2)
    # validation 수정, thresh, pip install -U imbalanced-learn, conda install -c conda-forge imbalanced-learn
    #result, _ = epochVal_metrics_test(model, test_dl, args.model, n_classes=n_classes)
    #AUROCs, Accus = result

    # 시도 17. 
    # validation 수정, thresh, pip install -U imbalanced-learn, conda install -c conda-forge imbalanced-learn
    # cmd에 thresh=0.4 추가
    #AUROCs, Accus = epochVal_metrics_test(model, test_dl, args.model, n_classes=n_classes)

    # 시도 18.TypeError: epochVal_metrics_test() got an unexpected keyword argument 'thresh'
    # validation 수정, thresh, pip install -U imbalanced-learn, conda install -c conda-forge imbalanced-learn, thresh=thresh
    #AUROCs, Accus = epochVal_metrics_test(model, test_dl, args.model, thresh=0.4, n_classes=n_classes)

    # 시도 19.
    # validation 수정, thresh, pip install -U imbalanced-learn, conda install -c conda-forge imbalanced-learn, thresh=thresh
    #AUROCs, Accus = epochVal_metrics_test(model, test_dl, args.model, thresh=0.4, n_classes=n_classes)

    # 시도 20. TypeError: epochVal_metrics_test() got an unexpected keyword argument 'thresh'
    # validation 수정, thresh, pip install -U imbalanced-learn, conda install -c conda-forge imbalanced-learn, thresh=thresh
    # epochVal_metrics_test(model, dataLoader, model_type, n_classes)
    #AUROCs, Accus = epochVal_metrics_test(model, test_dl, args.model, thresh=0.4, n_classes=n_classes)
    
    #231120
    # 시도 21.
    # validation 수정, thresh, pip install -U imbalanced-learn, conda install -c conda-forge imbalanced-learn, thresh=thresh
    # epochVal_metrics_test(model, dataLoader, model_type, n_classes)
    AUROCs, Accus, Pre, Recall = epochVal_metrics_test(model, test_dl, args.model, n_classes=n_classes)

    #231120
    # 시도 22.
    #아직 안해봄
    #AUROCs, Accus, Pre, Recall = epochVal_metrics_test(model, test_dl, args.model, 0.4, n_classes=n_classes)

    AUROC_avg = np.array(AUROCs).mean()
    Accus_avg = np.array(Accus).mean()

    return AUROC_avg, Accus_avg

if __name__ == '__main__':
    args = args_parser()

    supervised_user_id = [0]
    unsupervised_user_id = list(range(len(supervised_user_id), args.unsup_num + len(supervised_user_id)))
    sup_num = len(supervised_user_id)
    unsup_num = len(unsupervised_user_id)
    total_num = sup_num + unsup_num

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    time_current = 'attempt0'
    if args.log_file_name is None:
        args.log_file_name = 'log-%s' % (datetime.datetime.now().strftime("%m-%d-%H%M-%S"))
    log_path = args.log_file_name + '.log'
    logging.basicConfig(filename=os.path.join(args.logdir, log_path), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.info(str(args))
    logger.info(time_current)
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    if not os.path.isdir('tensorboard'):
        os.mkdir('tensorboard')

    if args.dataset == 'SVHN':
        if not os.path.isdir('tensorboard/SVHN/' + time_current):
            os.mkdir('tensorboard/cares_SVHN/' + time_current)
        writer = SummaryWriter('tensorboard/SVHN/' + time_current)

    elif args.dataset == 'cifar100':
        if not os.path.isdir('tensorboard/cifar100/' + time_current):
            os.mkdir('tensorboard/cifar100/' + time_current)
        writer = SummaryWriter('tensorboard/cifar100/' + time_current)
    #231211
    elif args.dataset == 'pneumonia':
        if not os.path.isdir('tensorboard/pneumonia/' + time_current):
            os.mkdir('tensorboard/pneumonia/' + time_current)
        writer = SummaryWriter('tensorboard/pneumonia/' + time_current)
        #231128
    elif args.dataset == 'skin':
        if not os.path.isdir('tensorboard/skin/' + time_current):
            os.mkdir('tensorboard/skin/' + time_current)
        writer = SummaryWriter('tensorboard/skin/' + time_current)

    snapshot_path = 'model/'
    if not os.path.isdir(snapshot_path):
        os.mkdir(snapshot_path)
    if args.dataset == 'SVHN':
        snapshot_path = 'model/SVHN/'
    if args.dataset == 'cifar100':
        snapshot_path = 'model/cifar100/'
    #231211
    if args.dataset == 'pneumonia':
        snapshot_path = 'model/pneumonia/'
    if args.dataset == 'skin':
        snapshot_path = 'model/skin/'
    if not os.path.isdir(snapshot_path):
        os.mkdir(snapshot_path)

    print('==> Reloading data partitioning strategy..')
    assert os.path.isdir('partition_strategy'), 'Error: no partition_strategy directory found!'

    #dataset확인
    if args.dataset == 'SVHN':
        #파일로드
        #partition변수에 할당
        partition = torch.load('partition_strategy/SVHN_noniid_10%labeled.pth')
        #데이터 파티셔닝 정보 추출
        #partition 딕셔너리에서 data_partition 키에 해당하는 값을 net_dataidx_map 변수에 저장
        #이는 데이터셋의 특정 부분을 나타내고, 논리적으로 데이터를 여러 네트워크 노드에 분산하기 위해 인덱싱 정보를 포함
        net_dataidx_map = partition['data_partition']
    elif args.dataset == 'cifar100':
        partition = torch.load('partition_strategy/cifar100_noniid_10%labeled.pth')
        net_dataidx_map = partition['data_partition']
    #231211
    elif args.dataset == 'pneumonia':
        partition = torch.load('partition_strategy/pneumonia_noniid_10%labeled(1).pth')
        # 231219
        # data_partition is not find
        net_dataidx_map = partition['data_partition']
        #이렇게 하면 skin 데이터를 염
    elif args.dataset == 'skin':
        partition = torch.load('partition_strategy/skin_noniid_10%labeled.pth')
        net_dataidx_map = partition['data_partition']

    X_train, y_train, X_test, y_test, _, traindata_cls_counts = partition_data_allnoniid(
        args.dataset, args.datadir, partition=args.partition, n_parties=total_num, beta=args.beta)

    if args.dataset == 'SVHN':
        X_train = X_train.transpose([0, 2, 3, 1])
        X_test = X_test.transpose([0, 2, 3, 1])

    if args.dataset == 'cifar10' or args.dataset == 'SVHN' or args.dataset == 'pneumonia':
        n_classes = 10
    elif args.dataset == 'cifar100':
        n_classes = 100
    elif args.dataset == 'skin':
        n_classes = 7
    net_glob = ModelFedCon(args.model, args.out_dim, n_classes=n_classes)

    if args.resume:
        print('==> Resuming from checkpoint..')
        if args.dataset == 'cifar100':
            checkpoint = torch.load('warmup/cifar100.pth')
        elif args.dataset == 'SVHN':
            checkpoint = torch.load('warmup/SVHN.pth')
        #231211
        elif args.dataset == 'pneumonia':
            checkpoint = torch.load('warmup/pneumonia.pth')
        #231207
        elif args.dataset == 'skin':
            checkpoint = torch.load('warmup/skin.pth')
        elif args.dataset == 'cifar10':
            checkpoint = torch.load('warmup/weights_resnet18_cifar10.pth')

        net_glob.load_state_dict(checkpoint['state_dict'])
        start_epoch = 7
    else:
        start_epoch = 0

    #if len(args.gpu.split(',')) > 1:
    #    net_glob = torch.nn.DataParallel(net_glob, device_ids=[i for i in range(round(len(args.gpu) / 2))])  #
    net_glob.train()
    w_glob = net_glob.state_dict()
    w_locals = []
    w_ema_unsup = []
    lab_trainer_locals = []
    unlab_trainer_locals = []
    sup_net_locals = []
    unsup_net_locals = []
    sup_optim_locals = []
    unsup_optim_locals = []

    dist_scale_f = args.dist_scale

    total_lenth = sum([len(net_dataidx_map[i]) for i in range(len(net_dataidx_map))])
    each_lenth = [len(net_dataidx_map[i]) for i in range(len(net_dataidx_map))]
    client_freq = [len(net_dataidx_map[i]) / total_lenth for i in range(len(net_dataidx_map))]

    for i in supervised_user_id:
        lab_trainer_locals.append(SupervisedLocalUpdate(args, net_dataidx_map[i], n_classes))
        w_locals.append(copy.deepcopy(w_glob))
        sup_net_locals.append(copy.deepcopy(net_glob))
        if args.opt == 'adam':
            optimizer = torch.optim.Adam(sup_net_locals[i].parameters(), lr=args.base_lr,
                                         betas=(0.9, 0.999), weight_decay=5e-4)
        elif args.opt == 'sgd':
            optimizer = torch.optim.SGD(sup_net_locals[i].parameters(), lr=args.base_lr, momentum=0.9,
                                        weight_decay=5e-4)
        elif args.opt == 'adamw':
            optimizer = torch.optim.AdamW(sup_net_locals[i].parameters(), lr=args.base_lr, weight_decay=0.02)
        if args.resume:
            optimizer.load_state_dict(checkpoint['sup_optimizers'][i])
        sup_optim_locals.append(copy.deepcopy(optimizer.state_dict()))

    for i in unsupervised_user_id:
        unlab_trainer_locals.append(
            UnsupervisedLocalUpdate(args, net_dataidx_map[i], n_classes))
        w_locals.append(copy.deepcopy(w_glob))
        w_ema_unsup.append(copy.deepcopy(w_glob))
        unsup_net_locals.append(copy.deepcopy(net_glob))
        if args.opt == 'adam':
            optimizer = torch.optim.Adam(unsup_net_locals[i - sup_num].parameters(), lr=args.unsup_lr,
                                         betas=(0.9, 0.999), weight_decay=5e-4)
        elif args.opt == 'sgd':
            optimizer = torch.optim.SGD(unsup_net_locals[i - sup_num].parameters(),
                                        lr=args.unsup_lr, momentum=0.9,
                                        weight_decay=5e-4)
        elif args.opt == 'adamw':
            optimizer = torch.optim.AdamW(unsup_net_locals[i - sup_num].parameters(), lr=args.unsup_lr,
                                          weight_decay=0.02)
        if args.resume and len(checkpoint['unsup_optimizers']) != 0:
            optimizer.load_state_dict(checkpoint['unsup_optimizers'][i - sup_num])
        unsup_optim_locals.append(copy.deepcopy(optimizer.state_dict()))

        if args.resume and len(checkpoint['unsup_ema_state_dict']) != 0 and not args.from_labeled:
            w_ema_unsup = copy.deepcopy(checkpoint['unsup_ema_state_dict'])
            unlab_trainer_locals[i - sup_num].ema_model.load_state_dict(w_ema_unsup[i - sup_num])
            unlab_trainer_locals[i - sup_num].flag = False
            print('Unsup EMA reloaded')

    for com_round in trange(start_epoch, args.rounds):
        print("************* Comm round %d begins *************" % com_round)
        loss_locals = []
        clt_this_comm_round = []
        w_per_meta = []

        for meta_round in range(args.meta_round):
            clt_list_this_meta_round = random.sample(list(range(0, total_num)), args.meta_client_num)
            clt_this_comm_round.extend(clt_list_this_meta_round)
            chosen_sup = [j for j in supervised_user_id if j in clt_list_this_meta_round]
            logger.info(f'Comm round {com_round} meta round {meta_round} chosen client: {clt_list_this_meta_round}')
            w_locals_this_meta_round = []
            for client_idx in clt_list_this_meta_round:
                if client_idx in supervised_user_id:
                    local = lab_trainer_locals[client_idx]
                    optimizer = sup_optim_locals[client_idx]
                    train_dl_local, train_ds_local = get_dataloader(args, X_train[net_dataidx_map[client_idx]],
                                                                    y_train[net_dataidx_map[client_idx]],
                                                                    args.dataset, args.datadir, args.batch_size,
                                                                    is_labeled=True,
                                                                    data_idxs=net_dataidx_map[client_idx],
                                                                    pre_sz=args.pre_sz, input_sz=args.input_sz)
                    w, loss, op = local.train(args, sup_net_locals[client_idx].state_dict(), optimizer,
                                                           train_dl_local, n_classes)  # network, loss, optimizer
                    writer.add_scalar('Supervised loss on sup client %d' % client_idx, loss, global_step=com_round)

                    w_locals_this_meta_round.append(copy.deepcopy(w))
                    sup_optim_locals[client_idx] = copy.deepcopy(op)
                    loss_locals.append(copy.deepcopy(loss))
                    logger.info(
                        'Labeled client {} sample num: {} training loss : {} lr : {}'.format(client_idx,
                                                                                             len(train_ds_local),
                                                                                             loss,
                                                                                             sup_optim_locals[
                                                                                                 client_idx][
                                                                                                 'param_groups'][0][
                                                                                                 'lr']))
                else:
                    local = unlab_trainer_locals[client_idx - sup_num]
                    optimizer = unsup_optim_locals[client_idx - sup_num]
                    train_dl_local, train_ds_local = get_dataloader(args,
                                                                    X_train[net_dataidx_map[client_idx]],
                                                                    y_train[net_dataidx_map[client_idx]],
                                                                    args.dataset,
                                                                    args.datadir, args.batch_size, is_labeled=False,
                                                                    data_idxs=net_dataidx_map[client_idx],
                                                                    pre_sz=args.pre_sz, input_sz=args.input_sz)
                    w, w_ema, loss, op, ratio, correct_pseu, all_pseu, test_right, train_right, test_right_ema, same_pred_num = local.train(
                        args,
                        unsup_net_locals[client_idx - sup_num].state_dict(),
                        optimizer,
                        com_round * args.local_ep,
                        client_idx,
                        train_dl_local, n_classes)
                    writer.add_scalar('Unsupervised loss on unsup client %d' % client_idx, loss, global_step=com_round)

                    w_locals_this_meta_round.append(copy.deepcopy(w))
                    w_ema_unsup[client_idx - sup_num] = copy.deepcopy(w_ema)
                    unsup_optim_locals[client_idx - sup_num] = copy.deepcopy(op)
                    loss_locals.append(copy.deepcopy(loss))
                    logger.info(
                        'Unlabeled client {} sample num: {} Training loss: {}, unsupervised loss ratio: {}, lr {}, {} pseu out of {} are correct, {} correct by model, {} correct by ema before train, {} by model during train, total {}'.format(
                            client_idx, len(train_ds_local), loss,
                            ratio,
                            unsup_optim_locals[
                                client_idx - sup_num][
                                'param_groups'][
                                0]['lr'], correct_pseu, all_pseu, test_right, test_right_ema, train_right,
                            len(net_dataidx_map[client_idx])))

            each_lenth_this_meta_round = [each_lenth[clt] for clt in clt_list_this_meta_round]
            each_lenth_this_meta_raw = copy.deepcopy(each_lenth_this_meta_round)
            # total_lenth_this_meta = sum(each_lenth_this_meta_round)

            if args.w_mul_times != 1 and 0 in clt_list_this_meta_round and (
                    args.un_dist == '' or args.un_dist_onlyunsup):  # and com_round<=40:  # :
                for sup_idx in chosen_sup:
                    each_lenth_this_meta_round[clt_list_this_meta_round.index(sup_idx)] *= args.w_mul_times

            total_lenth_this_meta = sum(each_lenth_this_meta_round)
            clt_freq_this_meta_round = [i / total_lenth_this_meta for i in each_lenth_this_meta_round]
            print('Based on data amount: ' + f'{clt_freq_this_meta_round}')
            clt_freq_this_meta_raw = copy.deepcopy(clt_freq_this_meta_round)

            w_avg_temp = FedAvg(w_locals_this_meta_round, clt_freq_this_meta_round)
            dist_list = []
            for cli_idx in range(args.meta_client_num):
                dist = model_dist(w_locals_this_meta_round[cli_idx], w_avg_temp)
                dist_list.append(dist)
            print(
                'Normed dist * 1e4 : ' + f'{[dist_list[i] * 1e5 / each_lenth_this_meta_raw[i] for i in range(args.meta_client_num)]}')

            if len(chosen_sup) != 0:
                clt_freq_this_meta_uncer = [
                    np.exp(-dist_list[i] * args.sup_scale / each_lenth_this_meta_raw[i]) * clt_freq_this_meta_round[i] for i
                    in
                    range(args.meta_client_num)]
                for sup_idx in chosen_sup:
                    mul_times = args.w_mul_times
                    clt_freq_this_meta_uncer[clt_list_this_meta_round.index(
                            sup_idx)] *= mul_times  # (args.w_mul_times/len(chosen_sup))
            else:
                clt_freq_this_meta_uncer = [
                    np.exp(-dist_list[i] * dist_scale_f / each_lenth_this_meta_raw[i]) * clt_freq_this_meta_round[i]
                    for i
                    in range(args.meta_client_num)]

            total = sum(clt_freq_this_meta_uncer)
            clt_freq_this_meta_dist = [clt_freq_this_meta_uncer[i] / total for i in range(args.meta_client_num)]
            clt_freq_this_meta_round = clt_freq_this_meta_dist
            print('After dist-based uncertainty : ' + f'{clt_freq_this_meta_round}')

            assert sum(clt_freq_this_meta_round) - 1.0 <= 1e-3, "Error: sum(freq) != 0"
            w_this_meta = FedAvg(w_locals_this_meta_round, clt_freq_this_meta_round)

            w_per_meta.append(w_this_meta)

        each_lenth_this_round = [each_lenth[clt] for clt in clt_this_comm_round]

        if args.w_mul_times != 1 and 0 in clt_this_comm_round:
            each_lenth_this_round[clt_this_comm_round.index(0)] *= args.w_mul_times
        total_lenth_this = sum(each_lenth_this_round)
        clt_freq_this_round = [i / total_lenth_this for i in each_lenth_this_round]

        with torch.no_grad():
            freq = [1 / args.meta_round for i in range(args.meta_round)]
            w_glob = FedAvg(w_per_meta, freq)

        net_glob.load_state_dict(w_glob)
        for i in supervised_user_id:
            sup_net_locals[i].load_state_dict(w_glob)
        for i in unsupervised_user_id:
            unsup_net_locals[i - sup_num].load_state_dict(w_glob)

        loss_avg = sum(loss_locals) / len(loss_locals)
        logger.info(
            '************ Loss Avg {}, LR {}, Round {} ends ************  '.format(loss_avg, args.base_lr, com_round))
        if com_round % 6 == 0:
            if not os.path.isdir(snapshot_path + time_current):
                os.mkdir(snapshot_path + time_current)
            save_mode_path = os.path.join(snapshot_path + time_current, 'epoch_' + str(com_round) + '.pth')
            if len(args.gpu) != 1:
                torch.save({
                    'state_dict': net_glob.module.state_dict(),
                    'unsup_ema_state_dict': w_ema_unsup,
                    'sup_optimizers': sup_optim_locals,
                    'unsup_optimizers': unsup_optim_locals,
                    'start_epoch': com_round
                }
                    , save_mode_path
                )
            else:
                torch.save({
                    'state_dict': net_glob.state_dict(),
                    'unsup_ema_state_dict': w_ema_unsup,
                    'sup_optimizers': sup_optim_locals,
                    'unsup_optimizers': unsup_optim_locals,
                    'start_epoch': com_round
                }
                    , save_mode_path
                )
        AUROC_avg, Accus_avg = test(com_round, net_glob.state_dict(), X_test, y_test, n_classes)
        writer.add_scalar('AUC', AUROC_avg, global_step=com_round)
        writer.add_scalar('Acc', Accus_avg, global_step=com_round)
        logger.info("\nTEST Student: Epoch: {}".format(com_round))
        logger.info("\nTEST AUROC: {:6f}, TEST Accus: {:6f}"
                    .format(AUROC_avg, Accus_avg))

#231212

#--dataset=pneumonia: 사용할 데이터 세트를 지정합니다.
#--batch_size=32: 배치 크기를 설정합니다. 최적의 배치 크기는 데이터 세트 및 모델 아키텍처에 따라 달라질 수 있습니다.
#--base_lr=0.01and --unsup_lr=0.005: 기본 학습률과 비지도 학습률을 설정합니다.
#--max_grad_norm=5: 최대 기울기 표준을 설정합니다.
#--rounds=50: 훈련 라운드 수입니다.
#--pre_sz=250및 --input_sz=224: 이미지의 전처리 및 입력 크기. 이러한 값은 이미지 데이터세트에 일반적이지만 폐렴 데이터세트의 세부 사항에 따라 조정되어야 합니다.
#--resume --from_labeled: 해당하는 경우 레이블이 지정된 데이터에서 훈련을 계속합니다.


##.pth에 대해서
#데이터 세트 선택 : 잠재적으로 폐렴 데이터 세트를 포함하여 다양한 흉부 X선 데이터 세트를 제공하는 TorchXRayVision 라이브러리를 활용합니다.

#비 IID 파티션 생성 : 비 IID(비독립적이고 동일하게 분산됨) 설정에서 데이터 세트는 데이터 분포가 여러 하위 집합에 따라 달라지는 방식으로 분할됩니다. 
#
#파일 의 경우 pneumonia_noniid_10%labeled.pth데이터의 10%에만 레이블이 지정되도록 폐렴 데이터 세트를 분할합니다.

#모델 훈련 : 이 분할된 데이터 세트에서 모델(예: CNN)을 훈련합니다. 
#
#훈련 중에는 일반적으로 모델과 폐렴 데이터세트의 특성에 적합한 손실 함수, 최적화 도구, 다양한 하이퍼파라미터를 사용하게 됩니다.

#모델 상태 저장 : 학습 후 모델의 상태 사전을 .pth파일에 저장합니다. 
#
#이 파일에는 모델의 학습된 매개변수가 포함되어 있습니다.