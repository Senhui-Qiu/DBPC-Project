# Copyright 2022 by Senhui Qiu, Ulster University.
# All rights reserved.

import os
import pprint
import torch
from pypc import datasets

from pypc import utils
from pypc.models_pc import PC_Model
from pypc import optim
from pypc.similarity_algorithm import similarity_tensor, save_dif_pre_image_order




DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Initialize_Paremeter():
    cf = utils.AttrDict()
    cf.Epoch_start = 1 # at least 1
    cf.batch_size = 32
    cf.dy_beta = 2
    cf.dy_alpha = 0
    cf.q_lr = None
    cf.init_std = 0.01
    cf.shuffle = True
    cf.train_run = True
    cf.test2_run = True
    cf.AddRotaion = True
    cf.ResizedCrop_size = 28
    cf.each_epoch_change_datasets = True
    cf.plot_reconstruction = True
    cf.AddGaussianNoise = 0
    cf.use_error_c = False
    cf.use_population_coding = False
    cf.expand_size = 1
    cf.shrink_size = 10
    cf.train_dataset = 0  # 0 represent mnist, 1 represent CIFAR10
    cf.weight_no_share_num = 0
    ##########convolution para################
    if cf.train_dataset == 0:
        cf.num_channels = 1
        cf.batch_size = 32
        cf.Train_iteration = 20
        cf.Test_iteration = 20
        cf.Epoch_end = 50
        cf.du_beta = 1
        cf.du_alpha = 0.02
        cf.dy_lr = 0.001
        cf.du_lr = 0.0006
    elif cf.train_dataset == 1:
        cf.num_channels = 3
        cf.Train_iteration = 50
        cf.Test_iteration = 50
        cf.du_beta = 1
        cf.du_alpha = 0.01
        cf.dy_lr = 0.0001
        cf.du_lr = 0.0006

    cf.weight_share = True
    if cf.train_dataset == 0:
        if cf.ResizedCrop_size != 28:
            cf.nodes = [cf.ResizedCrop_size*cf.ResizedCrop_size, 1600, 900, 400, cf.shrink_size * cf.expand_size]
        else:
            cf.nodes = [784, 1000, 400, 100, cf.shrink_size * cf.expand_size]

    else:
        cf.nodes = [32 * 32 * cf.num_channels, 2000, 1000, 512, 256, 128, cf.shrink_size * cf.expand_size]

    cf.act_function = utils.ReLU()
    cf.use_bias = False

    cf.optim_id = "Adam"
    cf.optim_batch_scale = False
    cf.optim_grad_clip = None
    cf.optim_weight_decay = None
    cf.decay_dy = 0

    # dataset params
    cf.train_size = None
    cf.test_size = None
    cf.label_scale = None
    cf.normalize = True

    cf.test_every = 1
    cf.test_Epoch_start = 1
    cf.fixed_Y_Pre_train = True
    cf.fixed_Y_Pre_test = False

    cf.logdir = "data/model/"
    cf.imgdir = cf.logdir + "imgs/"
    cf.result_reconstruction = "result_reconstruction/"
    return cf

def main(cf):
    #####################--create directors and print information###############################
    print("Starting PC reconstruction and classification experiment")
    pprint.pprint(cf)
    os.makedirs(cf.logdir, exist_ok=True)
    os.makedirs(cf.imgdir, exist_ok=True)
    os.makedirs(cf.result_reconstruction, exist_ok=True)
    ######################--download Dataset--##################
    if cf.train_dataset == 0:
        train_dataset = datasets.MNIST(cf, train=True)
        test_dataset = datasets.MNIST(cf, train=False)
    elif cf.train_dataset == 1:
        train_dataset = datasets.CIFAR10(cf, train=True)
        test_dataset = datasets.CIFAR10(cf, train=False)
    train_loader = datasets.get_dataloader(train_dataset, cf.shuffle, cf.batch_size)
    test_loader = datasets.get_dataloader(test_dataset, cf.shuffle, cf.batch_size)
    print("Loaded data [train batches: {}, test batches: {}]".format(len(train_loader), len(test_loader)))

    model = PC_Model(cf)
    optimizer = optim.get_optim(model.params, cf)

    for epoch in range(cf.Epoch_start, cf.Epoch_end + 1):
        real_batch_id = 0
        real_test_dataset_batch_id = 0
        number_im_train = 1
        acc_train = 0
        train_similarity_value = torch.zeros(len(cf.nodes)-1).to(DEVICE)

        if cf.train_run == True:
            for batch_id, (img_batch, label_batch) in enumerate(train_loader):
                model.train_class(img_batch, label_batch, cf)
                real_batch_id = real_batch_id + 1
                optimizer.step(curr_epoch=epoch,curr_batch=real_batch_id,n_batches=number_im_train,batch_size=img_batch.size(0),)
                model.reconstruction_cal()
                train_similarity_value += similarity_tensor(model.all_layers_reconstruction, model.Y[0])
                acc_train += datasets.accuracy(model.Y_Pre[-1], label_batch)

            print("Average of train is {}.".format(acc_train/real_batch_id))
            print("similarity_value of train is {}.".format(train_similarity_value[0]/real_batch_id))

            ###############TEST-REC################
            if cf.plot_reconstruction == True:
                model.dif_layers_reconstruction(img_batch)
                save_dif_pre_image_order(model.layers_reconstruction, model.Y[0], epoch, cf, train=True)

        if cf.test2_run == True:
            acc_test = 0
            class_time = 0
            reconstruction_time_last_layer = 0
            test_similarity_value = torch.zeros(len(cf.nodes) - 1).to(DEVICE)

            for batch_id, (img_batch, label_batch) in enumerate(test_loader):
                real_test_dataset_batch_id = real_test_dataset_batch_id + 1
                model.test_class(img_batch, label_batch, cf)

                acc_test += datasets.accuracy(model.pre_class, label_batch)
                class_time += model.class_time
                reconstruction_time_last_layer += model.reconstruction_time[-1]
                model.reconstruction_cal()
                test_similarity_value += similarity_tensor(model.all_layers_reconstruction, model.Y[0])

            print("Average of test_2 is {}.".format(acc_test/real_test_dataset_batch_id))
            print("similarity_value of test_2 is {}.".format(test_similarity_value[0]/real_test_dataset_batch_id))
            if cf.plot_reconstruction == True:
                save_dif_pre_image_order(model.layers_reconstruction, model.Y[0], epoch, cf, train=False)

if __name__ == "__main__":
    cf = Initialize_Paremeter()
    main(cf)

