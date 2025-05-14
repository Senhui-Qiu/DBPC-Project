import os
import pprint
import torch
from pypc import datasets
from pypc import utils
from pypc.models_pc import PC_Model,set_seed
from pypc import optim
from torch.utils.tensorboard import SummaryWriter
from pypc.similarity_algorithm import similarity_tensor, ssim_tensor, save_test_dif_pre_image_order

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def Initialize_Paremeter():
    cf = utils.AttrDict()
    set_seed()
    cf.Epoch_end = 100
    cf.Epoch_start = 1
    cf.batch_size = 32
    cf.Train_iteration = 50
    cf.Test_iteration = 50
    cf.dy_lr = 0.6
    cf.du_lr = 0.0006
    cf.du_beta = 1
    cf.du_alpha = 0.5
    cf.dy_beta = 1
    cf.dy_alpha = 0
    cf.init_std = 0.01
    cf.shuffle = True
    cf.Y0_limited = False
    cf.comparison_neuron_max = False
    cf.load_model = False
    cf.train_run = True
    cf.test1_run = False
    cf.test2_run = True
    cf.calculate_neuron_value_train = False
    cf.calculate_neuron_value_test_1 = False
    cf.calculate_neuron_value_test_2 = False
    cf.AddGaussianNoise = False
    cf.addGaussianNoiselables = False
    cf.AddRotaion = True
    cf.ResizedCrop_size = 28
    cf.each_epoch_change_datasets = True
    cf.plot_reconstruction = True
    cf.use_error_c = False
    cf.use_population_coding = False
    cf.expand_size = 1
    cf.shrink_size = 10
    cf.train_dataset = 0
    cf.weight_no_share_num = 0
    ##########convolution para################
    if cf.train_dataset == 0:
        cf.Epoch_end = 100
        cf.first_layer_channel = 1
        cf.out_size = 28
        cf.Train_iteration = 5
        cf.Test_iteration = 5
        cf.dy_lr = 0.6
        cf.du_lr = 0.0006
        cf.du_beta = 0.2
        cf.du_alpha = 0.004
    elif cf.train_dataset == 1:
        cf.first_layer_channel = 3
        cf.out_size = 32
        cf.Train_iteration = 5
        cf.Test_iteration = 5
        cf.dy_lr = 0.01
        cf.du_lr = 0.0001
        cf.du_beta = 1
        cf.du_alpha = 0.1
    elif cf.train_dataset == 2:
        cf.batch_size = 64
        cf.first_layer_channel = 1
        cf.out_size = 28
        cf.Train_iteration = 25
        cf.Test_iteration = 25
        cf.dy_lr = 0.6
        cf.du_lr = 0.0006
        cf.du_beta = 1
        cf.du_alpha = 0.04
    cf.weight_share = True
    cf.network_layer = 0
    #output_size = [(W-K+2P)/S]+1
    if cf.network_layer == 0:
        cf.num_channels = [cf.first_layer_channel, 16, 32, 32, 48, 48, 0]
        cf.kernel_size = [3, 3, 3, 3, 3, 0]
        cf.stride = [1, 1, 1, 1, 1, 0]
        cf.padding = [1, 1, 1, 1, 1, 0]
        cf.nodes = [cf.out_size, cf.out_size, cf.out_size, cf.out_size, cf.out_size, cf.out_size, 10*cf.expand_size]  # kernel=3 stride=1 padding=1
    elif cf.network_layer == 1:
        cf.num_channels = [cf.first_layer_channel, 16, 32, 32, 48, 48, 64, 64, 0]
        cf.kernel_size = [3, 3, 3, 3, 3, 3, 3, 0]
        cf.stride = [1, 1, 1, 1, 1, 1, 1, 0]
        cf.padding = [1, 1, 1, 1, 1, 1, 1, 0]
        cf.nodes = [cf.out_size, cf.out_size, cf.out_size, cf.out_size, cf.out_size, cf.out_size, cf.out_size, cf.out_size, 10*cf.expand_size]  # kernel=3 stride=1 padding=1
    elif cf.network_layer == 2:
        cf.num_channels = [cf.first_layer_channel, 16, 32, 32, 48, 48, 64, 64, 96, 96, 0]
        cf.kernel_size = [3, 3, 3, 3, 3, 3, 3, 3, 3, 0]
        cf.stride = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
        cf.padding = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
        cf.nodes = [cf.out_size, cf.out_size, cf.out_size, cf.out_size, cf.out_size, cf.out_size, cf.out_size, cf.out_size, cf.out_size, cf.out_size, 10*cf.expand_size]  # kernel=3 stride=1 padding=1
    elif cf.network_layer == 3:
        cf.num_channels = [cf.first_layer_channel, 16, 32, 32, 64, 64, 96, 96, 0]
        cf.kernel_size = [3, 3, 3, 3, 3, 3, 3, 0]
        cf.stride = [1, 1, 1, 1, 1, 1, 1, 0]
        cf.padding = [1, 1, 1, 1, 1, 1, 1, 0]
        cf.nodes = [cf.out_size, cf.out_size, cf.out_size, cf.out_size, cf.out_size, cf.out_size, cf.out_size, cf.out_size, 10*cf.expand_size]  # kernel=3 stride=1 padding=1

    cf.act_function = utils.ReLU()
    cf.use_bias = False
    cf.optim = "Adam"
    cf.optim_batch_scale = False
    cf.optim_grad_clip = None
    cf.optim_weight_decay = None
    cf.decay_dy = 0
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
    elif cf.train_dataset == 2:
        train_dataset = datasets.FashionMNIST(cf, train=True)
        test_dataset = datasets.FashionMNIST(cf, train=False)
    train_loader = datasets.get_dataloader(train_dataset, cf.shuffle, cf.batch_size)
    test_loader = datasets.get_dataloader(test_dataset, cf.shuffle, cf.batch_size)
    print("Loaded data [train batches: {}, test batches: {}]".format(len(train_loader),len(test_loader)))

    ######################--Creat PC Model--##################
    model = PC_Model(cf)
    ######################--Method of optimization--##################
    optimizer = optim.get_optim(model.params, cf)

    Log_dir = os.path.join('tensorboard', 'curve_1')
    writer_1 = SummaryWriter(log_dir=Log_dir)
    Log_dir = os.path.join('tensorboard', 'curve_2')
    writer_2 = SummaryWriter(log_dir=Log_dir)
    Log_dir = os.path.join('tensorboard', 'curve_3')
    writer_3 = SummaryWriter(log_dir=Log_dir)
    Log_dir = os.path.join('tensorboard', 'curve_4')
    writer_4 = SummaryWriter(log_dir=Log_dir)
    Log_dir = os.path.join('tensorboard', 'curve_5')
    writer_5 = SummaryWriter(log_dir=Log_dir)
    Log_dir = os.path.join('tensorboard', 'curve_6')
    writer_6 = SummaryWriter(log_dir=Log_dir)

    Log_dir = os.path.join('tensorboard', 'curve_21')
    writer_21 = SummaryWriter(log_dir=Log_dir)
    Log_dir = os.path.join('tensorboard', 'curve_22')
    writer_22 = SummaryWriter(log_dir=Log_dir)
    Log_dir = os.path.join('tensorboard', 'curve_23')
    writer_23 = SummaryWriter(log_dir=Log_dir)
    Log_dir = os.path.join('tensorboard', 'curve_24')
    writer_24 = SummaryWriter(log_dir=Log_dir)
    Log_dir = os.path.join('tensorboard', 'curve_25')
    writer_25 = SummaryWriter(log_dir=Log_dir)
    Log_dir = os.path.join('tensorboard', 'curve_26')
    writer_26 = SummaryWriter(log_dir=Log_dir)

    Log_dir = os.path.join('tensorboard', 'curve_31')
    writer_31 = SummaryWriter(log_dir=Log_dir)
    Log_dir = os.path.join('tensorboard', 'curve_32')
    writer_32 = SummaryWriter(log_dir=Log_dir)
    Log_dir = os.path.join('tensorboard', 'curve_33')
    writer_33 = SummaryWriter(log_dir=Log_dir)
    Log_dir = os.path.join('tensorboard', 'curve_34')
    writer_34 = SummaryWriter(log_dir=Log_dir)
    Log_dir = os.path.join('tensorboard', 'curve_35')
    writer_35 = SummaryWriter(log_dir=Log_dir)
    Log_dir = os.path.join('tensorboard', 'curve_36')
    writer_36 = SummaryWriter(log_dir=Log_dir)

    Log_dir = os.path.join('tensorboard', 'curve_51')
    writer_51 = SummaryWriter(log_dir=Log_dir)
    Log_dir = os.path.join('tensorboard', 'curve_52')
    writer_52 = SummaryWriter(log_dir=Log_dir)
    Log_dir = os.path.join('tensorboard', 'curve_53')
    writer_53 = SummaryWriter(log_dir=Log_dir)
    Log_dir = os.path.join('tensorboard', 'curve_54')
    writer_54 = SummaryWriter(log_dir=Log_dir)
    Log_dir = os.path.join('tensorboard', 'curve_55')
    writer_55 = SummaryWriter(log_dir=Log_dir)
    Log_dir = os.path.join('tensorboard', 'curve_56')
    writer_56 = SummaryWriter(log_dir=Log_dir)

    if cf.load_model == True:
        epoch_totall_number = len(train_loader)
        epoch_testdataset_totall_number = len(test_loader)
    else:
        epoch_totall_number = 0
        epoch_testdataset_totall_number = 0
    with torch.no_grad():
        for epoch in range(cf.Epoch_start, cf.Epoch_end + 1):
            if epoch % 1 == 0:
                print("The number of train epoch is {}.".format(epoch))
            if cf.each_epoch_change_datasets:
                if cf.train_dataset == 0:
                    train_dataset = datasets.MNIST(cf, train=True)
                    train_loader = datasets.get_dataloader(train_dataset, cf.shuffle, cf.batch_size)
                elif cf.train_dataset == 1:
                    train_dataset = datasets.CIFAR10(cf, train=True)
                    train_loader = datasets.get_dataloader(train_dataset, cf.shuffle, cf.batch_size)
                elif cf.train_dataset == 2:
                    train_dataset = datasets.FashionMNIST(cf, train=True)
                    train_loader = datasets.get_dataloader(train_dataset, cf.shuffle, cf.batch_size)
            real_batch_id = 0
            real_test_dataset_batch_id = 0
            number_im_train = 1
            acc_train = 0
            train_similarity_value = torch.zeros(len(cf.nodes)-1).to(DEVICE)
            train_ssim_loss_value = torch.zeros(len(cf.nodes)-1).to(DEVICE)

            if cf.train_run == True:
                if cf.load_model == True:
                    if epoch == cf.Epoch_start:#only download once
                        model.load(optimizer, cf.logdir, epoch - 1, cf.weight_share)
                for batch_id, (img_batch, label_batch) in enumerate(train_loader):
                    model.train_class(img_batch, label_batch, cf)
                    real_batch_id = real_batch_id + 1
                    optimizer.step(curr_epoch=epoch,curr_batch=real_batch_id,n_batches=number_im_train,batch_size=img_batch.size(0))

                    if real_batch_id % 100  == 0:
                        for l in range (0, len(cf.nodes)):
                            if l == 0:
                                writer_1.add_scalar('loss_train_class', model.loss_list[0], global_step=(epoch-1)*epoch_totall_number+real_batch_id)
                                writer_31.add_scalar('loss_train_recon', model.loss_list_rev[0], global_step=(epoch - 1) * epoch_totall_number + real_batch_id)
                            if l == 1:
                                writer_2.add_scalar('loss_train_class', model.loss_list[1], global_step=(epoch-1)*epoch_totall_number+real_batch_id)
                                writer_32.add_scalar('loss_train_recon', model.loss_list_rev[1],global_step=(epoch - 1) * epoch_totall_number + real_batch_id)
                            if l == 2:
                                writer_3.add_scalar('loss_train_class', model.loss_list[2], global_step=(epoch-1)*epoch_totall_number+real_batch_id)
                                writer_33.add_scalar('loss_train_recon', model.loss_list_rev[2], global_step=(epoch - 1) * epoch_totall_number + real_batch_id)
                            if l == 3:
                                writer_4.add_scalar('loss_train_class', model.loss_list[3], global_step=(epoch-1)*epoch_totall_number+real_batch_id)
                                writer_34.add_scalar('loss_train_recon', model.loss_list_rev[3], global_step=(epoch - 1) * epoch_totall_number + real_batch_id)
                            if l == 4:
                                writer_5.add_scalar('loss_train_class', model.loss_list[4], global_step=(epoch-1)*epoch_totall_number+real_batch_id)
                                writer_35.add_scalar('loss_train_recon', model.loss_list_rev[4], global_step=(epoch - 1) * epoch_totall_number + real_batch_id)
                            if l == 5:
                                writer_6.add_scalar('loss_train_class', model.loss_list[5], global_step=(epoch-1)*epoch_totall_number+real_batch_id)
                                writer_36.add_scalar('loss_train_recon', model.loss_list_rev[5], global_step=(epoch - 1) * epoch_totall_number + real_batch_id)

                    model.reconstruction_cal()
                    train_similarity_value += similarity_tensor(model.all_layers_reconstruction, model.Y[0])
                    train_ssim_loss_value += ssim_tensor(model.all_layers_reconstruction, model.Y[0])
                    if cf.expand_size > 1:
                        if epoch == cf.Epoch_end:
                            acc_train += datasets.accuracy_expand_average_simple(model.Y_Pre[-1], label_batch, cf)
                    elif cf.expand_size == 1:
                        acc_train += datasets.accuracy(model.Y_Pre[-1], label_batch)
                    print("Average of train is {}.".format(acc_train / real_batch_id))
                writer_1.add_scalar('Accuracy', acc_train / real_batch_id,global_step=epoch)
                writer_1.add_scalar('similarity_value', train_similarity_value[0]/ real_batch_id, global_step=epoch)
                writer_2.add_scalar('similarity_value', train_similarity_value[1] / real_batch_id, global_step=epoch)
                writer_3.add_scalar('similarity_value', train_similarity_value[2] / real_batch_id, global_step=epoch)
                writer_4.add_scalar('similarity_value', train_similarity_value[3] / real_batch_id, global_step=epoch)
                writer_5.add_scalar('similarity_value', train_similarity_value[4] / real_batch_id, global_step=epoch)
                writer_6.add_scalar('similarity_value', train_similarity_value[5] / real_batch_id, global_step=epoch)
                writer_21.add_scalar('ssim_loss_value', train_ssim_loss_value[0] / real_batch_id, global_step=epoch)
                writer_22.add_scalar('ssim_loss_value', train_ssim_loss_value[1] / real_batch_id, global_step=epoch)
                writer_23.add_scalar('ssim_loss_value', train_ssim_loss_value[2] / real_batch_id, global_step=epoch)
                writer_24.add_scalar('ssim_loss_value', train_ssim_loss_value[3] / real_batch_id, global_step=epoch)
                writer_25.add_scalar('ssim_loss_value', train_ssim_loss_value[4] / real_batch_id, global_step=epoch)
                writer_26.add_scalar('ssim_loss_value', train_ssim_loss_value[5] / real_batch_id, global_step=epoch)

                print("Average of train is {}.".format(acc_train/real_batch_id))
                print("similarity_value of train is {}.".format(train_similarity_value[-1]/real_batch_id))

                if epoch_totall_number == 0:
                    epoch_totall_number = real_batch_id

                if epoch > 40:
                    model.save(model, optimizer, cf.logdir, epoch, cf.weight_share)

            if cf.test2_run == True:
                acc_test = 0
                class_time = 0
                reconstruction_time_last_layer = 0
                test_similarity_value = torch.zeros(len(cf.nodes) - 1).to(DEVICE)
                test_ssim_loss_value = torch.zeros(len(cf.nodes) - 1).to(DEVICE)

                for batch_id, (img_batch, label_batch) in enumerate(test_loader):
                    real_test_dataset_batch_id = real_test_dataset_batch_id + 1
                    model.test_class(img_batch, label_batch, cf)
                    if real_test_dataset_batch_id % 100  == 0:
                        for l in range (0, len(cf.nodes)):
                            if l == 0:
                                writer_21.add_scalar('loss_test_class', model.test_class_loss_list[0], global_step=(epoch-1)*epoch_testdataset_totall_number+real_test_dataset_batch_id)
                                writer_51.add_scalar('loss_test_recon', model.Error_reconstruction[0], global_step=(epoch - 1) * epoch_testdataset_totall_number + real_test_dataset_batch_id)
                            if l == 1:
                                writer_22.add_scalar('loss_test_class', model.test_class_loss_list[1], global_step=(epoch-1)*epoch_testdataset_totall_number+real_test_dataset_batch_id)
                                writer_52.add_scalar('loss_test_recon', model.Error_reconstruction[1], global_step=(epoch - 1) * epoch_testdataset_totall_number + real_test_dataset_batch_id)
                            if l == 2:
                                writer_23.add_scalar('loss_test_class', model.test_class_loss_list[2], global_step=(epoch-1)*epoch_testdataset_totall_number+real_test_dataset_batch_id)
                                writer_53.add_scalar('loss_test_recon', model.Error_reconstruction[2], global_step=(epoch - 1) * epoch_testdataset_totall_number + real_test_dataset_batch_id)
                            if l == 3:
                                writer_24.add_scalar('loss_test_class', model.test_class_loss_list[3], global_step=(epoch-1)*epoch_testdataset_totall_number+real_test_dataset_batch_id)
                                writer_54.add_scalar('loss_test_recon', model.Error_reconstruction[3], global_step=(epoch - 1) * epoch_testdataset_totall_number + real_test_dataset_batch_id)
                            if l == 4:
                                writer_25.add_scalar('loss_test_class', model.test_class_loss_list[4], global_step=(epoch-1)*epoch_testdataset_totall_number+real_test_dataset_batch_id)
                                writer_55.add_scalar('loss_test_recon', model.Error_reconstruction[4], global_step=(epoch - 1) * epoch_testdataset_totall_number + real_test_dataset_batch_id)
                            if l == 5:
                                writer_26.add_scalar('loss_test_class', model.test_class_loss_list[5], global_step=(epoch-1)*epoch_testdataset_totall_number+real_test_dataset_batch_id)
                                writer_56.add_scalar('loss_test_recon', model.Error_reconstruction[5], global_step=(epoch - 1) * epoch_testdataset_totall_number + real_test_dataset_batch_id)
                    #############################################
                    if cf.expand_size > 1:
                        acc_test += datasets.accuracy_expand_average_simple(model.pre_class, label_batch, cf)
                    elif cf.expand_size == 1:
                        acc_test += datasets.accuracy(model.pre_class, label_batch)
                    class_time += model.class_time
                    reconstruction_time_last_layer += model.reconstruction_time[-1]
                    model.reconstruction_cal()
                    test_similarity_value += similarity_tensor(model.all_layers_reconstruction, model.Y[0])
                    test_ssim_loss_value += ssim_tensor(model.all_layers_reconstruction, model.Y[0])
                    ###############--print train_similarity_value--####################
                writer_1.add_scalar('class_time', class_time / (real_batch_id), global_step=epoch)
                writer_1.add_scalar('reconstruction_time_last_layer',reconstruction_time_last_layer / (real_batch_id),global_step=epoch)
                
                writer_21.add_scalar('similarity_value', test_similarity_value[0] / real_test_dataset_batch_id,global_step=epoch)
                writer_22.add_scalar('similarity_value', test_similarity_value[1] / real_test_dataset_batch_id,global_step=epoch)
                writer_23.add_scalar('similarity_value', test_similarity_value[2] / real_test_dataset_batch_id,global_step=epoch)
                writer_24.add_scalar('similarity_value', test_similarity_value[3] / real_test_dataset_batch_id, global_step=epoch)
                writer_25.add_scalar('similarity_value', test_similarity_value[4] / real_test_dataset_batch_id,global_step=epoch)
                writer_26.add_scalar('similarity_value', test_similarity_value[5] / real_test_dataset_batch_id, global_step=epoch)

                writer_31.add_scalar('ssim_loss_value', test_ssim_loss_value[0] / real_test_dataset_batch_id,global_step=epoch)
                writer_32.add_scalar('ssim_loss_value', test_ssim_loss_value[1] / real_test_dataset_batch_id,global_step=epoch)
                writer_33.add_scalar('ssim_loss_value', test_ssim_loss_value[2] / real_test_dataset_batch_id,global_step=epoch)
                writer_34.add_scalar('ssim_loss_value', test_ssim_loss_value[3] / real_test_dataset_batch_id, global_step=epoch)
                writer_35.add_scalar('ssim_loss_value', test_ssim_loss_value[4] / real_test_dataset_batch_id,global_step=epoch)
                writer_36.add_scalar('ssim_loss_value', test_ssim_loss_value[5] / real_test_dataset_batch_id, global_step=epoch)

                writer_5.add_scalar('Accuracy', acc_test/real_test_dataset_batch_id, global_step=epoch)
                print("Average of test_2 is {}.".format(acc_test/real_test_dataset_batch_id))
                print("similarity_value of test_2 is {}.".format(test_similarity_value[-1]/real_test_dataset_batch_id))

                if epoch_testdataset_totall_number == 0:
                    epoch_testdataset_totall_number = real_test_dataset_batch_id

                if cf.plot_reconstruction == True:
                    save_test_dif_pre_image_order(model.layers_reconstruction, model.Y[0], epoch, cf, train=False)

    return 0
if __name__ == "__main__":
    cf = Initialize_Paremeter()
    main(cf)

