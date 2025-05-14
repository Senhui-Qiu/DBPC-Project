import matplotlib.pyplot as plt
import torch
from numpy import *
from torchvision.utils import save_image
import torchvision
from torchvision.utils import make_grid
import math
import numpy as np
from pypc import ssim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
###################--Peak Signal-to-Noise Ratio--##########################
def psnr(img1, img2):
    mse = np.mean(((img1/1.0) - (img2/1.0)) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10*math.log10(255.0**2/mse)

def psnr_1(img1, img2):
    mse = np.mean(((img1/255.) - (img2/255.)) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20*math.log10(PIXEL_MAX/math.sqrt(mse))


def similarity_tensor(all_layers_reconstruction, Y_batch):
    out = torch.zeros(len(all_layers_reconstruction)).to(DEVICE)
    for i in range(len(all_layers_reconstruction)):
        Y_tem = all_layers_reconstruction[i]
        image_rec = Y_tem.mul(255.0).add_(0.5).clamp_(0, 255)
        image_original = Y_batch.mul(255.0).add_(0.5).clamp_(0, 255)
        # mse_1 = torch.mean(((Y_tem/1.0) - (Y_batch/1.0)) ** 2)
        mse = torch.mean(((image_rec/1.0) - (image_original/1.0)) ** 2)
        if mse < 1.0e-10:
            out[i] = 100
        out[i] = 10*math.log10(255.0**2/mse)
    return out

def ssim_tensor(all_layers_reconstruction, Y_batch):
    out = torch.zeros(len(all_layers_reconstruction)).to(DEVICE)
    for i in range(len(all_layers_reconstruction)):
        Y_tem = all_layers_reconstruction[i]
        image_rec = Y_tem.mul(255.0).add_(0.5).clamp_(0, 255).cpu()
        image_original = Y_batch.mul(255.0).add_(0.5).clamp_(0, 255).cpu()
        # save_image(Y_tem[0,], "result_reconstruction/ssmi_image_recon_{}.png".format((i)))
        # save_image(Y_batch[0,], "result_reconstruction/ssmi_image_original_{}.png".format((i)))
        out[i] = ssim.ssim(image_original, image_rec, data_range=255)
    return out


def save_pre_image(Y_Pre_batch, Y_batch, epoch):
    if epoch % 1 == 0:
        trans = torchvision.transforms.Resize(size=112)
        Y_Pre_out = trans(Y_Pre_batch[0])
        Y_out = trans(Y_batch[0])
        save_image(Y_Pre_out, "result_reconstruction/image_{}_pre.png".format((epoch - 1)))
        save_image(Y_out, "result_reconstruction/image_{}.png".format((epoch - 1)))


def save_dif_pre_image(Y_dif_Pre_batch, Y_batch, epoch, train):
    trans = torchvision.transforms.Resize(size=112)
    if (Y_batch.size())[1] == 784:
        Y_temp_out = torch.reshape(Y_batch[0], (1, 28, 28))  # fc need to add
    else:
        Y_temp_out = torch.reshape(Y_batch[0], (3, int(math.sqrt((Y_batch.size())[1]/3)), int(math.sqrt((Y_batch.size())[1]/3))))  # fc need to add
    Y_out = trans(Y_temp_out)
    if train == True:
        save_image(Y_out, "result_reconstruction/image_train_e{}.png".format(epoch))
    else:
        save_image(Y_out, "result_reconstruction/image_test_e{}.png".format(epoch))
    for n in range(0, len(Y_dif_Pre_batch)):
        if (Y_batch.size())[1] == 784:
            Y_Pre_temp_out = torch.reshape(Y_dif_Pre_batch[n][0], (1, 28, 28))#fc need to add
        else:
            Y_Pre_temp_out = torch.reshape(Y_dif_Pre_batch[n][0], (3, int(math.sqrt((Y_batch.size())[1]/3)), int(math.sqrt((Y_batch.size())[1]/3))))#fc need to add
        Y_Pre_out = trans(Y_Pre_temp_out)
        if train == True:
            save_image(Y_Pre_out, "result_reconstruction/image_train_e{}_l{}_pre.png".format(epoch, n))
        else:
            save_image(Y_Pre_out, "result_reconstruction/image_test_e{}_l{}_pre.png".format(epoch, n))


def save_dif_pre_image_order(Y_dif_Pre_batch_input, Y_batch, epoch, cf, train):
    plot_row = len(cf.nodes)-1
    if (Y_batch.size())[1] == 784:
        Y_temp_out = torch.reshape(Y_batch, ((Y_batch.size())[0], 1, 28, 28))  # fc need to add
    elif (Y_batch.size())[1] == 1024:
        Y_temp_out = torch.reshape(Y_batch, ((Y_batch.size())[0], 3, int(math.sqrt((Y_batch.size())[1]/3)), int(math.sqrt((Y_batch.size())[1]/3))))  # fc need to add
    else:
        Y_temp_out = Y_batch


    Y_dif_Pre_batch = []
    for n in range(0, len(Y_dif_Pre_batch_input)):
        if (Y_batch.size())[1] == 784:
            Y_Pre_temp_out = torch.reshape(Y_dif_Pre_batch_input[n], ((Y_batch.size())[0], 1, 28, 28))  # fc need to add
        elif (Y_batch.size())[1] == 1024:
            Y_Pre_temp_out = torch.reshape(Y_dif_Pre_batch_input[n], ((Y_batch.size())[0], 3, int(math.sqrt((Y_batch.size())[1]/3)), int(math.sqrt((Y_batch.size())[1]/3))))  # fc need to add
        else:
            Y_Pre_temp_out = Y_dif_Pre_batch_input[n]

        Y_dif_Pre_batch.append(Y_Pre_temp_out)
    # make grid
    if plot_row == 4:
        grid = make_grid(
            [Y_temp_out[0], Y_dif_Pre_batch[0][0], Y_dif_Pre_batch[1][0], Y_dif_Pre_batch[2][0],
             Y_temp_out[1], Y_dif_Pre_batch[0][1], Y_dif_Pre_batch[1][1], Y_dif_Pre_batch[2][1],
             Y_temp_out[2], Y_dif_Pre_batch[0][2], Y_dif_Pre_batch[1][2], Y_dif_Pre_batch[2][2],
             Y_temp_out[3], Y_dif_Pre_batch[0][3], Y_dif_Pre_batch[1][3], Y_dif_Pre_batch[2][3],
             Y_temp_out[4], Y_dif_Pre_batch[0][4], Y_dif_Pre_batch[1][4], Y_dif_Pre_batch[2][4],
             Y_temp_out[5], Y_dif_Pre_batch[0][5], Y_dif_Pre_batch[1][5], Y_dif_Pre_batch[2][5],
             Y_temp_out[6], Y_dif_Pre_batch[0][6], Y_dif_Pre_batch[1][6], Y_dif_Pre_batch[2][6],
             Y_temp_out[7], Y_dif_Pre_batch[0][7], Y_dif_Pre_batch[1][7], Y_dif_Pre_batch[2][7],
             Y_temp_out[8], Y_dif_Pre_batch[0][8], Y_dif_Pre_batch[1][8], Y_dif_Pre_batch[2][8],
             Y_temp_out[9], Y_dif_Pre_batch[0][9], Y_dif_Pre_batch[1][9], Y_dif_Pre_batch[2][9],
             ], nrow=plot_row)
    elif plot_row == 5:
        grid = make_grid(
            [Y_temp_out[0], Y_dif_Pre_batch[0][0], Y_dif_Pre_batch[1][0], Y_dif_Pre_batch[2][0], Y_dif_Pre_batch[3][0],
             Y_temp_out[1], Y_dif_Pre_batch[0][1], Y_dif_Pre_batch[1][1], Y_dif_Pre_batch[2][1], Y_dif_Pre_batch[3][1],
             Y_temp_out[2], Y_dif_Pre_batch[0][2], Y_dif_Pre_batch[1][2], Y_dif_Pre_batch[2][2], Y_dif_Pre_batch[3][2],
             Y_temp_out[3], Y_dif_Pre_batch[0][3], Y_dif_Pre_batch[1][3], Y_dif_Pre_batch[2][3], Y_dif_Pre_batch[3][3],
             Y_temp_out[4], Y_dif_Pre_batch[0][4], Y_dif_Pre_batch[1][4], Y_dif_Pre_batch[2][4], Y_dif_Pre_batch[3][4],
             Y_temp_out[5], Y_dif_Pre_batch[0][5], Y_dif_Pre_batch[1][5], Y_dif_Pre_batch[2][5], Y_dif_Pre_batch[3][5],
             Y_temp_out[6], Y_dif_Pre_batch[0][6], Y_dif_Pre_batch[1][6], Y_dif_Pre_batch[2][6], Y_dif_Pre_batch[3][6],
             Y_temp_out[7], Y_dif_Pre_batch[0][7], Y_dif_Pre_batch[1][7], Y_dif_Pre_batch[2][7], Y_dif_Pre_batch[3][7],
             Y_temp_out[8], Y_dif_Pre_batch[0][8], Y_dif_Pre_batch[1][8], Y_dif_Pre_batch[2][8], Y_dif_Pre_batch[3][8],
             Y_temp_out[9], Y_dif_Pre_batch[0][9], Y_dif_Pre_batch[1][9], Y_dif_Pre_batch[2][9], Y_dif_Pre_batch[3][9],
             ], nrow=plot_row)
    elif plot_row == 6:
        grid = make_grid(
            [Y_temp_out[0], Y_dif_Pre_batch[0][0], Y_dif_Pre_batch[1][0], Y_dif_Pre_batch[2][0], Y_dif_Pre_batch[3][0], Y_dif_Pre_batch[4][0],
             Y_temp_out[1], Y_dif_Pre_batch[0][1], Y_dif_Pre_batch[1][1], Y_dif_Pre_batch[2][1], Y_dif_Pre_batch[3][1], Y_dif_Pre_batch[4][1],
             Y_temp_out[2], Y_dif_Pre_batch[0][2], Y_dif_Pre_batch[1][2], Y_dif_Pre_batch[2][2], Y_dif_Pre_batch[3][2], Y_dif_Pre_batch[4][2],
             Y_temp_out[3], Y_dif_Pre_batch[0][3], Y_dif_Pre_batch[1][3], Y_dif_Pre_batch[2][3], Y_dif_Pre_batch[3][3], Y_dif_Pre_batch[4][3],
             Y_temp_out[4], Y_dif_Pre_batch[0][4], Y_dif_Pre_batch[1][4], Y_dif_Pre_batch[2][4], Y_dif_Pre_batch[3][4], Y_dif_Pre_batch[4][4],
             Y_temp_out[5], Y_dif_Pre_batch[0][5], Y_dif_Pre_batch[1][5], Y_dif_Pre_batch[2][5], Y_dif_Pre_batch[3][5], Y_dif_Pre_batch[4][5],
             Y_temp_out[6], Y_dif_Pre_batch[0][6], Y_dif_Pre_batch[1][6], Y_dif_Pre_batch[2][6], Y_dif_Pre_batch[3][6], Y_dif_Pre_batch[4][6],
             Y_temp_out[7], Y_dif_Pre_batch[0][7], Y_dif_Pre_batch[1][7], Y_dif_Pre_batch[2][7], Y_dif_Pre_batch[3][7], Y_dif_Pre_batch[4][7],
             Y_temp_out[8], Y_dif_Pre_batch[0][8], Y_dif_Pre_batch[1][8], Y_dif_Pre_batch[2][8], Y_dif_Pre_batch[3][8], Y_dif_Pre_batch[4][8],
             Y_temp_out[9], Y_dif_Pre_batch[0][9], Y_dif_Pre_batch[1][9], Y_dif_Pre_batch[2][9], Y_dif_Pre_batch[3][9], Y_dif_Pre_batch[4][9],
             ], nrow=plot_row)
    elif plot_row == 7:
        grid = make_grid(
            [Y_temp_out[0], Y_dif_Pre_batch[0][0], Y_dif_Pre_batch[1][0], Y_dif_Pre_batch[2][0], Y_dif_Pre_batch[3][0], Y_dif_Pre_batch[4][0], Y_dif_Pre_batch[5][0],
             Y_temp_out[1], Y_dif_Pre_batch[0][1], Y_dif_Pre_batch[1][1], Y_dif_Pre_batch[2][1], Y_dif_Pre_batch[3][1], Y_dif_Pre_batch[4][1], Y_dif_Pre_batch[5][1],
             Y_temp_out[2], Y_dif_Pre_batch[0][2], Y_dif_Pre_batch[1][2], Y_dif_Pre_batch[2][2], Y_dif_Pre_batch[3][2], Y_dif_Pre_batch[4][2], Y_dif_Pre_batch[5][2],
             Y_temp_out[3], Y_dif_Pre_batch[0][3], Y_dif_Pre_batch[1][3], Y_dif_Pre_batch[2][3], Y_dif_Pre_batch[3][3], Y_dif_Pre_batch[4][3], Y_dif_Pre_batch[5][3],
             Y_temp_out[4], Y_dif_Pre_batch[0][4], Y_dif_Pre_batch[1][4], Y_dif_Pre_batch[2][4], Y_dif_Pre_batch[3][4], Y_dif_Pre_batch[4][4], Y_dif_Pre_batch[5][4],
             Y_temp_out[5], Y_dif_Pre_batch[0][5], Y_dif_Pre_batch[1][5], Y_dif_Pre_batch[2][5], Y_dif_Pre_batch[3][5], Y_dif_Pre_batch[4][5], Y_dif_Pre_batch[5][5],
             Y_temp_out[5], Y_dif_Pre_batch[0][5], Y_dif_Pre_batch[1][5], Y_dif_Pre_batch[2][5], Y_dif_Pre_batch[3][5], Y_dif_Pre_batch[4][5], Y_dif_Pre_batch[5][5],
             Y_temp_out[6], Y_dif_Pre_batch[0][6], Y_dif_Pre_batch[1][6], Y_dif_Pre_batch[2][6], Y_dif_Pre_batch[3][6], Y_dif_Pre_batch[4][6], Y_dif_Pre_batch[5][6],
             Y_temp_out[7], Y_dif_Pre_batch[0][7], Y_dif_Pre_batch[1][7], Y_dif_Pre_batch[2][7], Y_dif_Pre_batch[3][7], Y_dif_Pre_batch[4][7], Y_dif_Pre_batch[5][7],
             Y_temp_out[8], Y_dif_Pre_batch[0][8], Y_dif_Pre_batch[1][8], Y_dif_Pre_batch[2][8], Y_dif_Pre_batch[3][8], Y_dif_Pre_batch[4][8], Y_dif_Pre_batch[5][8],
             Y_temp_out[9], Y_dif_Pre_batch[0][9], Y_dif_Pre_batch[1][9], Y_dif_Pre_batch[2][9], Y_dif_Pre_batch[3][9], Y_dif_Pre_batch[4][9], Y_dif_Pre_batch[5][9],
             ], nrow=plot_row)
    elif plot_row == 8:
        grid = make_grid(
            [Y_temp_out[0], Y_dif_Pre_batch[0][0], Y_dif_Pre_batch[1][0], Y_dif_Pre_batch[2][0], Y_dif_Pre_batch[3][0], Y_dif_Pre_batch[4][0], Y_dif_Pre_batch[5][0], Y_dif_Pre_batch[6][0],
             Y_temp_out[1], Y_dif_Pre_batch[0][1], Y_dif_Pre_batch[1][1], Y_dif_Pre_batch[2][1], Y_dif_Pre_batch[3][1], Y_dif_Pre_batch[4][1], Y_dif_Pre_batch[5][1], Y_dif_Pre_batch[6][1],
             Y_temp_out[2], Y_dif_Pre_batch[0][2], Y_dif_Pre_batch[1][2], Y_dif_Pre_batch[2][2], Y_dif_Pre_batch[3][2], Y_dif_Pre_batch[4][2], Y_dif_Pre_batch[5][2], Y_dif_Pre_batch[6][2],
             Y_temp_out[3], Y_dif_Pre_batch[0][3], Y_dif_Pre_batch[1][3], Y_dif_Pre_batch[2][3], Y_dif_Pre_batch[3][3], Y_dif_Pre_batch[4][3], Y_dif_Pre_batch[5][3], Y_dif_Pre_batch[6][3],
             Y_temp_out[4], Y_dif_Pre_batch[0][4], Y_dif_Pre_batch[1][4], Y_dif_Pre_batch[2][4], Y_dif_Pre_batch[3][4], Y_dif_Pre_batch[4][4], Y_dif_Pre_batch[5][4], Y_dif_Pre_batch[6][4],
             Y_temp_out[5], Y_dif_Pre_batch[0][5], Y_dif_Pre_batch[1][5], Y_dif_Pre_batch[2][5], Y_dif_Pre_batch[3][5], Y_dif_Pre_batch[4][5], Y_dif_Pre_batch[5][5], Y_dif_Pre_batch[6][5],
             Y_temp_out[6], Y_dif_Pre_batch[0][6], Y_dif_Pre_batch[1][6], Y_dif_Pre_batch[2][6], Y_dif_Pre_batch[3][6], Y_dif_Pre_batch[4][6], Y_dif_Pre_batch[5][6], Y_dif_Pre_batch[6][6],
             Y_temp_out[5], Y_dif_Pre_batch[0][5], Y_dif_Pre_batch[1][5], Y_dif_Pre_batch[2][5], Y_dif_Pre_batch[3][5], Y_dif_Pre_batch[4][5], Y_dif_Pre_batch[5][5], Y_dif_Pre_batch[6][5],
             Y_temp_out[6], Y_dif_Pre_batch[0][6], Y_dif_Pre_batch[1][6], Y_dif_Pre_batch[2][6], Y_dif_Pre_batch[3][6], Y_dif_Pre_batch[4][6], Y_dif_Pre_batch[5][6], Y_dif_Pre_batch[6][6],
             Y_temp_out[7], Y_dif_Pre_batch[0][7], Y_dif_Pre_batch[1][7], Y_dif_Pre_batch[2][7], Y_dif_Pre_batch[3][7], Y_dif_Pre_batch[4][7], Y_dif_Pre_batch[5][7], Y_dif_Pre_batch[6][7],
             Y_temp_out[8], Y_dif_Pre_batch[0][8], Y_dif_Pre_batch[1][8], Y_dif_Pre_batch[2][8], Y_dif_Pre_batch[3][8], Y_dif_Pre_batch[4][8], Y_dif_Pre_batch[5][8], Y_dif_Pre_batch[6][8],
             Y_temp_out[9], Y_dif_Pre_batch[0][9], Y_dif_Pre_batch[1][9], Y_dif_Pre_batch[2][9], Y_dif_Pre_batch[3][9], Y_dif_Pre_batch[4][9], Y_dif_Pre_batch[5][9], Y_dif_Pre_batch[6][9],
             ], nrow=plot_row)
    elif plot_row == 9:
        grid = make_grid(
            [Y_temp_out[0], Y_dif_Pre_batch[0][0], Y_dif_Pre_batch[1][0], Y_dif_Pre_batch[2][0], Y_dif_Pre_batch[3][0], Y_dif_Pre_batch[4][0], Y_dif_Pre_batch[5][0], Y_dif_Pre_batch[6][0], Y_dif_Pre_batch[7][0],
             Y_temp_out[1], Y_dif_Pre_batch[0][1], Y_dif_Pre_batch[1][1], Y_dif_Pre_batch[2][1], Y_dif_Pre_batch[3][1], Y_dif_Pre_batch[4][1], Y_dif_Pre_batch[5][1], Y_dif_Pre_batch[6][1], Y_dif_Pre_batch[7][1],
             Y_temp_out[2], Y_dif_Pre_batch[0][2], Y_dif_Pre_batch[1][2], Y_dif_Pre_batch[2][2], Y_dif_Pre_batch[3][2], Y_dif_Pre_batch[4][2], Y_dif_Pre_batch[5][2], Y_dif_Pre_batch[6][2], Y_dif_Pre_batch[7][2],
             Y_temp_out[3], Y_dif_Pre_batch[0][3], Y_dif_Pre_batch[1][3], Y_dif_Pre_batch[2][3], Y_dif_Pre_batch[3][3], Y_dif_Pre_batch[4][3], Y_dif_Pre_batch[5][3], Y_dif_Pre_batch[6][3], Y_dif_Pre_batch[7][3],
             Y_temp_out[4], Y_dif_Pre_batch[0][4], Y_dif_Pre_batch[1][4], Y_dif_Pre_batch[2][4], Y_dif_Pre_batch[3][4], Y_dif_Pre_batch[4][4], Y_dif_Pre_batch[5][4], Y_dif_Pre_batch[6][4], Y_dif_Pre_batch[7][4],
             Y_temp_out[5], Y_dif_Pre_batch[0][5], Y_dif_Pre_batch[1][5], Y_dif_Pre_batch[2][5], Y_dif_Pre_batch[3][5], Y_dif_Pre_batch[4][5], Y_dif_Pre_batch[5][5], Y_dif_Pre_batch[6][5], Y_dif_Pre_batch[7][5],
             Y_temp_out[6], Y_dif_Pre_batch[0][6], Y_dif_Pre_batch[1][6], Y_dif_Pre_batch[2][6], Y_dif_Pre_batch[3][6], Y_dif_Pre_batch[4][6], Y_dif_Pre_batch[5][6], Y_dif_Pre_batch[6][6], Y_dif_Pre_batch[7][6],
             ], nrow=plot_row)
    elif plot_row == 10:
        grid = make_grid(
            [Y_temp_out[0], Y_dif_Pre_batch[0][0], Y_dif_Pre_batch[1][0], Y_dif_Pre_batch[2][0], Y_dif_Pre_batch[3][0], Y_dif_Pre_batch[4][0], Y_dif_Pre_batch[5][0], Y_dif_Pre_batch[6][0], Y_dif_Pre_batch[7][0], Y_dif_Pre_batch[8][0],
             Y_temp_out[1], Y_dif_Pre_batch[0][1], Y_dif_Pre_batch[1][1], Y_dif_Pre_batch[2][1], Y_dif_Pre_batch[3][1], Y_dif_Pre_batch[4][1], Y_dif_Pre_batch[5][1], Y_dif_Pre_batch[6][1], Y_dif_Pre_batch[7][1], Y_dif_Pre_batch[8][1],
             Y_temp_out[2], Y_dif_Pre_batch[0][2], Y_dif_Pre_batch[1][2], Y_dif_Pre_batch[2][2], Y_dif_Pre_batch[3][2], Y_dif_Pre_batch[4][2], Y_dif_Pre_batch[5][2], Y_dif_Pre_batch[6][2], Y_dif_Pre_batch[7][2], Y_dif_Pre_batch[8][2],
             Y_temp_out[3], Y_dif_Pre_batch[0][3], Y_dif_Pre_batch[1][3], Y_dif_Pre_batch[2][3], Y_dif_Pre_batch[3][3], Y_dif_Pre_batch[4][3], Y_dif_Pre_batch[5][3], Y_dif_Pre_batch[6][3], Y_dif_Pre_batch[7][3], Y_dif_Pre_batch[8][3],
             Y_temp_out[4], Y_dif_Pre_batch[0][4], Y_dif_Pre_batch[1][4], Y_dif_Pre_batch[2][4], Y_dif_Pre_batch[3][4], Y_dif_Pre_batch[4][4], Y_dif_Pre_batch[5][4], Y_dif_Pre_batch[6][4], Y_dif_Pre_batch[7][4], Y_dif_Pre_batch[8][4],
             Y_temp_out[5], Y_dif_Pre_batch[0][5], Y_dif_Pre_batch[1][5], Y_dif_Pre_batch[2][5], Y_dif_Pre_batch[3][5], Y_dif_Pre_batch[4][5], Y_dif_Pre_batch[5][5], Y_dif_Pre_batch[6][5], Y_dif_Pre_batch[7][5], Y_dif_Pre_batch[8][5],
             Y_temp_out[6], Y_dif_Pre_batch[0][6], Y_dif_Pre_batch[1][6], Y_dif_Pre_batch[2][6], Y_dif_Pre_batch[3][6], Y_dif_Pre_batch[4][6], Y_dif_Pre_batch[5][6], Y_dif_Pre_batch[6][6], Y_dif_Pre_batch[7][6], Y_dif_Pre_batch[8][6],
             ], nrow=plot_row)
    elif plot_row == 11:
        grid = make_grid(
            [Y_temp_out[0], Y_dif_Pre_batch[0][0], Y_dif_Pre_batch[1][0], Y_dif_Pre_batch[2][0], Y_dif_Pre_batch[3][0], Y_dif_Pre_batch[4][0], Y_dif_Pre_batch[5][0], Y_dif_Pre_batch[6][0], Y_dif_Pre_batch[7][0], Y_dif_Pre_batch[8][0], Y_dif_Pre_batch[9][0],
             Y_temp_out[1], Y_dif_Pre_batch[0][1], Y_dif_Pre_batch[1][1], Y_dif_Pre_batch[2][1], Y_dif_Pre_batch[3][1], Y_dif_Pre_batch[4][1], Y_dif_Pre_batch[5][1], Y_dif_Pre_batch[6][1], Y_dif_Pre_batch[7][1], Y_dif_Pre_batch[8][1], Y_dif_Pre_batch[9][1],
             Y_temp_out[2], Y_dif_Pre_batch[0][2], Y_dif_Pre_batch[1][2], Y_dif_Pre_batch[2][2], Y_dif_Pre_batch[3][2], Y_dif_Pre_batch[4][2], Y_dif_Pre_batch[5][2], Y_dif_Pre_batch[6][2], Y_dif_Pre_batch[7][2], Y_dif_Pre_batch[8][2], Y_dif_Pre_batch[9][2],
             Y_temp_out[3], Y_dif_Pre_batch[0][3], Y_dif_Pre_batch[1][3], Y_dif_Pre_batch[2][3], Y_dif_Pre_batch[3][3], Y_dif_Pre_batch[4][3], Y_dif_Pre_batch[5][3], Y_dif_Pre_batch[6][3], Y_dif_Pre_batch[7][3], Y_dif_Pre_batch[8][3], Y_dif_Pre_batch[9][3],
             Y_temp_out[4], Y_dif_Pre_batch[0][4], Y_dif_Pre_batch[1][4], Y_dif_Pre_batch[2][4], Y_dif_Pre_batch[3][4], Y_dif_Pre_batch[4][4], Y_dif_Pre_batch[5][4], Y_dif_Pre_batch[6][4], Y_dif_Pre_batch[7][4], Y_dif_Pre_batch[8][4], Y_dif_Pre_batch[9][4],
             Y_temp_out[5], Y_dif_Pre_batch[0][5], Y_dif_Pre_batch[1][5], Y_dif_Pre_batch[2][5], Y_dif_Pre_batch[3][5], Y_dif_Pre_batch[4][5], Y_dif_Pre_batch[5][5], Y_dif_Pre_batch[6][5], Y_dif_Pre_batch[7][5], Y_dif_Pre_batch[8][5], Y_dif_Pre_batch[9][5],
             Y_temp_out[6], Y_dif_Pre_batch[0][6], Y_dif_Pre_batch[1][6], Y_dif_Pre_batch[2][6], Y_dif_Pre_batch[3][6], Y_dif_Pre_batch[4][6], Y_dif_Pre_batch[5][6], Y_dif_Pre_batch[6][6], Y_dif_Pre_batch[7][6], Y_dif_Pre_batch[8][6], Y_dif_Pre_batch[9][6],
             ], nrow=plot_row)
    elif plot_row == 12:
        grid = make_grid(
            [Y_temp_out[0], Y_dif_Pre_batch[0][0], Y_dif_Pre_batch[1][0], Y_dif_Pre_batch[2][0], Y_dif_Pre_batch[3][0], Y_dif_Pre_batch[4][0], Y_dif_Pre_batch[5][0], Y_dif_Pre_batch[6][0], Y_dif_Pre_batch[7][0], Y_dif_Pre_batch[8][0], Y_dif_Pre_batch[9][0], Y_dif_Pre_batch[10][0],
             Y_temp_out[1], Y_dif_Pre_batch[0][1], Y_dif_Pre_batch[1][1], Y_dif_Pre_batch[2][1], Y_dif_Pre_batch[3][1], Y_dif_Pre_batch[4][1], Y_dif_Pre_batch[5][1], Y_dif_Pre_batch[6][1], Y_dif_Pre_batch[7][1], Y_dif_Pre_batch[8][1], Y_dif_Pre_batch[9][1], Y_dif_Pre_batch[10][1],
             Y_temp_out[2], Y_dif_Pre_batch[0][2], Y_dif_Pre_batch[1][2], Y_dif_Pre_batch[2][2], Y_dif_Pre_batch[3][2], Y_dif_Pre_batch[4][2], Y_dif_Pre_batch[5][2], Y_dif_Pre_batch[6][2], Y_dif_Pre_batch[7][2], Y_dif_Pre_batch[8][2], Y_dif_Pre_batch[9][2], Y_dif_Pre_batch[10][2],
             Y_temp_out[3], Y_dif_Pre_batch[0][3], Y_dif_Pre_batch[1][3], Y_dif_Pre_batch[2][3], Y_dif_Pre_batch[3][3], Y_dif_Pre_batch[4][3], Y_dif_Pre_batch[5][3], Y_dif_Pre_batch[6][3], Y_dif_Pre_batch[7][3], Y_dif_Pre_batch[8][3], Y_dif_Pre_batch[9][3], Y_dif_Pre_batch[10][3],
             Y_temp_out[4], Y_dif_Pre_batch[0][4], Y_dif_Pre_batch[1][4], Y_dif_Pre_batch[2][4], Y_dif_Pre_batch[3][4], Y_dif_Pre_batch[4][4], Y_dif_Pre_batch[5][4], Y_dif_Pre_batch[6][4], Y_dif_Pre_batch[7][4], Y_dif_Pre_batch[8][4], Y_dif_Pre_batch[9][4], Y_dif_Pre_batch[10][4],
             Y_temp_out[5], Y_dif_Pre_batch[0][5], Y_dif_Pre_batch[1][5], Y_dif_Pre_batch[2][5], Y_dif_Pre_batch[3][5], Y_dif_Pre_batch[4][5], Y_dif_Pre_batch[5][5], Y_dif_Pre_batch[6][5], Y_dif_Pre_batch[7][5], Y_dif_Pre_batch[8][5], Y_dif_Pre_batch[9][5], Y_dif_Pre_batch[10][5],
             Y_temp_out[6], Y_dif_Pre_batch[0][6], Y_dif_Pre_batch[1][6], Y_dif_Pre_batch[2][6], Y_dif_Pre_batch[3][6], Y_dif_Pre_batch[4][6], Y_dif_Pre_batch[5][6], Y_dif_Pre_batch[6][6], Y_dif_Pre_batch[7][6], Y_dif_Pre_batch[8][6], Y_dif_Pre_batch[9][6], Y_dif_Pre_batch[10][6],
             ], nrow=plot_row)
    elif plot_row == 13:
        grid = make_grid(
            [Y_temp_out[0], Y_dif_Pre_batch[0][0], Y_dif_Pre_batch[1][0], Y_dif_Pre_batch[2][0], Y_dif_Pre_batch[3][0], Y_dif_Pre_batch[4][0], Y_dif_Pre_batch[5][0], Y_dif_Pre_batch[6][0], Y_dif_Pre_batch[7][0], Y_dif_Pre_batch[8][0], Y_dif_Pre_batch[9][0], Y_dif_Pre_batch[10][0], Y_dif_Pre_batch[11][0],
             Y_temp_out[1], Y_dif_Pre_batch[0][1], Y_dif_Pre_batch[1][1], Y_dif_Pre_batch[2][1], Y_dif_Pre_batch[3][1], Y_dif_Pre_batch[4][1], Y_dif_Pre_batch[5][1], Y_dif_Pre_batch[6][1], Y_dif_Pre_batch[7][1], Y_dif_Pre_batch[8][1], Y_dif_Pre_batch[9][1], Y_dif_Pre_batch[10][1], Y_dif_Pre_batch[11][1],
             Y_temp_out[2], Y_dif_Pre_batch[0][2], Y_dif_Pre_batch[1][2], Y_dif_Pre_batch[2][2], Y_dif_Pre_batch[3][2], Y_dif_Pre_batch[4][2], Y_dif_Pre_batch[5][2], Y_dif_Pre_batch[6][2], Y_dif_Pre_batch[7][2], Y_dif_Pre_batch[8][2], Y_dif_Pre_batch[9][2], Y_dif_Pre_batch[10][2], Y_dif_Pre_batch[11][2],
             Y_temp_out[3], Y_dif_Pre_batch[0][3], Y_dif_Pre_batch[1][3], Y_dif_Pre_batch[2][3], Y_dif_Pre_batch[3][3], Y_dif_Pre_batch[4][3], Y_dif_Pre_batch[5][3], Y_dif_Pre_batch[6][3], Y_dif_Pre_batch[7][3], Y_dif_Pre_batch[8][3], Y_dif_Pre_batch[9][3], Y_dif_Pre_batch[10][3], Y_dif_Pre_batch[11][3],
             Y_temp_out[4], Y_dif_Pre_batch[0][4], Y_dif_Pre_batch[1][4], Y_dif_Pre_batch[2][4], Y_dif_Pre_batch[3][4], Y_dif_Pre_batch[4][4], Y_dif_Pre_batch[5][4], Y_dif_Pre_batch[6][4], Y_dif_Pre_batch[7][4], Y_dif_Pre_batch[8][4], Y_dif_Pre_batch[9][4], Y_dif_Pre_batch[10][4], Y_dif_Pre_batch[11][4],
             Y_temp_out[5], Y_dif_Pre_batch[0][5], Y_dif_Pre_batch[1][5], Y_dif_Pre_batch[2][5], Y_dif_Pre_batch[3][5], Y_dif_Pre_batch[4][5], Y_dif_Pre_batch[5][5], Y_dif_Pre_batch[6][5], Y_dif_Pre_batch[7][5], Y_dif_Pre_batch[8][5], Y_dif_Pre_batch[9][5], Y_dif_Pre_batch[10][5], Y_dif_Pre_batch[11][5],
             Y_temp_out[6], Y_dif_Pre_batch[0][6], Y_dif_Pre_batch[1][6], Y_dif_Pre_batch[2][6], Y_dif_Pre_batch[3][6], Y_dif_Pre_batch[4][6], Y_dif_Pre_batch[5][6], Y_dif_Pre_batch[6][6], Y_dif_Pre_batch[7][6], Y_dif_Pre_batch[8][6], Y_dif_Pre_batch[9][6], Y_dif_Pre_batch[10][6], Y_dif_Pre_batch[11][6],
             ], nrow=plot_row)
    elif plot_row == 14:
        grid = make_grid(
            [Y_temp_out[0], Y_dif_Pre_batch[0][0], Y_dif_Pre_batch[1][0], Y_dif_Pre_batch[2][0], Y_dif_Pre_batch[3][0], Y_dif_Pre_batch[4][0], Y_dif_Pre_batch[5][0], Y_dif_Pre_batch[6][0], Y_dif_Pre_batch[7][0], Y_dif_Pre_batch[8][0], Y_dif_Pre_batch[9][0], Y_dif_Pre_batch[10][0], Y_dif_Pre_batch[11][0], Y_dif_Pre_batch[12][0],
             Y_temp_out[1], Y_dif_Pre_batch[0][1], Y_dif_Pre_batch[1][1], Y_dif_Pre_batch[2][1], Y_dif_Pre_batch[3][1], Y_dif_Pre_batch[4][1], Y_dif_Pre_batch[5][1], Y_dif_Pre_batch[6][1], Y_dif_Pre_batch[7][1], Y_dif_Pre_batch[8][1], Y_dif_Pre_batch[9][1], Y_dif_Pre_batch[10][1], Y_dif_Pre_batch[11][1], Y_dif_Pre_batch[12][1],
             Y_temp_out[2], Y_dif_Pre_batch[0][2], Y_dif_Pre_batch[1][2], Y_dif_Pre_batch[2][2], Y_dif_Pre_batch[3][2], Y_dif_Pre_batch[4][2], Y_dif_Pre_batch[5][2], Y_dif_Pre_batch[6][2], Y_dif_Pre_batch[7][2], Y_dif_Pre_batch[8][2], Y_dif_Pre_batch[9][2], Y_dif_Pre_batch[10][2], Y_dif_Pre_batch[11][2], Y_dif_Pre_batch[12][2],
             Y_temp_out[3], Y_dif_Pre_batch[0][3], Y_dif_Pre_batch[1][3], Y_dif_Pre_batch[2][3], Y_dif_Pre_batch[3][3], Y_dif_Pre_batch[4][3], Y_dif_Pre_batch[5][3], Y_dif_Pre_batch[6][3], Y_dif_Pre_batch[7][3], Y_dif_Pre_batch[8][3], Y_dif_Pre_batch[9][3], Y_dif_Pre_batch[10][3], Y_dif_Pre_batch[11][3], Y_dif_Pre_batch[12][3],
             Y_temp_out[4], Y_dif_Pre_batch[0][4], Y_dif_Pre_batch[1][4], Y_dif_Pre_batch[2][4], Y_dif_Pre_batch[3][4], Y_dif_Pre_batch[4][4], Y_dif_Pre_batch[5][4], Y_dif_Pre_batch[6][4], Y_dif_Pre_batch[7][4], Y_dif_Pre_batch[8][4], Y_dif_Pre_batch[9][4], Y_dif_Pre_batch[10][4], Y_dif_Pre_batch[11][4], Y_dif_Pre_batch[12][4],
             Y_temp_out[5], Y_dif_Pre_batch[0][5], Y_dif_Pre_batch[1][5], Y_dif_Pre_batch[2][5], Y_dif_Pre_batch[3][5], Y_dif_Pre_batch[4][5], Y_dif_Pre_batch[5][5], Y_dif_Pre_batch[6][5], Y_dif_Pre_batch[7][5], Y_dif_Pre_batch[8][5], Y_dif_Pre_batch[9][5], Y_dif_Pre_batch[10][5], Y_dif_Pre_batch[11][5], Y_dif_Pre_batch[12][5],
             Y_temp_out[6], Y_dif_Pre_batch[0][6], Y_dif_Pre_batch[1][6], Y_dif_Pre_batch[2][6], Y_dif_Pre_batch[3][6], Y_dif_Pre_batch[4][6], Y_dif_Pre_batch[5][6], Y_dif_Pre_batch[6][6], Y_dif_Pre_batch[7][6], Y_dif_Pre_batch[8][6], Y_dif_Pre_batch[9][6], Y_dif_Pre_batch[10][6], Y_dif_Pre_batch[11][6], Y_dif_Pre_batch[12][6],
             ], nrow=plot_row)
    elif plot_row == 15:
        grid = make_grid(
            [Y_temp_out[0], Y_dif_Pre_batch[0][0], Y_dif_Pre_batch[1][0], Y_dif_Pre_batch[2][0], Y_dif_Pre_batch[3][0], Y_dif_Pre_batch[4][0], Y_dif_Pre_batch[5][0], Y_dif_Pre_batch[6][0], Y_dif_Pre_batch[7][0], Y_dif_Pre_batch[8][0], Y_dif_Pre_batch[9][0], Y_dif_Pre_batch[10][0], Y_dif_Pre_batch[11][0], Y_dif_Pre_batch[12][0], Y_dif_Pre_batch[13][0],
             Y_temp_out[1], Y_dif_Pre_batch[0][1], Y_dif_Pre_batch[1][1], Y_dif_Pre_batch[2][1], Y_dif_Pre_batch[3][1], Y_dif_Pre_batch[4][1], Y_dif_Pre_batch[5][1], Y_dif_Pre_batch[6][1], Y_dif_Pre_batch[7][1], Y_dif_Pre_batch[8][1], Y_dif_Pre_batch[9][1], Y_dif_Pre_batch[10][1], Y_dif_Pre_batch[11][1], Y_dif_Pre_batch[12][1], Y_dif_Pre_batch[13][1],
             Y_temp_out[2], Y_dif_Pre_batch[0][2], Y_dif_Pre_batch[1][2], Y_dif_Pre_batch[2][2], Y_dif_Pre_batch[3][2], Y_dif_Pre_batch[4][2], Y_dif_Pre_batch[5][2], Y_dif_Pre_batch[6][2], Y_dif_Pre_batch[7][2], Y_dif_Pre_batch[8][2], Y_dif_Pre_batch[9][2], Y_dif_Pre_batch[10][2], Y_dif_Pre_batch[11][2], Y_dif_Pre_batch[12][2], Y_dif_Pre_batch[13][2],
             Y_temp_out[3], Y_dif_Pre_batch[0][3], Y_dif_Pre_batch[1][3], Y_dif_Pre_batch[2][3], Y_dif_Pre_batch[3][3], Y_dif_Pre_batch[4][3], Y_dif_Pre_batch[5][3], Y_dif_Pre_batch[6][3], Y_dif_Pre_batch[7][3], Y_dif_Pre_batch[8][3], Y_dif_Pre_batch[9][3], Y_dif_Pre_batch[10][3], Y_dif_Pre_batch[11][3], Y_dif_Pre_batch[12][3], Y_dif_Pre_batch[13][3],
             Y_temp_out[4], Y_dif_Pre_batch[0][4], Y_dif_Pre_batch[1][4], Y_dif_Pre_batch[2][4], Y_dif_Pre_batch[3][4], Y_dif_Pre_batch[4][4], Y_dif_Pre_batch[5][4], Y_dif_Pre_batch[6][4], Y_dif_Pre_batch[7][4], Y_dif_Pre_batch[8][4], Y_dif_Pre_batch[9][4], Y_dif_Pre_batch[10][4], Y_dif_Pre_batch[11][4], Y_dif_Pre_batch[12][4], Y_dif_Pre_batch[13][4],
             Y_temp_out[5], Y_dif_Pre_batch[0][5], Y_dif_Pre_batch[1][5], Y_dif_Pre_batch[2][5], Y_dif_Pre_batch[3][5], Y_dif_Pre_batch[4][5], Y_dif_Pre_batch[5][5], Y_dif_Pre_batch[6][5], Y_dif_Pre_batch[7][5], Y_dif_Pre_batch[8][5], Y_dif_Pre_batch[9][5], Y_dif_Pre_batch[10][5], Y_dif_Pre_batch[11][5], Y_dif_Pre_batch[12][5], Y_dif_Pre_batch[13][5],
             Y_temp_out[6], Y_dif_Pre_batch[0][6], Y_dif_Pre_batch[1][6], Y_dif_Pre_batch[2][6], Y_dif_Pre_batch[3][6], Y_dif_Pre_batch[4][6], Y_dif_Pre_batch[5][6], Y_dif_Pre_batch[6][6], Y_dif_Pre_batch[7][6], Y_dif_Pre_batch[8][6], Y_dif_Pre_batch[9][6], Y_dif_Pre_batch[10][6], Y_dif_Pre_batch[11][6], Y_dif_Pre_batch[12][6], Y_dif_Pre_batch[13][6],
             ], nrow=plot_row)
    elif plot_row == 16:
        grid = make_grid(
            [Y_temp_out[0], Y_dif_Pre_batch[0][0], Y_dif_Pre_batch[1][0], Y_dif_Pre_batch[2][0], Y_dif_Pre_batch[3][0], Y_dif_Pre_batch[4][0], Y_dif_Pre_batch[5][0], Y_dif_Pre_batch[6][0], Y_dif_Pre_batch[7][0], Y_dif_Pre_batch[8][0], Y_dif_Pre_batch[9][0], Y_dif_Pre_batch[10][0], Y_dif_Pre_batch[11][0], Y_dif_Pre_batch[12][0], Y_dif_Pre_batch[13][0], Y_dif_Pre_batch[14][0],
             Y_temp_out[1], Y_dif_Pre_batch[0][1], Y_dif_Pre_batch[1][1], Y_dif_Pre_batch[2][1], Y_dif_Pre_batch[3][1], Y_dif_Pre_batch[4][1], Y_dif_Pre_batch[5][1], Y_dif_Pre_batch[6][1], Y_dif_Pre_batch[7][1], Y_dif_Pre_batch[8][1], Y_dif_Pre_batch[9][1], Y_dif_Pre_batch[10][1], Y_dif_Pre_batch[11][1], Y_dif_Pre_batch[12][1], Y_dif_Pre_batch[13][1], Y_dif_Pre_batch[14][1],
             Y_temp_out[2], Y_dif_Pre_batch[0][2], Y_dif_Pre_batch[1][2], Y_dif_Pre_batch[2][2], Y_dif_Pre_batch[3][2], Y_dif_Pre_batch[4][2], Y_dif_Pre_batch[5][2], Y_dif_Pre_batch[6][2], Y_dif_Pre_batch[7][2], Y_dif_Pre_batch[8][2], Y_dif_Pre_batch[9][2], Y_dif_Pre_batch[10][2], Y_dif_Pre_batch[11][2], Y_dif_Pre_batch[12][2], Y_dif_Pre_batch[13][2], Y_dif_Pre_batch[14][2],
             Y_temp_out[3], Y_dif_Pre_batch[0][3], Y_dif_Pre_batch[1][3], Y_dif_Pre_batch[2][3], Y_dif_Pre_batch[3][3], Y_dif_Pre_batch[4][3], Y_dif_Pre_batch[5][3], Y_dif_Pre_batch[6][3], Y_dif_Pre_batch[7][3], Y_dif_Pre_batch[8][3], Y_dif_Pre_batch[9][3], Y_dif_Pre_batch[10][3], Y_dif_Pre_batch[11][3], Y_dif_Pre_batch[12][3], Y_dif_Pre_batch[13][3], Y_dif_Pre_batch[14][3],
             Y_temp_out[4], Y_dif_Pre_batch[0][4], Y_dif_Pre_batch[1][4], Y_dif_Pre_batch[2][4], Y_dif_Pre_batch[3][4], Y_dif_Pre_batch[4][4], Y_dif_Pre_batch[5][4], Y_dif_Pre_batch[6][4], Y_dif_Pre_batch[7][4], Y_dif_Pre_batch[8][4], Y_dif_Pre_batch[9][4], Y_dif_Pre_batch[10][4], Y_dif_Pre_batch[11][4], Y_dif_Pre_batch[12][4], Y_dif_Pre_batch[13][4], Y_dif_Pre_batch[14][4],
             Y_temp_out[5], Y_dif_Pre_batch[0][5], Y_dif_Pre_batch[1][5], Y_dif_Pre_batch[2][5], Y_dif_Pre_batch[3][5], Y_dif_Pre_batch[4][5], Y_dif_Pre_batch[5][5], Y_dif_Pre_batch[6][5], Y_dif_Pre_batch[7][5], Y_dif_Pre_batch[8][5], Y_dif_Pre_batch[9][5], Y_dif_Pre_batch[10][5], Y_dif_Pre_batch[11][5], Y_dif_Pre_batch[12][5], Y_dif_Pre_batch[13][5], Y_dif_Pre_batch[14][5],
             Y_temp_out[6], Y_dif_Pre_batch[0][6], Y_dif_Pre_batch[1][6], Y_dif_Pre_batch[2][6], Y_dif_Pre_batch[3][6], Y_dif_Pre_batch[4][6], Y_dif_Pre_batch[5][6], Y_dif_Pre_batch[6][6], Y_dif_Pre_batch[7][6], Y_dif_Pre_batch[8][6], Y_dif_Pre_batch[9][6], Y_dif_Pre_batch[10][6], Y_dif_Pre_batch[11][6], Y_dif_Pre_batch[12][6], Y_dif_Pre_batch[13][6], Y_dif_Pre_batch[14][6],
             ], nrow=plot_row)
    elif plot_row == 17:
        grid = make_grid(
            [Y_temp_out[0], Y_dif_Pre_batch[0][0], Y_dif_Pre_batch[1][0], Y_dif_Pre_batch[2][0], Y_dif_Pre_batch[3][0], Y_dif_Pre_batch[4][0], Y_dif_Pre_batch[5][0], Y_dif_Pre_batch[6][0], Y_dif_Pre_batch[7][0], Y_dif_Pre_batch[8][0], Y_dif_Pre_batch[9][0], Y_dif_Pre_batch[10][0], Y_dif_Pre_batch[11][0], Y_dif_Pre_batch[12][0], Y_dif_Pre_batch[13][0], Y_dif_Pre_batch[14][0], Y_dif_Pre_batch[15][0],
             Y_temp_out[1], Y_dif_Pre_batch[0][1], Y_dif_Pre_batch[1][1], Y_dif_Pre_batch[2][1], Y_dif_Pre_batch[3][1], Y_dif_Pre_batch[4][1], Y_dif_Pre_batch[5][1], Y_dif_Pre_batch[6][1], Y_dif_Pre_batch[7][1], Y_dif_Pre_batch[8][1], Y_dif_Pre_batch[9][1], Y_dif_Pre_batch[10][1], Y_dif_Pre_batch[11][1], Y_dif_Pre_batch[12][1], Y_dif_Pre_batch[13][1], Y_dif_Pre_batch[14][1], Y_dif_Pre_batch[15][1],
             Y_temp_out[2], Y_dif_Pre_batch[0][2], Y_dif_Pre_batch[1][2], Y_dif_Pre_batch[2][2], Y_dif_Pre_batch[3][2], Y_dif_Pre_batch[4][2], Y_dif_Pre_batch[5][2], Y_dif_Pre_batch[6][2], Y_dif_Pre_batch[7][2], Y_dif_Pre_batch[8][2], Y_dif_Pre_batch[9][2], Y_dif_Pre_batch[10][2], Y_dif_Pre_batch[11][2], Y_dif_Pre_batch[12][2], Y_dif_Pre_batch[13][2], Y_dif_Pre_batch[14][2], Y_dif_Pre_batch[15][2],
             Y_temp_out[3], Y_dif_Pre_batch[0][3], Y_dif_Pre_batch[1][3], Y_dif_Pre_batch[2][3], Y_dif_Pre_batch[3][3], Y_dif_Pre_batch[4][3], Y_dif_Pre_batch[5][3], Y_dif_Pre_batch[6][3], Y_dif_Pre_batch[7][3], Y_dif_Pre_batch[8][3], Y_dif_Pre_batch[9][3], Y_dif_Pre_batch[10][3], Y_dif_Pre_batch[11][3], Y_dif_Pre_batch[12][3], Y_dif_Pre_batch[13][3], Y_dif_Pre_batch[14][3], Y_dif_Pre_batch[15][3],
             Y_temp_out[4], Y_dif_Pre_batch[0][4], Y_dif_Pre_batch[1][4], Y_dif_Pre_batch[2][4], Y_dif_Pre_batch[3][4], Y_dif_Pre_batch[4][4], Y_dif_Pre_batch[5][4], Y_dif_Pre_batch[6][4], Y_dif_Pre_batch[7][4], Y_dif_Pre_batch[8][4], Y_dif_Pre_batch[9][4], Y_dif_Pre_batch[10][4], Y_dif_Pre_batch[11][4], Y_dif_Pre_batch[12][4], Y_dif_Pre_batch[13][4], Y_dif_Pre_batch[14][4], Y_dif_Pre_batch[15][4],
             Y_temp_out[5], Y_dif_Pre_batch[0][5], Y_dif_Pre_batch[1][5], Y_dif_Pre_batch[2][5], Y_dif_Pre_batch[3][5], Y_dif_Pre_batch[4][5], Y_dif_Pre_batch[5][5], Y_dif_Pre_batch[6][5], Y_dif_Pre_batch[7][5], Y_dif_Pre_batch[8][5], Y_dif_Pre_batch[9][5], Y_dif_Pre_batch[10][5], Y_dif_Pre_batch[11][5], Y_dif_Pre_batch[12][5], Y_dif_Pre_batch[13][5], Y_dif_Pre_batch[14][5], Y_dif_Pre_batch[15][5],
             Y_temp_out[6], Y_dif_Pre_batch[0][6], Y_dif_Pre_batch[1][6], Y_dif_Pre_batch[2][6], Y_dif_Pre_batch[3][6], Y_dif_Pre_batch[4][6], Y_dif_Pre_batch[5][6], Y_dif_Pre_batch[6][6], Y_dif_Pre_batch[7][6], Y_dif_Pre_batch[8][6], Y_dif_Pre_batch[9][6], Y_dif_Pre_batch[10][6], Y_dif_Pre_batch[11][6], Y_dif_Pre_batch[12][6], Y_dif_Pre_batch[13][6], Y_dif_Pre_batch[14][6], Y_dif_Pre_batch[15][6],
             ], nrow=plot_row)

    if train == True:
        save_image(grid, "result_reconstruction/image_train_e{}_grid_pre.png".format(epoch))
    else:
        save_image(grid, "result_reconstruction/image_test_e{}_grid_pre.png".format(epoch))

def save_test_pre_image(Y_Pre_batch, Y_batch, epoch):
    if epoch % 1 == 0:
        trans = torchvision.transforms.Resize(size=112)
        Y_Pre_out = trans(Y_Pre_batch[0])
        Y_out = trans(Y_batch[0])
        save_image(Y_Pre_out, "result_reconstruction/image_test_{}_pre.png".format((epoch - 1)))
        save_image(Y_out, "result_reconstruction/image_test_{}.png".format((epoch - 1)))

def save_test_dif_pre_image(Y_dif_Pre_batch, Y_batch, epoch):
        trans = torchvision.transforms.Resize(size=112)
        Y_out = trans(Y_batch[0])
        save_image(Y_out, "result_reconstruction/image_test_e{}.png".format((epoch - 1)))
        for n in range(0, len(Y_dif_Pre_batch)):
            Y_Pre_out = trans(Y_dif_Pre_batch[n][0])
            save_image(Y_Pre_out, "result_reconstruction/image_test_e{}_l{}_pre.png".format((epoch - 1), n))

    
def save_test_dif_pre_image_order(Y_dif_Pre_batch_input, Y_batch, epoch, cf, train):
    plot_row = len(cf.nodes)-1
    Y_temp_out = Y_batch
    Y_dif_Pre_batch = Y_dif_Pre_batch_input
    # make grid
    if plot_row == 5:
        grid = make_grid(
            [Y_temp_out[0], Y_dif_Pre_batch[0][0], Y_dif_Pre_batch[1][0], Y_dif_Pre_batch[2][0], Y_dif_Pre_batch[3][0],
             Y_temp_out[1], Y_dif_Pre_batch[0][1], Y_dif_Pre_batch[1][1], Y_dif_Pre_batch[2][1], Y_dif_Pre_batch[3][1],
             Y_temp_out[2], Y_dif_Pre_batch[0][2], Y_dif_Pre_batch[1][2], Y_dif_Pre_batch[2][2], Y_dif_Pre_batch[3][2],
             Y_temp_out[3], Y_dif_Pre_batch[0][3], Y_dif_Pre_batch[1][3], Y_dif_Pre_batch[2][3], Y_dif_Pre_batch[3][3],
             Y_temp_out[4], Y_dif_Pre_batch[0][4], Y_dif_Pre_batch[1][4], Y_dif_Pre_batch[2][4], Y_dif_Pre_batch[3][4],
             Y_temp_out[5], Y_dif_Pre_batch[0][5], Y_dif_Pre_batch[1][5], Y_dif_Pre_batch[2][5], Y_dif_Pre_batch[3][5],
             Y_temp_out[6], Y_dif_Pre_batch[0][6], Y_dif_Pre_batch[1][6], Y_dif_Pre_batch[2][6], Y_dif_Pre_batch[3][6],
             Y_temp_out[7], Y_dif_Pre_batch[0][7], Y_dif_Pre_batch[1][7], Y_dif_Pre_batch[2][7], Y_dif_Pre_batch[3][7],
             Y_temp_out[8], Y_dif_Pre_batch[0][8], Y_dif_Pre_batch[1][8], Y_dif_Pre_batch[2][8], Y_dif_Pre_batch[3][8],
             Y_temp_out[9], Y_dif_Pre_batch[0][9], Y_dif_Pre_batch[1][9], Y_dif_Pre_batch[2][9], Y_dif_Pre_batch[3][9],
             ], nrow=plot_row)
    elif plot_row == 6:
        grid = make_grid(
            [Y_temp_out[0], Y_dif_Pre_batch[0][0], Y_dif_Pre_batch[1][0], Y_dif_Pre_batch[2][0], Y_dif_Pre_batch[3][0], Y_dif_Pre_batch[4][0],
             Y_temp_out[1], Y_dif_Pre_batch[0][1], Y_dif_Pre_batch[1][1], Y_dif_Pre_batch[2][1], Y_dif_Pre_batch[3][1], Y_dif_Pre_batch[4][1],
             Y_temp_out[2], Y_dif_Pre_batch[0][2], Y_dif_Pre_batch[1][2], Y_dif_Pre_batch[2][2], Y_dif_Pre_batch[3][2], Y_dif_Pre_batch[4][2],
             Y_temp_out[3], Y_dif_Pre_batch[0][3], Y_dif_Pre_batch[1][3], Y_dif_Pre_batch[2][3], Y_dif_Pre_batch[3][3], Y_dif_Pre_batch[4][3],
             Y_temp_out[4], Y_dif_Pre_batch[0][4], Y_dif_Pre_batch[1][4], Y_dif_Pre_batch[2][4], Y_dif_Pre_batch[3][4], Y_dif_Pre_batch[4][4],
             Y_temp_out[5], Y_dif_Pre_batch[0][5], Y_dif_Pre_batch[1][5], Y_dif_Pre_batch[2][5], Y_dif_Pre_batch[3][5], Y_dif_Pre_batch[4][5],
             Y_temp_out[6], Y_dif_Pre_batch[0][6], Y_dif_Pre_batch[1][6], Y_dif_Pre_batch[2][6], Y_dif_Pre_batch[3][6], Y_dif_Pre_batch[4][6],
             Y_temp_out[7], Y_dif_Pre_batch[0][7], Y_dif_Pre_batch[1][7], Y_dif_Pre_batch[2][7], Y_dif_Pre_batch[3][7], Y_dif_Pre_batch[4][7],
             Y_temp_out[8], Y_dif_Pre_batch[0][8], Y_dif_Pre_batch[1][8], Y_dif_Pre_batch[2][8], Y_dif_Pre_batch[3][8], Y_dif_Pre_batch[4][8],
             Y_temp_out[9], Y_dif_Pre_batch[0][9], Y_dif_Pre_batch[1][9], Y_dif_Pre_batch[2][9], Y_dif_Pre_batch[3][9], Y_dif_Pre_batch[4][9],
             ], nrow=plot_row)
    elif plot_row == 7:
        grid = make_grid(
            [Y_temp_out[0], Y_dif_Pre_batch[0][0], Y_dif_Pre_batch[1][0], Y_dif_Pre_batch[2][0], Y_dif_Pre_batch[3][0], Y_dif_Pre_batch[4][0], Y_dif_Pre_batch[5][0],
             Y_temp_out[1], Y_dif_Pre_batch[0][1], Y_dif_Pre_batch[1][1], Y_dif_Pre_batch[2][1], Y_dif_Pre_batch[3][1], Y_dif_Pre_batch[4][1], Y_dif_Pre_batch[5][1],
             Y_temp_out[2], Y_dif_Pre_batch[0][2], Y_dif_Pre_batch[1][2], Y_dif_Pre_batch[2][2], Y_dif_Pre_batch[3][2], Y_dif_Pre_batch[4][2], Y_dif_Pre_batch[5][2],
             Y_temp_out[3], Y_dif_Pre_batch[0][3], Y_dif_Pre_batch[1][3], Y_dif_Pre_batch[2][3], Y_dif_Pre_batch[3][3], Y_dif_Pre_batch[4][3], Y_dif_Pre_batch[5][3],
             Y_temp_out[4], Y_dif_Pre_batch[0][4], Y_dif_Pre_batch[1][4], Y_dif_Pre_batch[2][4], Y_dif_Pre_batch[3][4], Y_dif_Pre_batch[4][4], Y_dif_Pre_batch[5][4],
             Y_temp_out[5], Y_dif_Pre_batch[0][5], Y_dif_Pre_batch[1][5], Y_dif_Pre_batch[2][5], Y_dif_Pre_batch[3][5], Y_dif_Pre_batch[4][5], Y_dif_Pre_batch[5][5],
             Y_temp_out[6], Y_dif_Pre_batch[0][6], Y_dif_Pre_batch[1][6], Y_dif_Pre_batch[2][6], Y_dif_Pre_batch[3][6], Y_dif_Pre_batch[4][6], Y_dif_Pre_batch[5][6],
             Y_temp_out[7], Y_dif_Pre_batch[0][7], Y_dif_Pre_batch[1][7], Y_dif_Pre_batch[2][7], Y_dif_Pre_batch[3][7], Y_dif_Pre_batch[4][7], Y_dif_Pre_batch[5][7],
             Y_temp_out[8], Y_dif_Pre_batch[0][8], Y_dif_Pre_batch[1][8], Y_dif_Pre_batch[2][8], Y_dif_Pre_batch[3][8], Y_dif_Pre_batch[4][8], Y_dif_Pre_batch[5][8],
             Y_temp_out[9], Y_dif_Pre_batch[0][9], Y_dif_Pre_batch[1][9], Y_dif_Pre_batch[2][9], Y_dif_Pre_batch[3][9], Y_dif_Pre_batch[4][9], Y_dif_Pre_batch[5][9],
             ], nrow=plot_row)
    elif plot_row == 8:
        grid = make_grid(
            [Y_temp_out[0], Y_dif_Pre_batch[0][0], Y_dif_Pre_batch[1][0], Y_dif_Pre_batch[2][0], Y_dif_Pre_batch[3][0], Y_dif_Pre_batch[4][0], Y_dif_Pre_batch[5][0], Y_dif_Pre_batch[6][0],
             Y_temp_out[1], Y_dif_Pre_batch[0][1], Y_dif_Pre_batch[1][1], Y_dif_Pre_batch[2][1], Y_dif_Pre_batch[3][1], Y_dif_Pre_batch[4][1], Y_dif_Pre_batch[5][1], Y_dif_Pre_batch[6][1],
             Y_temp_out[2], Y_dif_Pre_batch[0][2], Y_dif_Pre_batch[1][2], Y_dif_Pre_batch[2][2], Y_dif_Pre_batch[3][2], Y_dif_Pre_batch[4][2], Y_dif_Pre_batch[5][2], Y_dif_Pre_batch[6][2],
             Y_temp_out[3], Y_dif_Pre_batch[0][3], Y_dif_Pre_batch[1][3], Y_dif_Pre_batch[2][3], Y_dif_Pre_batch[3][3], Y_dif_Pre_batch[4][3], Y_dif_Pre_batch[5][3], Y_dif_Pre_batch[6][3],
             Y_temp_out[4], Y_dif_Pre_batch[0][4], Y_dif_Pre_batch[1][4], Y_dif_Pre_batch[2][4], Y_dif_Pre_batch[3][4], Y_dif_Pre_batch[4][4], Y_dif_Pre_batch[5][4], Y_dif_Pre_batch[6][4],
             Y_temp_out[5], Y_dif_Pre_batch[0][5], Y_dif_Pre_batch[1][5], Y_dif_Pre_batch[2][5], Y_dif_Pre_batch[3][5], Y_dif_Pre_batch[4][5], Y_dif_Pre_batch[5][5], Y_dif_Pre_batch[6][5],
             Y_temp_out[6], Y_dif_Pre_batch[0][6], Y_dif_Pre_batch[1][6], Y_dif_Pre_batch[2][6], Y_dif_Pre_batch[3][6], Y_dif_Pre_batch[4][6], Y_dif_Pre_batch[5][6], Y_dif_Pre_batch[6][6],
             Y_temp_out[7], Y_dif_Pre_batch[0][7], Y_dif_Pre_batch[1][7], Y_dif_Pre_batch[2][7], Y_dif_Pre_batch[3][7], Y_dif_Pre_batch[4][7], Y_dif_Pre_batch[5][7], Y_dif_Pre_batch[6][7],
             Y_temp_out[8], Y_dif_Pre_batch[0][8], Y_dif_Pre_batch[1][8], Y_dif_Pre_batch[2][8], Y_dif_Pre_batch[3][8], Y_dif_Pre_batch[4][8], Y_dif_Pre_batch[5][8], Y_dif_Pre_batch[6][8],
             Y_temp_out[9], Y_dif_Pre_batch[0][9], Y_dif_Pre_batch[1][9], Y_dif_Pre_batch[2][9], Y_dif_Pre_batch[3][9], Y_dif_Pre_batch[4][9], Y_dif_Pre_batch[5][9], Y_dif_Pre_batch[6][9],
             ], nrow=plot_row)
    elif plot_row == 9:
        grid = make_grid(
            [Y_temp_out[0], Y_dif_Pre_batch[0][0], Y_dif_Pre_batch[1][0], Y_dif_Pre_batch[2][0], Y_dif_Pre_batch[3][0], Y_dif_Pre_batch[4][0], Y_dif_Pre_batch[5][0], Y_dif_Pre_batch[6][0], Y_dif_Pre_batch[7][0],
             Y_temp_out[1], Y_dif_Pre_batch[0][1], Y_dif_Pre_batch[1][1], Y_dif_Pre_batch[2][1], Y_dif_Pre_batch[3][1], Y_dif_Pre_batch[4][1], Y_dif_Pre_batch[5][1], Y_dif_Pre_batch[6][1], Y_dif_Pre_batch[7][1],
             Y_temp_out[2], Y_dif_Pre_batch[0][2], Y_dif_Pre_batch[1][2], Y_dif_Pre_batch[2][2], Y_dif_Pre_batch[3][2], Y_dif_Pre_batch[4][2], Y_dif_Pre_batch[5][2], Y_dif_Pre_batch[6][2], Y_dif_Pre_batch[7][2],
             Y_temp_out[3], Y_dif_Pre_batch[0][3], Y_dif_Pre_batch[1][3], Y_dif_Pre_batch[2][3], Y_dif_Pre_batch[3][3], Y_dif_Pre_batch[4][3], Y_dif_Pre_batch[5][3], Y_dif_Pre_batch[6][3], Y_dif_Pre_batch[7][3],
             Y_temp_out[4], Y_dif_Pre_batch[0][4], Y_dif_Pre_batch[1][4], Y_dif_Pre_batch[2][4], Y_dif_Pre_batch[3][4], Y_dif_Pre_batch[4][4], Y_dif_Pre_batch[5][4], Y_dif_Pre_batch[6][4], Y_dif_Pre_batch[7][4],
             ], nrow=plot_row)
    elif plot_row == 10:
        grid = make_grid(
            [Y_temp_out[0], Y_dif_Pre_batch[0][0], Y_dif_Pre_batch[1][0], Y_dif_Pre_batch[2][0], Y_dif_Pre_batch[3][0], Y_dif_Pre_batch[4][0], Y_dif_Pre_batch[5][0], Y_dif_Pre_batch[6][0], Y_dif_Pre_batch[7][0], Y_dif_Pre_batch[8][0],
             Y_temp_out[1], Y_dif_Pre_batch[0][1], Y_dif_Pre_batch[1][1], Y_dif_Pre_batch[2][1], Y_dif_Pre_batch[3][1], Y_dif_Pre_batch[4][1], Y_dif_Pre_batch[5][1], Y_dif_Pre_batch[6][1], Y_dif_Pre_batch[7][1], Y_dif_Pre_batch[8][1],
             Y_temp_out[2], Y_dif_Pre_batch[0][2], Y_dif_Pre_batch[1][2], Y_dif_Pre_batch[2][2], Y_dif_Pre_batch[3][2], Y_dif_Pre_batch[4][2], Y_dif_Pre_batch[5][2], Y_dif_Pre_batch[6][2], Y_dif_Pre_batch[7][2], Y_dif_Pre_batch[8][2],
             Y_temp_out[3], Y_dif_Pre_batch[0][3], Y_dif_Pre_batch[1][3], Y_dif_Pre_batch[2][3], Y_dif_Pre_batch[3][3], Y_dif_Pre_batch[4][3], Y_dif_Pre_batch[5][3], Y_dif_Pre_batch[6][3], Y_dif_Pre_batch[7][3], Y_dif_Pre_batch[8][3],
             Y_temp_out[4], Y_dif_Pre_batch[0][4], Y_dif_Pre_batch[1][4], Y_dif_Pre_batch[2][4], Y_dif_Pre_batch[3][4], Y_dif_Pre_batch[4][4], Y_dif_Pre_batch[5][4], Y_dif_Pre_batch[6][4], Y_dif_Pre_batch[7][4], Y_dif_Pre_batch[8][4],
             ], nrow=plot_row)
    elif plot_row == 11:
        grid = make_grid(
            [Y_temp_out[0], Y_dif_Pre_batch[0][0], Y_dif_Pre_batch[1][0], Y_dif_Pre_batch[2][0], Y_dif_Pre_batch[3][0], Y_dif_Pre_batch[4][0], Y_dif_Pre_batch[5][0], Y_dif_Pre_batch[6][0], Y_dif_Pre_batch[7][0], Y_dif_Pre_batch[8][0], Y_dif_Pre_batch[9][0],
             Y_temp_out[1], Y_dif_Pre_batch[0][1], Y_dif_Pre_batch[1][1], Y_dif_Pre_batch[2][1], Y_dif_Pre_batch[3][1], Y_dif_Pre_batch[4][1], Y_dif_Pre_batch[5][1], Y_dif_Pre_batch[6][1], Y_dif_Pre_batch[7][1], Y_dif_Pre_batch[8][1], Y_dif_Pre_batch[9][1],
             Y_temp_out[2], Y_dif_Pre_batch[0][2], Y_dif_Pre_batch[1][2], Y_dif_Pre_batch[2][2], Y_dif_Pre_batch[3][2], Y_dif_Pre_batch[4][2], Y_dif_Pre_batch[5][2], Y_dif_Pre_batch[6][2], Y_dif_Pre_batch[7][2], Y_dif_Pre_batch[8][2], Y_dif_Pre_batch[9][2],
             Y_temp_out[3], Y_dif_Pre_batch[0][3], Y_dif_Pre_batch[1][3], Y_dif_Pre_batch[2][3], Y_dif_Pre_batch[3][3], Y_dif_Pre_batch[4][3], Y_dif_Pre_batch[5][3], Y_dif_Pre_batch[6][3], Y_dif_Pre_batch[7][3], Y_dif_Pre_batch[8][3], Y_dif_Pre_batch[9][3],
             Y_temp_out[4], Y_dif_Pre_batch[0][4], Y_dif_Pre_batch[1][4], Y_dif_Pre_batch[2][4], Y_dif_Pre_batch[3][4], Y_dif_Pre_batch[4][4], Y_dif_Pre_batch[5][4], Y_dif_Pre_batch[6][4], Y_dif_Pre_batch[7][4], Y_dif_Pre_batch[8][4], Y_dif_Pre_batch[9][4],
             Y_temp_out[5], Y_dif_Pre_batch[0][5], Y_dif_Pre_batch[1][5], Y_dif_Pre_batch[2][5], Y_dif_Pre_batch[3][5], Y_dif_Pre_batch[4][5], Y_dif_Pre_batch[5][5], Y_dif_Pre_batch[6][5], Y_dif_Pre_batch[7][5], Y_dif_Pre_batch[8][5], Y_dif_Pre_batch[9][5],
             Y_temp_out[6], Y_dif_Pre_batch[0][6], Y_dif_Pre_batch[1][6], Y_dif_Pre_batch[2][6], Y_dif_Pre_batch[3][6], Y_dif_Pre_batch[4][6], Y_dif_Pre_batch[5][6], Y_dif_Pre_batch[6][6], Y_dif_Pre_batch[7][6], Y_dif_Pre_batch[8][6], Y_dif_Pre_batch[9][6],
             ], nrow=plot_row)
    elif plot_row == 12:
        grid = make_grid(
            [Y_temp_out[0], Y_dif_Pre_batch[0][0], Y_dif_Pre_batch[1][0], Y_dif_Pre_batch[2][0], Y_dif_Pre_batch[3][0], Y_dif_Pre_batch[4][0], Y_dif_Pre_batch[5][0], Y_dif_Pre_batch[6][0], Y_dif_Pre_batch[7][0], Y_dif_Pre_batch[8][0], Y_dif_Pre_batch[9][0], Y_dif_Pre_batch[10][0],
             Y_temp_out[1], Y_dif_Pre_batch[0][1], Y_dif_Pre_batch[1][1], Y_dif_Pre_batch[2][1], Y_dif_Pre_batch[3][1], Y_dif_Pre_batch[4][1], Y_dif_Pre_batch[5][1], Y_dif_Pre_batch[6][1], Y_dif_Pre_batch[7][1], Y_dif_Pre_batch[8][1], Y_dif_Pre_batch[9][1], Y_dif_Pre_batch[10][1],
             Y_temp_out[2], Y_dif_Pre_batch[0][2], Y_dif_Pre_batch[1][2], Y_dif_Pre_batch[2][2], Y_dif_Pre_batch[3][2], Y_dif_Pre_batch[4][2], Y_dif_Pre_batch[5][2], Y_dif_Pre_batch[6][2], Y_dif_Pre_batch[7][2], Y_dif_Pre_batch[8][2], Y_dif_Pre_batch[9][2], Y_dif_Pre_batch[10][2],
             Y_temp_out[3], Y_dif_Pre_batch[0][3], Y_dif_Pre_batch[1][3], Y_dif_Pre_batch[2][3], Y_dif_Pre_batch[3][3], Y_dif_Pre_batch[4][3], Y_dif_Pre_batch[5][3], Y_dif_Pre_batch[6][3], Y_dif_Pre_batch[7][3], Y_dif_Pre_batch[8][3], Y_dif_Pre_batch[9][3], Y_dif_Pre_batch[10][3],
             Y_temp_out[4], Y_dif_Pre_batch[0][4], Y_dif_Pre_batch[1][4], Y_dif_Pre_batch[2][4], Y_dif_Pre_batch[3][4], Y_dif_Pre_batch[4][4], Y_dif_Pre_batch[5][4], Y_dif_Pre_batch[6][4], Y_dif_Pre_batch[7][4], Y_dif_Pre_batch[8][4], Y_dif_Pre_batch[9][4], Y_dif_Pre_batch[10][4],
             Y_temp_out[5], Y_dif_Pre_batch[0][5], Y_dif_Pre_batch[1][5], Y_dif_Pre_batch[2][5], Y_dif_Pre_batch[3][5], Y_dif_Pre_batch[4][5], Y_dif_Pre_batch[5][5], Y_dif_Pre_batch[6][5], Y_dif_Pre_batch[7][5], Y_dif_Pre_batch[8][5], Y_dif_Pre_batch[9][5], Y_dif_Pre_batch[10][5],
             Y_temp_out[6], Y_dif_Pre_batch[0][6], Y_dif_Pre_batch[1][6], Y_dif_Pre_batch[2][6], Y_dif_Pre_batch[3][6], Y_dif_Pre_batch[4][6], Y_dif_Pre_batch[5][6], Y_dif_Pre_batch[6][6], Y_dif_Pre_batch[7][6], Y_dif_Pre_batch[8][6], Y_dif_Pre_batch[9][6], Y_dif_Pre_batch[10][6],
             ], nrow=plot_row)
    elif plot_row == 13:
        grid = make_grid(
            [Y_temp_out[0], Y_dif_Pre_batch[0][0], Y_dif_Pre_batch[1][0], Y_dif_Pre_batch[2][0], Y_dif_Pre_batch[3][0], Y_dif_Pre_batch[4][0], Y_dif_Pre_batch[5][0], Y_dif_Pre_batch[6][0], Y_dif_Pre_batch[7][0], Y_dif_Pre_batch[8][0], Y_dif_Pre_batch[9][0], Y_dif_Pre_batch[10][0], Y_dif_Pre_batch[11][0],
             Y_temp_out[1], Y_dif_Pre_batch[0][1], Y_dif_Pre_batch[1][1], Y_dif_Pre_batch[2][1], Y_dif_Pre_batch[3][1], Y_dif_Pre_batch[4][1], Y_dif_Pre_batch[5][1], Y_dif_Pre_batch[6][1], Y_dif_Pre_batch[7][1], Y_dif_Pre_batch[8][1], Y_dif_Pre_batch[9][1], Y_dif_Pre_batch[10][1], Y_dif_Pre_batch[11][1],
             Y_temp_out[2], Y_dif_Pre_batch[0][2], Y_dif_Pre_batch[1][2], Y_dif_Pre_batch[2][2], Y_dif_Pre_batch[3][2], Y_dif_Pre_batch[4][2], Y_dif_Pre_batch[5][2], Y_dif_Pre_batch[6][2], Y_dif_Pre_batch[7][2], Y_dif_Pre_batch[8][2], Y_dif_Pre_batch[9][2], Y_dif_Pre_batch[10][2], Y_dif_Pre_batch[11][2],
             Y_temp_out[3], Y_dif_Pre_batch[0][3], Y_dif_Pre_batch[1][3], Y_dif_Pre_batch[2][3], Y_dif_Pre_batch[3][3], Y_dif_Pre_batch[4][3], Y_dif_Pre_batch[5][3], Y_dif_Pre_batch[6][3], Y_dif_Pre_batch[7][3], Y_dif_Pre_batch[8][3], Y_dif_Pre_batch[9][3], Y_dif_Pre_batch[10][3], Y_dif_Pre_batch[11][3],
             Y_temp_out[4], Y_dif_Pre_batch[0][4], Y_dif_Pre_batch[1][4], Y_dif_Pre_batch[2][4], Y_dif_Pre_batch[3][4], Y_dif_Pre_batch[4][4], Y_dif_Pre_batch[5][4], Y_dif_Pre_batch[6][4], Y_dif_Pre_batch[7][4], Y_dif_Pre_batch[8][4], Y_dif_Pre_batch[9][4], Y_dif_Pre_batch[10][4], Y_dif_Pre_batch[11][4],
             Y_temp_out[5], Y_dif_Pre_batch[0][5], Y_dif_Pre_batch[1][5], Y_dif_Pre_batch[2][5], Y_dif_Pre_batch[3][5], Y_dif_Pre_batch[4][5], Y_dif_Pre_batch[5][5], Y_dif_Pre_batch[6][5], Y_dif_Pre_batch[7][5], Y_dif_Pre_batch[8][5], Y_dif_Pre_batch[9][5], Y_dif_Pre_batch[10][5], Y_dif_Pre_batch[11][5],
             Y_temp_out[6], Y_dif_Pre_batch[0][6], Y_dif_Pre_batch[1][6], Y_dif_Pre_batch[2][6], Y_dif_Pre_batch[3][6], Y_dif_Pre_batch[4][6], Y_dif_Pre_batch[5][6], Y_dif_Pre_batch[6][6], Y_dif_Pre_batch[7][6], Y_dif_Pre_batch[8][6], Y_dif_Pre_batch[9][6], Y_dif_Pre_batch[10][6], Y_dif_Pre_batch[11][6],
             ], nrow=plot_row)
    elif plot_row == 14:
        grid = make_grid(
            [Y_temp_out[0], Y_dif_Pre_batch[0][0], Y_dif_Pre_batch[1][0], Y_dif_Pre_batch[2][0], Y_dif_Pre_batch[3][0], Y_dif_Pre_batch[4][0], Y_dif_Pre_batch[5][0], Y_dif_Pre_batch[6][0], Y_dif_Pre_batch[7][0], Y_dif_Pre_batch[8][0], Y_dif_Pre_batch[9][0], Y_dif_Pre_batch[10][0], Y_dif_Pre_batch[11][0], Y_dif_Pre_batch[12][0],
             Y_temp_out[1], Y_dif_Pre_batch[0][1], Y_dif_Pre_batch[1][1], Y_dif_Pre_batch[2][1], Y_dif_Pre_batch[3][1], Y_dif_Pre_batch[4][1], Y_dif_Pre_batch[5][1], Y_dif_Pre_batch[6][1], Y_dif_Pre_batch[7][1], Y_dif_Pre_batch[8][1], Y_dif_Pre_batch[9][1], Y_dif_Pre_batch[10][1], Y_dif_Pre_batch[11][1], Y_dif_Pre_batch[12][1],
             Y_temp_out[2], Y_dif_Pre_batch[0][2], Y_dif_Pre_batch[1][2], Y_dif_Pre_batch[2][2], Y_dif_Pre_batch[3][2], Y_dif_Pre_batch[4][2], Y_dif_Pre_batch[5][2], Y_dif_Pre_batch[6][2], Y_dif_Pre_batch[7][2], Y_dif_Pre_batch[8][2], Y_dif_Pre_batch[9][2], Y_dif_Pre_batch[10][2], Y_dif_Pre_batch[11][2], Y_dif_Pre_batch[12][2],
             Y_temp_out[3], Y_dif_Pre_batch[0][3], Y_dif_Pre_batch[1][3], Y_dif_Pre_batch[2][3], Y_dif_Pre_batch[3][3], Y_dif_Pre_batch[4][3], Y_dif_Pre_batch[5][3], Y_dif_Pre_batch[6][3], Y_dif_Pre_batch[7][3], Y_dif_Pre_batch[8][3], Y_dif_Pre_batch[9][3], Y_dif_Pre_batch[10][3], Y_dif_Pre_batch[11][3], Y_dif_Pre_batch[12][3],
             Y_temp_out[4], Y_dif_Pre_batch[0][4], Y_dif_Pre_batch[1][4], Y_dif_Pre_batch[2][4], Y_dif_Pre_batch[3][4], Y_dif_Pre_batch[4][4], Y_dif_Pre_batch[5][4], Y_dif_Pre_batch[6][4], Y_dif_Pre_batch[7][4], Y_dif_Pre_batch[8][4], Y_dif_Pre_batch[9][4], Y_dif_Pre_batch[10][4], Y_dif_Pre_batch[11][4], Y_dif_Pre_batch[12][4],
             Y_temp_out[5], Y_dif_Pre_batch[0][5], Y_dif_Pre_batch[1][5], Y_dif_Pre_batch[2][5], Y_dif_Pre_batch[3][5], Y_dif_Pre_batch[4][5], Y_dif_Pre_batch[5][5], Y_dif_Pre_batch[6][5], Y_dif_Pre_batch[7][5], Y_dif_Pre_batch[8][5], Y_dif_Pre_batch[9][5], Y_dif_Pre_batch[10][5], Y_dif_Pre_batch[11][5], Y_dif_Pre_batch[12][5],
             Y_temp_out[6], Y_dif_Pre_batch[0][6], Y_dif_Pre_batch[1][6], Y_dif_Pre_batch[2][6], Y_dif_Pre_batch[3][6], Y_dif_Pre_batch[4][6], Y_dif_Pre_batch[5][6], Y_dif_Pre_batch[6][6], Y_dif_Pre_batch[7][6], Y_dif_Pre_batch[8][6], Y_dif_Pre_batch[9][6], Y_dif_Pre_batch[10][6], Y_dif_Pre_batch[11][6], Y_dif_Pre_batch[12][6],
             ], nrow=plot_row)
    elif plot_row == 15:
        grid = make_grid(
            [Y_temp_out[0], Y_dif_Pre_batch[0][0], Y_dif_Pre_batch[1][0], Y_dif_Pre_batch[2][0], Y_dif_Pre_batch[3][0], Y_dif_Pre_batch[4][0], Y_dif_Pre_batch[5][0], Y_dif_Pre_batch[6][0], Y_dif_Pre_batch[7][0], Y_dif_Pre_batch[8][0], Y_dif_Pre_batch[9][0], Y_dif_Pre_batch[10][0], Y_dif_Pre_batch[11][0], Y_dif_Pre_batch[12][0], Y_dif_Pre_batch[13][0],
             Y_temp_out[1], Y_dif_Pre_batch[0][1], Y_dif_Pre_batch[1][1], Y_dif_Pre_batch[2][1], Y_dif_Pre_batch[3][1], Y_dif_Pre_batch[4][1], Y_dif_Pre_batch[5][1], Y_dif_Pre_batch[6][1], Y_dif_Pre_batch[7][1], Y_dif_Pre_batch[8][1], Y_dif_Pre_batch[9][1], Y_dif_Pre_batch[10][1], Y_dif_Pre_batch[11][1], Y_dif_Pre_batch[12][1], Y_dif_Pre_batch[13][1],
             Y_temp_out[2], Y_dif_Pre_batch[0][2], Y_dif_Pre_batch[1][2], Y_dif_Pre_batch[2][2], Y_dif_Pre_batch[3][2], Y_dif_Pre_batch[4][2], Y_dif_Pre_batch[5][2], Y_dif_Pre_batch[6][2], Y_dif_Pre_batch[7][2], Y_dif_Pre_batch[8][2], Y_dif_Pre_batch[9][2], Y_dif_Pre_batch[10][2], Y_dif_Pre_batch[11][2], Y_dif_Pre_batch[12][2], Y_dif_Pre_batch[13][2],
             Y_temp_out[3], Y_dif_Pre_batch[0][3], Y_dif_Pre_batch[1][3], Y_dif_Pre_batch[2][3], Y_dif_Pre_batch[3][3], Y_dif_Pre_batch[4][3], Y_dif_Pre_batch[5][3], Y_dif_Pre_batch[6][3], Y_dif_Pre_batch[7][3], Y_dif_Pre_batch[8][3], Y_dif_Pre_batch[9][3], Y_dif_Pre_batch[10][3], Y_dif_Pre_batch[11][3], Y_dif_Pre_batch[12][3], Y_dif_Pre_batch[13][3],
             Y_temp_out[4], Y_dif_Pre_batch[0][4], Y_dif_Pre_batch[1][4], Y_dif_Pre_batch[2][4], Y_dif_Pre_batch[3][4], Y_dif_Pre_batch[4][4], Y_dif_Pre_batch[5][4], Y_dif_Pre_batch[6][4], Y_dif_Pre_batch[7][4], Y_dif_Pre_batch[8][4], Y_dif_Pre_batch[9][4], Y_dif_Pre_batch[10][4], Y_dif_Pre_batch[11][4], Y_dif_Pre_batch[12][4], Y_dif_Pre_batch[13][4],
             Y_temp_out[5], Y_dif_Pre_batch[0][5], Y_dif_Pre_batch[1][5], Y_dif_Pre_batch[2][5], Y_dif_Pre_batch[3][5], Y_dif_Pre_batch[4][5], Y_dif_Pre_batch[5][5], Y_dif_Pre_batch[6][5], Y_dif_Pre_batch[7][5], Y_dif_Pre_batch[8][5], Y_dif_Pre_batch[9][5], Y_dif_Pre_batch[10][5], Y_dif_Pre_batch[11][5], Y_dif_Pre_batch[12][5], Y_dif_Pre_batch[13][5],
             Y_temp_out[6], Y_dif_Pre_batch[0][6], Y_dif_Pre_batch[1][6], Y_dif_Pre_batch[2][6], Y_dif_Pre_batch[3][6], Y_dif_Pre_batch[4][6], Y_dif_Pre_batch[5][6], Y_dif_Pre_batch[6][6], Y_dif_Pre_batch[7][6], Y_dif_Pre_batch[8][6], Y_dif_Pre_batch[9][6], Y_dif_Pre_batch[10][6], Y_dif_Pre_batch[11][6], Y_dif_Pre_batch[12][6], Y_dif_Pre_batch[13][6],
             ], nrow=plot_row)
    elif plot_row == 16:
        grid = make_grid(
            [Y_temp_out[0], Y_dif_Pre_batch[0][0], Y_dif_Pre_batch[1][0], Y_dif_Pre_batch[2][0], Y_dif_Pre_batch[3][0], Y_dif_Pre_batch[4][0], Y_dif_Pre_batch[5][0], Y_dif_Pre_batch[6][0], Y_dif_Pre_batch[7][0], Y_dif_Pre_batch[8][0], Y_dif_Pre_batch[9][0], Y_dif_Pre_batch[10][0], Y_dif_Pre_batch[11][0], Y_dif_Pre_batch[12][0], Y_dif_Pre_batch[13][0], Y_dif_Pre_batch[14][0],
             Y_temp_out[1], Y_dif_Pre_batch[0][1], Y_dif_Pre_batch[1][1], Y_dif_Pre_batch[2][1], Y_dif_Pre_batch[3][1], Y_dif_Pre_batch[4][1], Y_dif_Pre_batch[5][1], Y_dif_Pre_batch[6][1], Y_dif_Pre_batch[7][1], Y_dif_Pre_batch[8][1], Y_dif_Pre_batch[9][1], Y_dif_Pre_batch[10][1], Y_dif_Pre_batch[11][1], Y_dif_Pre_batch[12][1], Y_dif_Pre_batch[13][1], Y_dif_Pre_batch[14][1],
             Y_temp_out[2], Y_dif_Pre_batch[0][2], Y_dif_Pre_batch[1][2], Y_dif_Pre_batch[2][2], Y_dif_Pre_batch[3][2], Y_dif_Pre_batch[4][2], Y_dif_Pre_batch[5][2], Y_dif_Pre_batch[6][2], Y_dif_Pre_batch[7][2], Y_dif_Pre_batch[8][2], Y_dif_Pre_batch[9][2], Y_dif_Pre_batch[10][2], Y_dif_Pre_batch[11][2], Y_dif_Pre_batch[12][2], Y_dif_Pre_batch[13][2], Y_dif_Pre_batch[14][2],
             Y_temp_out[3], Y_dif_Pre_batch[0][3], Y_dif_Pre_batch[1][3], Y_dif_Pre_batch[2][3], Y_dif_Pre_batch[3][3], Y_dif_Pre_batch[4][3], Y_dif_Pre_batch[5][3], Y_dif_Pre_batch[6][3], Y_dif_Pre_batch[7][3], Y_dif_Pre_batch[8][3], Y_dif_Pre_batch[9][3], Y_dif_Pre_batch[10][3], Y_dif_Pre_batch[11][3], Y_dif_Pre_batch[12][3], Y_dif_Pre_batch[13][3], Y_dif_Pre_batch[14][3],
             Y_temp_out[4], Y_dif_Pre_batch[0][4], Y_dif_Pre_batch[1][4], Y_dif_Pre_batch[2][4], Y_dif_Pre_batch[3][4], Y_dif_Pre_batch[4][4], Y_dif_Pre_batch[5][4], Y_dif_Pre_batch[6][4], Y_dif_Pre_batch[7][4], Y_dif_Pre_batch[8][4], Y_dif_Pre_batch[9][4], Y_dif_Pre_batch[10][4], Y_dif_Pre_batch[11][4], Y_dif_Pre_batch[12][4], Y_dif_Pre_batch[13][4], Y_dif_Pre_batch[14][4],
             Y_temp_out[5], Y_dif_Pre_batch[0][5], Y_dif_Pre_batch[1][5], Y_dif_Pre_batch[2][5], Y_dif_Pre_batch[3][5], Y_dif_Pre_batch[4][5], Y_dif_Pre_batch[5][5], Y_dif_Pre_batch[6][5], Y_dif_Pre_batch[7][5], Y_dif_Pre_batch[8][5], Y_dif_Pre_batch[9][5], Y_dif_Pre_batch[10][5], Y_dif_Pre_batch[11][5], Y_dif_Pre_batch[12][5], Y_dif_Pre_batch[13][5], Y_dif_Pre_batch[14][5],
             Y_temp_out[6], Y_dif_Pre_batch[0][6], Y_dif_Pre_batch[1][6], Y_dif_Pre_batch[2][6], Y_dif_Pre_batch[3][6], Y_dif_Pre_batch[4][6], Y_dif_Pre_batch[5][6], Y_dif_Pre_batch[6][6], Y_dif_Pre_batch[7][6], Y_dif_Pre_batch[8][6], Y_dif_Pre_batch[9][6], Y_dif_Pre_batch[10][6], Y_dif_Pre_batch[11][6], Y_dif_Pre_batch[12][6], Y_dif_Pre_batch[13][6], Y_dif_Pre_batch[14][6],
             ], nrow=plot_row)
    elif plot_row == 17:
        grid = make_grid(
            [Y_temp_out[0], Y_dif_Pre_batch[0][0], Y_dif_Pre_batch[1][0], Y_dif_Pre_batch[2][0], Y_dif_Pre_batch[3][0], Y_dif_Pre_batch[4][0], Y_dif_Pre_batch[5][0], Y_dif_Pre_batch[6][0], Y_dif_Pre_batch[7][0], Y_dif_Pre_batch[8][0], Y_dif_Pre_batch[9][0], Y_dif_Pre_batch[10][0], Y_dif_Pre_batch[11][0], Y_dif_Pre_batch[12][0], Y_dif_Pre_batch[13][0], Y_dif_Pre_batch[14][0], Y_dif_Pre_batch[15][0],
             Y_temp_out[1], Y_dif_Pre_batch[0][1], Y_dif_Pre_batch[1][1], Y_dif_Pre_batch[2][1], Y_dif_Pre_batch[3][1], Y_dif_Pre_batch[4][1], Y_dif_Pre_batch[5][1], Y_dif_Pre_batch[6][1], Y_dif_Pre_batch[7][1], Y_dif_Pre_batch[8][1], Y_dif_Pre_batch[9][1], Y_dif_Pre_batch[10][1], Y_dif_Pre_batch[11][1], Y_dif_Pre_batch[12][1], Y_dif_Pre_batch[13][1], Y_dif_Pre_batch[14][1], Y_dif_Pre_batch[15][1],
             Y_temp_out[2], Y_dif_Pre_batch[0][2], Y_dif_Pre_batch[1][2], Y_dif_Pre_batch[2][2], Y_dif_Pre_batch[3][2], Y_dif_Pre_batch[4][2], Y_dif_Pre_batch[5][2], Y_dif_Pre_batch[6][2], Y_dif_Pre_batch[7][2], Y_dif_Pre_batch[8][2], Y_dif_Pre_batch[9][2], Y_dif_Pre_batch[10][2], Y_dif_Pre_batch[11][2], Y_dif_Pre_batch[12][2], Y_dif_Pre_batch[13][2], Y_dif_Pre_batch[14][2], Y_dif_Pre_batch[15][2],
             Y_temp_out[3], Y_dif_Pre_batch[0][3], Y_dif_Pre_batch[1][3], Y_dif_Pre_batch[2][3], Y_dif_Pre_batch[3][3], Y_dif_Pre_batch[4][3], Y_dif_Pre_batch[5][3], Y_dif_Pre_batch[6][3], Y_dif_Pre_batch[7][3], Y_dif_Pre_batch[8][3], Y_dif_Pre_batch[9][3], Y_dif_Pre_batch[10][3], Y_dif_Pre_batch[11][3], Y_dif_Pre_batch[12][3], Y_dif_Pre_batch[13][3], Y_dif_Pre_batch[14][3], Y_dif_Pre_batch[15][3],
             Y_temp_out[4], Y_dif_Pre_batch[0][4], Y_dif_Pre_batch[1][4], Y_dif_Pre_batch[2][4], Y_dif_Pre_batch[3][4], Y_dif_Pre_batch[4][4], Y_dif_Pre_batch[5][4], Y_dif_Pre_batch[6][4], Y_dif_Pre_batch[7][4], Y_dif_Pre_batch[8][4], Y_dif_Pre_batch[9][4], Y_dif_Pre_batch[10][4], Y_dif_Pre_batch[11][4], Y_dif_Pre_batch[12][4], Y_dif_Pre_batch[13][4], Y_dif_Pre_batch[14][4], Y_dif_Pre_batch[15][4],
             Y_temp_out[5], Y_dif_Pre_batch[0][5], Y_dif_Pre_batch[1][5], Y_dif_Pre_batch[2][5], Y_dif_Pre_batch[3][5], Y_dif_Pre_batch[4][5], Y_dif_Pre_batch[5][5], Y_dif_Pre_batch[6][5], Y_dif_Pre_batch[7][5], Y_dif_Pre_batch[8][5], Y_dif_Pre_batch[9][5], Y_dif_Pre_batch[10][5], Y_dif_Pre_batch[11][5], Y_dif_Pre_batch[12][5], Y_dif_Pre_batch[13][5], Y_dif_Pre_batch[14][5], Y_dif_Pre_batch[15][5],
             Y_temp_out[6], Y_dif_Pre_batch[0][6], Y_dif_Pre_batch[1][6], Y_dif_Pre_batch[2][6], Y_dif_Pre_batch[3][6], Y_dif_Pre_batch[4][6], Y_dif_Pre_batch[5][6], Y_dif_Pre_batch[6][6], Y_dif_Pre_batch[7][6], Y_dif_Pre_batch[8][6], Y_dif_Pre_batch[9][6], Y_dif_Pre_batch[10][6], Y_dif_Pre_batch[11][6], Y_dif_Pre_batch[12][6], Y_dif_Pre_batch[13][6], Y_dif_Pre_batch[14][6], Y_dif_Pre_batch[15][6],
             ], nrow=plot_row)

    if train == True:
        save_image(grid, "result_reconstruction/image_train_e{}_grid_pre.png".format(epoch))
    else:
        save_image(grid, "result_reconstruction/image_test_e{}_grid_pre.png".format(epoch))
        
def calculate_histc(misclassified, i):
    # Calculate the histogram of the above created tensor
    len_list = len(misclassified)
    average_list = torch.zeros(1, len(misclassified[0]))
    expand_size = int(len(misclassified[0])/10)
    for l in range(0, len_list):
        average_list += misclassified[l]
    average_list = average_list/len_list

    qwert=average_list[0, i * expand_size:((i + 1) * expand_size)]
    hist = torch.histc(average_list[0, i * expand_size:((i + 1) * expand_size)], bins=expand_size, min=0, max=0)

    # Visualize above calculated histogram as bar diagram
    bins = expand_size
    x = range(bins)
    plt.bar(x, hist, align='center')
    plt.xlabel('Neurons')
    plt.ylabel('Numbers')
    # plt.show()
    plt.savefig('./plot_hist/Distribution_{}'.format(i))

def plot_x_y_axis(misclassified, k):
    len_list = len(misclassified)
    expand_size = int(len(misclassified[0]) / 10)
    neurons_value = torch.zeros(expand_size, len_list) # expand_size row neurons values.
    neurons_value_list = []
    neurons_value_all_list = []
    x_data = list(range(1, len_list+1))

    for l in range(0, len_list):
        for m in range(0, expand_size):
            neurons_value[m, l:l+1] = misclassified[l][(k * expand_size + m):(k * expand_size + m + 1)]
    for i in range(0, expand_size):
        neurons_value_list = neurons_value[i].numpy().tolist()
        neurons_value_all_list.append(neurons_value_list)

    for n in range(0, expand_size):
        if n == 0:
            ln1, = plt.plot(x_data, neurons_value_all_list[n], color='red', linewidth=1.0, linestyle=':')
            average = mean(neurons_value_all_list[n])
            average_line = [average]*len_list
            ln2, = plt.plot(x_data, average_line, color='blue', linewidth=1.0, linestyle='--')
            plt.legend(handles=[ln1, ln2], labels=['1st neuron value', 'average value'])
        elif n == 1:
            ln1, = plt.plot(x_data, neurons_value_all_list[n], color='green', linewidth=1.0, linestyle=':')
            average = mean(neurons_value_all_list[n])
            average_line = [average]*len_list
            ln2, = plt.plot(x_data, average_line, color='blue', linewidth=1.0, linestyle='--')
            plt.legend(handles=[ln1, ln2], labels=['2nd neuron value', 'average value'])
        elif n == 2:
            ln1, = plt.plot(x_data, neurons_value_all_list[n], color='purple', linewidth=1.0, linestyle=':')
            average = mean(neurons_value_all_list[n])
            average_line = [average]*len_list
            ln2, = plt.plot(x_data, average_line, color='blue', linewidth=1.0, linestyle='--')
            plt.legend(handles=[ln1, ln2], labels=['3rd neuron value', 'average value'])
        elif n == 3:
            ln1, = plt.plot(x_data, neurons_value_all_list[n], color='pink', linewidth=1.0, linestyle=':')
            average = mean(neurons_value_all_list[n])
            average_line = [average]*len_list
            ln2, = plt.plot(x_data, average_line, color='blue', linewidth=1.0, linestyle='--')
            plt.legend(handles=[ln1, ln2], labels=['4th neuron value', 'average value'])
        elif n == 4:
            ln1, = plt.plot(x_data, neurons_value_all_list[n], color='yellow', linewidth=1.0, linestyle=':')
            average = mean(neurons_value_all_list[n])
            average_line = [average]*len_list
            ln2, = plt.plot(x_data, average_line, color='blue', linewidth=1.0, linestyle='--')
            plt.legend(handles=[ln1, ln2], labels=['5th neuron value', 'average value'])
        elif n == 5:
            ln1, = plt.plot(x_data, neurons_value_all_list[n], color='orange', linewidth=1.0, linestyle=':')
            average = mean(neurons_value_all_list[n])
            average_line = [average]*len_list
            ln2, = plt.plot(x_data, average_line, color='blue', linewidth=1.0, linestyle='--')
            plt.legend(handles=[ln1, ln2], labels=['6th neuron value', 'average value'])


        # ln2, = plt.plot(x_data, neurons_value_all_list[1], color='blue', linewidth=2.0, linestyle='--')
        # ln3, = plt.plot(x_data, neurons_value_all_list[2], color='green', linewidth=2.0, linestyle='--')
        # ln4, = plt.plot(x_data, neurons_value_all_list[3], color='purple', linewidth=2.0, linestyle='--')
        # ln5, = plt.plot(x_data, neurons_value_all_list[4], color='pink', linewidth=2.0, linestyle='--')
        # ln6, = plt.plot(x_data, neurons_value_all_list[5], color='yellow', linewidth=2.0, linestyle='--')
        #
        # plt.legend(handles=[ln1, ln2, ln3, ln4, ln5, ln6], labels=['1st neuron', '2nd neuron', '3rd neuron', '4th neuron', '5th neuron', '6th neuron'])
        if n == 0:
            plt.title('Value of {}st neuron (Misclassification of digit {})'.format(n+1, k))
        elif n == 1:
            plt.title('Value of {}nd neuron (Misclassification of digit {})'.format(n+1, k))
        elif n == 2:
            plt.title('Value of {}rd neuron (Misclassification of digit {})'.format(n+1, k))
        else:
            plt.title('Value of {}th neuron (Misclassification of digit {})'.format(n+1, k))

        # plt.figure(dpi=2400)
        plt.xlabel('Number of misclassified samples')
        plt.ylabel('Neuron output value')
        plt.savefig('./plot_hist/Distribution_{}_digit_{}_neuron'.format(k,n), dpi=1200)
        plt.close()
        # plt.show()
    for n in range(0, expand_size):
        if n == 0:
            average = mean(neurons_value_all_list[n])
            average_line = [average] * len_list
            ln_0, = plt.plot(x_data, average_line, color='red', linewidth=1.0, linestyle='--')
        elif n == 1:
            average = mean(neurons_value_all_list[n])
            average_line = [average] * len_list
            ln_1, = plt.plot(x_data, average_line, color='green', linewidth=1.0, linestyle='--')
        elif n == 2:
            average = mean(neurons_value_all_list[n])
            average_line = [average] * len_list
            ln_2, = plt.plot(x_data, average_line, color='purple', linewidth=1.0, linestyle='--')
        elif n == 3:
            average = mean(neurons_value_all_list[n])
            average_line = [average] * len_list
            ln_3, = plt.plot(x_data, average_line, color='pink', linewidth=1.0, linestyle='--')
        elif n == 4:
            average = mean(neurons_value_all_list[n])
            average_line = [average] * len_list
            ln_4, = plt.plot(x_data, average_line, color='yellow', linewidth=1.0, linestyle='--')
        elif n == 5:
            average = mean(neurons_value_all_list[n])
            average_line = [average] * len_list
            ln_5, = plt.plot(x_data, average_line, color='orange', linewidth=1.0, linestyle='--')


    plt.legend(handles=[ln_0, ln_1, ln_2, ln_3, ln_4, ln_5], labels=['Average value of 1st neuron', 'Average value of 2nd neuron', 'Average value of 3rd neuron', 'Average value of 4th neuron', 'Average value of 5th neuron', 'Average value of 6th neuron'])
    plt.xlabel('Number of misclassified samples')
    plt.ylabel('Average of neuron outputs')
    plt.title('Average of neuron outputs (Misclassification of digit {})'.format(k))
    plt.savefig('./plot_hist/Average_{}_digit_{}_neuron'.format(k, n), dpi=1200)
    plt.close()


def plot_distribution(misclassified, k, calculate_flag, misclassified_flag):
    len_list = len(misclassified)
    if len_list == 0:
        return 0
    expand_size = int(len(misclassified[0]) / 10)
    neurons_value = torch.zeros(expand_size, len_list) # expand_size row neurons values.
    neurons_value_list = []
    neurons_value_all_list = []

    comparision_neurons_value = torch.zeros(expand_size, len_list) # expand_size row neurons values.
    comparision_neurons_value_list = []
    comparision_neurons_value_all_list = []

    x_data = list(range(1, len_list+1))

    # distribute 6 neurons to 6 row tensor
    for l in range(0, len_list):
        for m in range(0, expand_size):
            neurons_value[m, l:l+1] = misclassified[l][(k * expand_size + m):(k * expand_size + m + 1)]
            # when input 1 digit collected data, compares 0 digit collected data, otherwise compares 1 digit collected data.
            # compare other class distribution
            if k != 1:
                plot_class_number = 1
                comparision_neurons_value[m, l:l+1] = misclassified[l][(plot_class_number * expand_size + m):(plot_class_number * expand_size + m + 1)]
            else:
                plot_class_number = 0
                comparision_neurons_value[m, l:l+1] = misclassified[l][(plot_class_number * expand_size + m):(plot_class_number * expand_size + m + 1)]

    for i in range(0, expand_size):
        neurons_value_list = neurons_value[i].numpy().tolist()
        neurons_value_all_list.append(neurons_value_list)

        # when input 1 digit collected data, compares 0 digit collected data, otherwise compares 1 digit collected data.
        comparision_neurons_value_list = comparision_neurons_value[i].numpy().tolist()
        comparision_neurons_value_all_list.append(comparision_neurons_value_list)


    for n in range(0, expand_size):
        if n == 0:
            plt.hist(np.array(neurons_value_all_list[n]), bins= 50, color='red', label=['1st neuron value distribution'])
        elif n == 1:
            plt.hist(np.array(neurons_value_all_list[n]), bins= 50, color='green', label=['2nd neuron value distribution'])
        elif n == 2:
            plt.hist(np.array(neurons_value_all_list[n]), bins= 50, color='purple', label=['3rd neuron value distribution'])
        elif n == 3:
            plt.hist(np.array(neurons_value_all_list[n]), bins= 50, color='pink', label=['4th neuron value distribution'])
        elif n == 4:
            plt.hist(np.array(neurons_value_all_list[n]), bins= 50, color='yellow', label=['5th neuron value distribution'])
        elif n == 5:
            plt.hist(np.array(neurons_value_all_list[n]), bins= 50, color='orange', label=['6th neuron value distribution'])

        if misclassified_flag == True:
            if n == 0:
                plt.title('Distribution of value of {}st neuron (Misclassification of digit {})'.format(n+1, k))
            elif n == 1:
                plt.title('Distribution of  value of {}nd neuron (Misclassification of digit {})'.format(n+1, k))
            elif n == 2:
                plt.title('Distribution of value of {}rd neuron (Misclassification of digit {})'.format(n+1, k))
            else:
                plt.title('Distribution of value of {}th neuron (Misclassification of digit {})'.format(n+1, k))
        elif misclassified_flag == False:
            if n == 0:
                plt.title('Distribution of value of {}st neuron (Correct classification of digit {})'.format(n+1, k))
            elif n == 1:
                plt.title('Distribution of  value of {}nd neuron (Correct classification of digit {})'.format(n+1, k))
            elif n == 2:
                plt.title('Distribution of value of {}rd neuron (Correct classification of digit {})'.format(n+1, k))
            else:
                plt.title('Distribution of value of {}th neuron (Correct classification of digit {})'.format(n+1, k))

        plt.xlabel('Value')
        plt.ylabel('Number')
        if misclassified_flag == True:
            if calculate_flag == 1:
                plt.savefig('./plot_hist_train/Distribution_Mis_Train_{}_digit_{}_neuron'.format(k,n), dpi=1200)
            elif calculate_flag == 2:
                plt.savefig('./plot_hist_test_1/Distribution_Mis_Test_1_{}_digit_{}_neuron'.format(k,n), dpi=1200)
            elif calculate_flag == 3:
                plt.savefig('./plot_hist_test_2/Distribution_Mis_Test_2_{}_digit_{}_neuron'.format(k,n), dpi=1200)
            else:
                plt.savefig('./plot_hist/ERROR'.format(k,n), dpi=1200)
        elif misclassified_flag == False:
            if calculate_flag == 1:
                plt.savefig('./plot_hist_train/Distribution_Cor_Train_{}_digit_{}_neuron'.format(k,n), dpi=1200)
            elif calculate_flag == 2:
                plt.savefig('./plot_hist_test_1/Distribution_Cor_Test_1_{}_digit_{}_neuron'.format(k,n), dpi=1200)
            elif calculate_flag == 3:
                plt.savefig('./plot_hist_test_2/Distribution_Cor_Test_2_{}_digit_{}_neuron'.format(k,n), dpi=1200)
            else:
                plt.savefig('./plot_hist/ERROR'.format(k,n), dpi=1200)


        plt.close()
        # plt.show()

    ##################################plot comparision distribution############################################
    #when input correctly classified data, plot the distribution of other class
    if misclassified_flag == False:
        for n in range(0, expand_size):
            if n == 0:
                plt.hist(np.array(comparision_neurons_value_all_list[n]), bins= 50, color='red', label=['1st neuron value distribution'])
            elif n == 1:
                plt.hist(np.array(comparision_neurons_value_all_list[n]), bins= 50, color='green', label=['2nd neuron value distribution'])
            elif n == 2:
                plt.hist(np.array(comparision_neurons_value_all_list[n]), bins= 50, color='purple', label=['3rd neuron value distribution'])
            elif n == 3:
                plt.hist(np.array(comparision_neurons_value_all_list[n]), bins= 50, color='pink', label=['4th neuron value distribution'])
            elif n == 4:
                plt.hist(np.array(comparision_neurons_value_all_list[n]), bins= 50, color='yellow', label=['5th neuron value distribution'])
            elif n == 5:
                plt.hist(np.array(comparision_neurons_value_all_list[n]), bins= 50, color='orange', label=['6th neuron value distribution'])

            if misclassified_flag == True:
                if n == 0:
                    plt.title('Distribution of value of {}st neuron (Misclassification of digit {})'.format(n+1, k))
                elif n == 1:
                    plt.title('Distribution of  value of {}nd neuron (Misclassification of digit {})'.format(n+1, k))
                elif n == 2:
                    plt.title('Distribution of value of {}rd neuron (Misclassification of digit {})'.format(n+1, k))
                else:
                    plt.title('Distribution of value of {}th neuron (Misclassification of digit {})'.format(n+1, k))
            elif misclassified_flag == False:
                if n == 0:
                    plt.title('Distribution of value of {}st neuron (Compare with digit {})'.format(n+1, plot_class_number))
                elif n == 1:
                    plt.title('Distribution of  value of {}nd neuron (Compare with digit {})'.format(n+1, plot_class_number))
                elif n == 2:
                    plt.title('Distribution of value of {}rd neuron (Compare with digit {})'.format(n+1, plot_class_number))
                else:
                    plt.title('Distribution of value of {}th neuron (Compare with digit {})'.format(n+1, plot_class_number))

            plt.xlabel('Value')
            plt.ylabel('Number')
            if misclassified_flag == True:
                if calculate_flag == 1:
                    plt.savefig('./plot_hist_train/Distribution_Mis_Train_{}_digit_{}_neuron'.format(k,n), dpi=1200)
                elif calculate_flag == 2:
                    plt.savefig('./plot_hist_test_1/Distribution_Mis_Test_1_{}_digit_{}_neuron'.format(k,n), dpi=1200)
                elif calculate_flag == 3:
                    plt.savefig('./plot_hist_test_2/Distribution_Mis_Test_2_{}_digit_{}_neuron'.format(k,n), dpi=1200)
                else:
                    plt.savefig('./plot_hist/ERROR'.format(k,n), dpi=1200)
            elif misclassified_flag == False:
                if calculate_flag == 1:
                    plt.savefig('./plot_hist_train/Distribution_Train_Compare_with_{}_digit_{}_neuron'.format(plot_class_number,n), dpi=1200)
                elif calculate_flag == 2:
                    plt.savefig('./plot_hist_test_1/Distribution_Test_1_Compare_with_{}_digit_{}_neuron'.format(plot_class_number,n), dpi=1200)
                elif calculate_flag == 3:
                    plt.savefig('./plot_hist_test_2/Distribution_Test_2_Compare_with_{}_digit_{}_neuron'.format(plot_class_number,n), dpi=1200)
                else:
                    plt.savefig('./plot_hist/ERROR'.format(plot_class_number,n), dpi=1200)


            plt.close()
            # plt.show()

