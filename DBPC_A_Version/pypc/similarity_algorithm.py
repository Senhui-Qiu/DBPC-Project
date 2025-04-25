import torch
from numpy import *
from torchvision.utils import save_image
import torchvision
from torchvision.utils import make_grid




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

    if (Y_batch.size())[1] == 784:
        Y_batch_temp = torch.reshape(Y_batch, ((Y_batch.size())[0], 1, 28, 28))  # fc need to add
    else:
        Y_batch_temp = torch.reshape(Y_batch, ((Y_batch.size())[0], 3, int(math.sqrt((Y_batch.size())[1]/3)), int(math.sqrt((Y_batch.size())[1]/3))))  # fc need to add

    all_layers_reconstruction_tem = []
    for n in range(0, len(all_layers_reconstruction)):
        if (Y_batch.size())[1] == 784:
            Y_Pre_temp_out = torch.reshape(all_layers_reconstruction[n], ((Y_batch.size())[0], 1, 28, 28))  # fc need to add
        else:
            Y_Pre_temp_out = torch.reshape(all_layers_reconstruction[n], ((Y_batch.size())[0], 3, int(math.sqrt((Y_batch.size())[1]/3)), int(math.sqrt((Y_batch.size())[1]/3))))  # fc need to add
        all_layers_reconstruction_tem.append(Y_Pre_temp_out)

    for i in range(len(all_layers_reconstruction)):
        Y_tem = all_layers_reconstruction_tem[i]
        image_rec = Y_tem.mul(255.0).add_(0.5).clamp_(0, 255)
        image_original = Y_batch_temp.mul(255.0).add_(0.5).clamp_(0, 255)
        mse = torch.mean(((image_rec/1.0) - (image_original/1.0)) ** 2)
        if mse < 1.0e-10:
            out[i] = 100
        out[i] = 10*math.log10(255.0**2/mse)
    return out



def save_pre_image(Y_Pre_batch, Y_batch, epoch, train):
        trans = torchvision.transforms.Resize(size=112)
        if (Y_batch.size())[1] == 784:
            Y_Pre_temp_out = torch.reshape(Y_Pre_batch[0], (1, 28, 28))#fc need to add
            Y_temp_out = torch.reshape(Y_batch[0], (1, 28, 28))  # fc need to add
        else:
            Y_Pre_temp_out = torch.reshape(Y_Pre_batch[0], (3, int(math.sqrt((Y_batch.size())[1]/3)), int(math.sqrt((Y_batch.size())[1]/3))))#fc need to add
            Y_temp_out = torch.reshape(Y_batch[0], (3, int(math.sqrt((Y_batch.size())[1]/3)), int(math.sqrt((Y_batch.size())[1]/3))))  # fc need to add
        Y_Pre_out = trans(Y_Pre_temp_out)
        Y_out = trans(Y_temp_out)
        if train == True:
            save_image(Y_Pre_out, "result_reconstruction/image_{}_pre.png".format(epoch))
            save_image(Y_out, "result_reconstruction/image_{}.png".format(epoch))
        else:
            save_image(Y_Pre_out, "result_reconstruction/image_test_{}_pre.png".format(epoch))
            save_image(Y_out, "result_reconstruction/image_test_{}.png".format(epoch))

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
    else:
        Y_temp_out = torch.reshape(Y_batch, ((Y_batch.size())[0], 3, int(math.sqrt((Y_batch.size())[1]/3)), int(math.sqrt((Y_batch.size())[1]/3))))  # fc need to add

    Y_dif_Pre_batch = []
    for n in range(0, len(Y_dif_Pre_batch_input)):
        if (Y_batch.size())[1] == 784:
            Y_Pre_temp_out = torch.reshape(Y_dif_Pre_batch_input[n], ((Y_batch.size())[0], 1, 28, 28))  # fc need to add
        else:
            Y_Pre_temp_out = torch.reshape(Y_dif_Pre_batch_input[n], ((Y_batch.size())[0], 3, int(math.sqrt((Y_batch.size())[1]/3)), int(math.sqrt((Y_batch.size())[1]/3))))  # fc need to add
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

    if train == True:
        save_image(grid, "result_reconstruction/image_train_e{}_grid_pre.png".format(epoch))
    else:
        save_image(grid, "result_reconstruction/image_test_e{}_grid_pre.png".format(epoch))


