import os
import torch
import numpy as np
from pypc import utils
from torch import nn
import torch.nn.functional as F
import math
import time
import random
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PC_Model(nn.Module):
    def __init__(self, cf):
        super().__init__()
        self.nodes = cf.nodes
        self.Y_lr = cf.dy_lr
        self.l_nodes = len(cf.nodes)
        self.l_layers = len(cf.nodes)-1
        self.layers = []
        self.loss_list = []
        self.loss_list_rev = []
        self.dy_beta = cf.dy_beta
        self.dy_alpha = cf.dy_alpha
        self.batch_size = cf.batch_size
        self.expand_size = cf.expand_size
        self.decay_dy = cf.decay_dy
        self.Y0_limited = cf.Y0_limited
        self.weight_share = cf.weight_share
        self.weight_no_share_num = cf.weight_no_share_num
        for l in range (self.l_layers):
            self.num_channels = cf.num_channels[l]
            self.kernel_size=cf.kernel_size[l]
            self.stride=cf.stride[l]
            self.padding=cf.padding[l]

            if l == (self.l_layers-1):
                if l >= (self.l_layers - self.weight_no_share_num):
                    layer = ConFullNet(weight_share=False, in_size=cf.nodes[l], out_size=cf.nodes[l + 1], num_filters=self.num_channels, batch_size=cf.batch_size, act_function = cf.act_function, use_bias=cf.use_bias)
                else:
                    layer = ConFullNet(weight_share=cf.weight_share, in_size=cf.nodes[l], out_size=cf.nodes[l + 1],num_filters=self.num_channels, batch_size=cf.batch_size, act_function=cf.act_function, use_bias=cf.use_bias)
            else:
                num_filters = cf.num_channels[l+1]
                if l >= (self.l_layers - self.weight_no_share_num):
                    layer = ConvLayer(weight_share=False, input_size=cf.nodes[l], rev_input_size=cf.nodes[l+1], num_channels=self.num_channels, num_filters=num_filters, rev_num_filters=self.num_channels, batch_size=cf.batch_size, kernel_size=self.kernel_size, learning_rate=cf.dy_lr, act_fn=cf.act_function, padding=self.padding, stride=self.stride, device=DEVICE)
                else:
                    layer = ConvLayer(weight_share=cf.weight_share, input_size=cf.nodes[l], rev_input_size=cf.nodes[l+1], num_channels=self.num_channels, num_filters=num_filters, rev_num_filters=self.num_channels, batch_size=cf.batch_size, kernel_size=self.kernel_size, learning_rate=cf.dy_lr, act_fn=cf.act_function, padding=self.padding, stride=self.stride, device=DEVICE)
            self.layers.append(layer)

    def Define_PC_Param(self):
        self.Y_Pre = [[] for _ in range(self.l_nodes)]
        self.Error = [[] for _ in range(self.l_nodes)]
        self.test_class_Error = [[] for _ in range(self.l_nodes)]
        self.Y_Pre_rev = [[] for _ in range(self.l_nodes)]
        self.Error_rev = [[] for _ in range(self.l_nodes)]
        self.Y = [[] for _ in range(self.l_nodes)]

    def Reset_Y(self, batch_size, init_std):
        for l in range(self.l_layers):
            self.Y[l] = utils.set_tensor(torch.empty(batch_size, self.layers[l].in_size).normal_(mean=0, std=init_std).to(DEVICE))
    def Set_Y_train(self,):
        self.Y_train = utils.set_tensor(torch.zeros(self.batch_size, self.nodes[-1]).to(DEVICE))

    def Set_Y_0(self, ):
        self.Y[0] = utils.set_tensor( torch.zeros(self.batch_size, self.nodes[0]).to(DEVICE))

    def Set_input(self, input):
        self.Y[0] = input.clone()

    def Set_target(self, target):
        self.Y[-1] = target.clone()
        self.pre_class = target.clone()

    def Propagate_Y(self):
        for l in range(1, self.l_layers):
            self.Y[l] = self.layers[l - 1].forward(self.Y[l - 1])
    def Propagate_class(self, image_batch):
        self.Y[0] = image_batch.clone()
        start = time.time()
        for l in range(0, self.l_layers):
            self.Y[l+1] = self.layers[l].forward(self.Y[l])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.time()
        self.class_time = end - start

    def Propagate_reconstruction(self):
        Y_tem = torch.zeros(self.Y[-1].shape).to(DEVICE)
        for i in range(self.Y[-1].shape[0]):
            max_position = torch.argmax(self.Y[-1][i])
            Y_tem[i, max_position] = 1
        self.Y[-1] = Y_tem

        for n in range(self.l_layers - 1, -1, -1):
            self.Y[n] = self.layers[n].forward_rev(self.Y[n + 1])
        self.Y_Pre_rev[0] = self.Y[0]

    def dif_layers_reconstruction(self, image_batch):
        self.layers_reconstruction = []
        self.Error_reconstruction = []
        self.reconstruction_time = []
        start = time.time()
        for n in range(0, self.l_layers):
            self.Y_Pre_rev[n+1] = self.Y[n + 1]
            for l in range(n, -1, -1):
                self.Y_Pre_rev[l] = self.layers[l].forward_rev(self.Y_Pre_rev[l+1])               
            if torch.cuda.is_available():
                torch.cuda.synchronize()    # <---------------- extra line                
            end = time.time()
            self.reconstruction_time.append(end - start)
            self.layers_reconstruction.append(self.Y_Pre_rev[l])

            error_rec = image_batch - self.Y_Pre_rev[l]
            loss = torch.sum(error_rec ** 2).item()
            loss = loss / (torch.numel(error_rec))
            self.Error_reconstruction.append(loss)

    def reconstruction_cal(self):
        self.all_layers_reconstruction = []
        for n in range(0, self.l_layers):
            self.Y_Pre_rev[n+1] = self.Y[n + 1]
            for l in range(n, -1, -1):
                self.Y_Pre_rev[l] = self.layers[l].forward_rev(self.Y_Pre_rev[l+1])
            self.all_layers_reconstruction.append(self.Y_Pre_rev[l])

    def forward(self, val):
        for layer in self.layers:
            val = layer.forward(val)
        return val

    def train_class(self, img_batch, label_batch, cf):
        self.Define_PC_Param()
        if cf.addGaussianNoiselables == True:
            shrink_lable_noise = label_batch + (0.01) * torch.randn((label_batch.size())[0],(label_batch.size())[1]).to(DEVICE)
            self.Set_target(shrink_lable_noise)
        else:
            self.Set_target(label_batch)
        self.Set_input(img_batch)
        self.Set_Y_train()
        self.Propagate_Y()
        if cf.addGaussianNoiselables == True:
            self.train_updates_class(cf.Train_iteration, shrink_lable_noise, fixed_preds=cf.fixed_Y_Pre_train)
        else:
            self.train_updates_class(cf.Train_iteration, label_batch, fixed_preds=cf.fixed_Y_Pre_train)

        self.update_grads_double()
        # self.update_grads()
        self.get_all_loss()
        self.get_all_loss_rev()

    def test_class(self, img_batch, label_batch, cf):
        self.Propagate_class(img_batch)
        self.pre_class = self.Y[-1]
        self.Y[0] = self.Y[0].normal_(mean=0, std=cf.init_std).to(DEVICE)  # add
        self.dif_layers_reconstruction(img_batch)
        self.test_class_Error[0] = label_batch - self.pre_class
        self.Y[0] = img_batch
        self.get_all_test_class_loss()
        self.get_all_loss()
        self.get_all_loss_rev()

    def test_class_reconstrution(self, img_batch, label_batch, cf):
        self.Define_PC_Param()
        self.Propagate_class(img_batch)
        self.pre_class = self.Y[-1]  # get predictive classification results
        self.Y[0] = self.Y[0].normal_(mean=0, std=cf.init_std).to(DEVICE)  # add
        self.dif_layers_reconstruction(img_batch)
        self.test_class_Error[0] = label_batch - self.pre_class
        self.Y[0] = img_batch  # add
        self.get_all_test_class_loss()

    def train_updates_class(self, n_iters, label_batch, fixed_preds=False):
        for n in range(1, self.l_nodes):
            self.Y_Pre[n] = self.layers[n - 1].forward(self.Y[n - 1])
            self.Error[n] = self.Y[n] - self.Y_Pre[n]
        for n in range(self.l_nodes-1, 0, -1):
            self.Y_Pre_rev[n-1] = self.layers[n - 1].forward_rev(self.Y[n])
            self.Error_rev[n-1] = self.Y[n-1] - self.Y_Pre_rev[n-1]

        for itr in range(n_iters):
            for l in range(1, self.l_layers):
                dY =self.dy_beta * (self.layers[l].backward(self.Error[l + 1]) - self.Error[l]) + self.dy_alpha * (self.layers[l - 1].backward_rev(self.Error_rev[l - 1]) - self.Error_rev[l])
                self.Y[l] = self.Y[l] + self.Y_lr * (dY - self.decay_dy * self.Y[l])

            for n in range(1, self.l_nodes):
                if not fixed_preds:
                    self.Y_Pre[n] = self.layers[n - 1].forward(self.Y[n - 1])
                self.Error[n] = self.Y[n] - self.Y_Pre[n]

            for n in range(self.l_nodes - 1, 0, -1):
                if not fixed_preds:
                    self.Y_Pre_rev[n - 1] = self.layers[n - 1].forward_rev(self.Y[n])
                self.Error_rev[n - 1] = self.Y[n - 1] - self.Y_Pre_rev[n - 1]

    def update_grads(self):
        for l in range(self.l_layers):
            self.layers[l].update_gradient(self.Error[l + 1])

    def update_grads_double(self):
        for l in range(self.l_layers):
            self.layers[l].update_gradient(self.Error[l + 1])
            self.layers[l].update_gradient_rev(self.Error_rev[l])

    def get_target_loss(self):
        return torch.sum(self.Error[-1] ** 2).item()

    def get_all_loss(self):
        self.loss_list = []
        for l in range (1,self.l_nodes):
            loss = torch.sum(self.Error[l] ** 2).item()
            loss = loss / (torch.numel(self.Error[l]))  # add
            self.loss_list.append(loss)

    def get_all_test_class_loss(self):
        self.test_class_loss_list = []
        for l in range (1,self.l_nodes):
            loss = torch.sum(self.test_class_Error[0] ** 2).item()
            loss = loss / (torch.numel(self.test_class_Error[0]))  # add
            self.test_class_loss_list.append(loss)

    def get_all_loss_rev(self):
        self.loss_list_rev = []
        for l in range (0,self.l_nodes-1):
            loss = torch.sum(self.Error_rev[l] ** 2).item()
            loss = loss/(torch.numel(self.Error_rev[l]))#add
            self.loss_list_rev.append(loss)

    @property
    def params(self):
        return self.layers

    def get_weight(self, params, optimizer, weight_share):
        self.weights_list = []
        self.bias_list = []

        self.weights_rev_list = []
        self.bias_rev_list = []

        self.weights_grad_list = []
        self.bias_grad_list = []
        self.weights_rev_grad_list = []
        self.bias_rev_grad_list = []

        self.c_b_list = []
        self.c_w_list = []
        self.v_b_list = []
        self.v_w_list = []
        self.c_b_rev_list = []
        self.c_w_rev_list = []
        self.v_b_rev_list = []
        self.v_w_rev_list = []
        for p, param in enumerate(params):
            if weight_share == True:
                self.weights_list.append(param.weights)
                self.bias_list.append(param.bias)
            else:
                self.weights_list.append(param.weights)
                self.bias_list.append(param.bias)
                self.weights_rev_list.append(param.weights_rev)
                self.bias_rev_list.append(param.bias_rev)

            self.weights_grad_list.append(param.grad["weights"])
            self.bias_grad_list.append(param.grad["bias"])
            self.weights_rev_grad_list.append(param.grad_rev["weights_rev"])
            self.bias_rev_grad_list.append(param.grad_rev["bias_rev"])

            self.c_b_list.append(optimizer.c_b[p])
            self.c_w_list.append(optimizer.c_w[p])
            self.v_b_list.append(optimizer.v_b[p])
            self.v_w_list.append(optimizer.v_w[p])
            self.c_b_rev_list.append(optimizer.c_b_rev[p])
            self.c_w_rev_list.append(optimizer.c_w_rev[p])
            self.v_b_rev_list.append(optimizer.v_b_rev[p])
            self.v_w_rev_list.append(optimizer.v_w_rev[p])

        # create checkpoint variable and add important data
        if weight_share == True:
            self.checkpoint = {
                'weights_list': self.weights_list,
                'bias_list': self.bias_list,

                'weights_grad_list': self.weights_grad_list,
                'bias_grad_list': self.bias_grad_list,
                'weights_rev_grad_list': self.weights_rev_grad_list,
                'bias_rev_grad_list': self.bias_rev_grad_list,

                'c_b_list': self.c_b_list,
                'c_w_list': self.c_w_list,
                'v_b_list': self.v_b_list,
                'v_w_list': self.v_w_list,

                'c_b_rev_list': self.c_b_rev_list,
                'c_w_rev_list': self.c_w_rev_list,
                'v_b_rev_list': self.v_b_rev_list,
                'v_w_rev_list': self.v_w_rev_list,
            }
        else:
            self.checkpoint = {
                'weights_list': self.weights_list,
                'bias_list': self.bias_list,

                'weights_rev_list': self.weights_rev_list,
                'bias_rev_list': self.bias_rev_list,

                'weights_grad_list': self.weights_grad_list,
                'bias_grad_list': self.bias_grad_list,
                'weights_rev_grad_list': self.weights_rev_grad_list,
                'bias_rev_grad_list': self.bias_rev_grad_list,

                'c_b_list': self.c_b_list,
                'c_w_list': self.c_w_list,
                'v_b_list': self.v_b_list,
                'v_w_list': self.v_w_list,

                'c_b_rev_list': self.c_b_rev_list,
                'c_w_rev_list': self.c_w_rev_list,
                'v_b_rev_list': self.v_b_rev_list,
                'v_w_rev_list': self.v_w_rev_list,
            }

    def load_weight(self, optimizer, checkpoint, weight_share):
        for p, param in enumerate(self.params):

            param.weights = checkpoint['weights_list'][p]
            param.bias = checkpoint['bias_list'][p]

            if weight_share == False:
                param.weights_rev = checkpoint['weights_rev_list'][p]
                param.bias_rev = checkpoint['bias_rev_list'][p]

            param.grad["weights"] = checkpoint['weights_grad_list'][p]
            param.grad["bias"] = checkpoint['bias_grad_list'][p]
            param.grad_rev["weights_rev"] = checkpoint['weights_rev_grad_list'][p]
            param.grad_rev["bias_rev"] = checkpoint['bias_rev_grad_list'][p]

            optimizer.c_b[p] = checkpoint['c_b_list'][p]
            optimizer.c_w[p] = checkpoint['c_w_list'][p]
            optimizer.v_b[p] = checkpoint['v_b_list'][p]
            optimizer.v_w[p] = checkpoint['v_w_list'][p]

            optimizer.c_b_rev[p] = checkpoint['c_b_rev_list'][p]
            optimizer.c_w_rev[p] = checkpoint['c_w_rev_list'][p]
            optimizer.v_b_rev[p] = checkpoint['v_b_rev_list'][p]
            optimizer.v_w_rev[p] = checkpoint['v_w_rev_list'][p]

    def load_only_weight(self, checkpoint, weight_share):
        for p, param in enumerate(self.params):
            param.weights = checkpoint['weights_list'][p]
            param.bias = checkpoint['bias_list'][p]
            if weight_share == False:
                param.weights_rev = checkpoint['weights_rev_list'][p]
                param.bias_rev = checkpoint['bias_rev_list'][p]
                
    def test_load_weight(self, checkpoint, weight_share):
        for p, param in enumerate(self.params):
            param.weights = checkpoint['weights_list'][p]
            param.bias = checkpoint['bias_list'][p]

            if weight_share == False:
                param.weights_rev = checkpoint['weights_rev_list'][p]
                param.bias_rev = checkpoint['bias_rev_list'][p]

            param.grad["weights"] = checkpoint['weights_grad_list'][p]
            param.grad["bias"] = checkpoint['bias_grad_list'][p]
            param.grad_rev["weights_rev"] = checkpoint['weights_rev_grad_list'][p]
            param.grad_rev["bias_rev"] = checkpoint['bias_rev_grad_list'][p]

    def save(self, model, optimizer, dir_name, epoch, weight_share):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        file_path = os.path.join(dir_name, "model_{}".format(epoch))
        self.get_weight(self.params, optimizer, weight_share)
        torch.save(model.checkpoint, file_path)
        print("saved: {}_{}".format(dir_name, epoch))
        return file_path

    def save_v2(self, model, optimizer, dir_name, epoch, fold, weight_share):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        file_path = os.path.join(dir_name, "model_{}_{}".format(fold, epoch))
        self.get_weight(self.params, optimizer, weight_share)
        torch.save(model.checkpoint, file_path)
        print("saved: {}_{}_{}".format(dir_name, fold, epoch))


    def load(self, optimizer, dir_name, epoch, weight_share):
        file_path = os.path.join(dir_name, "model_{}".format(epoch))
        if not os.path.exists(file_path):
            print("saved file not found")
            return
        self.load_weight(optimizer, torch.load(file_path), weight_share)
        print("loaded: model_{}".format(epoch))

    def load_weights(self, dir_name, epoch, weight_share):
        file_path = os.path.join(dir_name, "model_{}".format(epoch))
        if not os.path.exists(file_path):
            print("saved file not found")
            return
        self.load_only_weight(torch.load(file_path), weight_share)
        print("loaded: model_{}".format(epoch))
        
    def test_load(self, dir_name, epoch, weight_share):
        # file_path = os.path.join(dir_name, "checkpoint")
        file_path = os.path.join(dir_name, "checkpoint\\model_{}".format(epoch))
        if not os.path.exists(file_path):
            print("saved file not found")
            return
        self.test_load_weight(torch.load(file_path), weight_share)
        print("loaded: checkpoint")

    def test_load_v2(self, dir_name, epoch, fold, weight_share):
        # file_path = os.path.join(dir_name, "checkpoint")
        file_path = os.path.join(dir_name, "checkpoint\\model_{}_{}".format(fold, epoch))
        if not os.path.exists(file_path):
            print("saved file not found")
            return
        self.test_load_weight(torch.load(file_path), weight_share)
        print("loaded: checkpoint")


class PC_Layer(object):
    def __init__(self, in_size, out_size, act_function, use_bias=False, is_forward=False):

        self.in_size = in_size
        self.out_size = out_size

        weights = torch.empty((self.in_size, self.out_size)).normal_(mean=0.0, std=0.05).to(DEVICE)
        bias = torch.zeros((self.out_size)).to(DEVICE)
        self.weights = utils.set_tensor(weights)
        self.bias = utils.set_tensor(bias)

        self.is_forward = is_forward

        self.grad = {"weights": None, "bias": None}
        self.grad_rev = {"weights_rev": None, "bias_rev": None}

        self.input = None
        self.input_reverse = None

        self.act_fn = act_function
        self.use_bias = use_bias

    def forward(self, input):
        self.input = input.clone()
        out = self.act_fn(torch.matmul(self.input, self.weights))
        if self.use_bias:
            out = out + self.bias
        return out

    def forward_rev(self, input):
        self.input_reverse = input.clone()
        out = self.act_fn(torch.matmul(self.input_reverse, self.weights.T))
        if self.use_bias:
            out = out + self.bias.T
        return out

    def backward(self, err):
        fn_deriv = self.act_fn.deriv(torch.matmul(self.input, self.weights))
        out = torch.matmul(err * fn_deriv, self.weights.T)
        return out

    def backward_rev(self, err):
        fn_deriv = self.act_fn.deriv(torch.matmul(self.input_reverse, self.weights.T))
        out = torch.matmul(err * fn_deriv, self.weights)
        return out

    def update_gradient(self, err):
        fn_deriv = self.act_fn.deriv(torch.matmul(self.input, self.weights))
        dU = torch.matmul(self.input.T, err * fn_deriv)
        self.grad["weights"] = dU
        if self.use_bias:
            self.grad["bias"] = torch.sum(err, axis=0)

    def update_gradient_rev(self, err):
        fn_deriv = self.act_fn.deriv(torch.matmul(self.input_reverse, self.weights.T))
        dU = torch.matmul(self.input_reverse.T, err * fn_deriv)
        self.grad_rev["weights_rev"] = dU
        if self.use_bias:
            self.grad_rev["bias_rev"] = torch.sum(err, axis=0)


    def _reset_grad(self):
        self.grad = {"weights": None, "bias": None}

    def _reset_grad_rev(self):
        self.grad_rev = {"weights_rev": None, "bias_rev": None}


######################################################################
class ConvLayer(object):
  def __init__(self, weight_share, input_size,rev_input_size, num_channels,num_filters, rev_num_filters, batch_size,kernel_size,learning_rate,act_fn, padding=0,stride=1, is_forward=False, use_bias=False, device="cpu"):
    self.input_size = input_size
    self.rev_input_size = rev_input_size
    self.num_channels = num_channels
    self.num_filters = num_filters
    self.batch_size = batch_size
    self.kernel_size = kernel_size
    self.padding = padding
    self.stride = stride
    self.rev_num_filters = rev_num_filters

    self.weight_share = weight_share
    self.output_size = math.floor((self.input_size + (2 * self.padding) - self.kernel_size)/self.stride) + 1
    self.learning_rate = learning_rate
    self.act_fn = act_fn
    self.device = device
    if self.weight_share == True:
        self.weights = torch.empty(self.num_filters, self.num_channels, self.kernel_size, self.kernel_size).normal_(mean=0, std=0.05).to(self.device)
        self.bias = torch.zeros((self.output_size)).to(DEVICE)
    else:
        self.weights = torch.empty(self.num_filters,self.num_channels,self.kernel_size,self.kernel_size).normal_(mean=0,std=0.05).to(self.device)
        self.weights_rev = torch.empty(self.num_filters,self.num_channels,self.kernel_size,self.kernel_size).normal_(mean=0,std=0.05).to(self.device)
        self.bias = torch.zeros((self.output_size)).to(DEVICE)
        self.bias_rev = torch.zeros((self.output_size)).to(DEVICE)

    self.grad = {"weights": None, "bias": None}
    self.grad_rev = {"weights_rev": None, "bias_rev": None}
    self.unfold = nn.Unfold(kernel_size=(self.kernel_size, self.kernel_size), padding=self.padding,stride=self.stride).to(self.device)
    self.fold = nn.Fold(output_size=(self.input_size, self.input_size),kernel_size=(self.kernel_size, self.kernel_size), padding=self.padding,stride=self.stride).to(self.device)

    self.is_forward = is_forward
    self.use_bias = use_bias
    ############################

  def forward(self,inp):
    self.X_col = self.unfold(inp.clone())
    self.flat_weights = self.weights.reshape(self.num_filters,-1)
    out = self.flat_weights @ self.X_col   # @ denote matmul(a, b)
    self.activations = out.reshape(self.batch_size, self.num_filters, self.output_size, self.output_size)
    return self.act_fn(self.activations)

  def forward_rev(self,inp):
    self.X_col_rev = self.unfold(inp.clone())
    if self.weight_share == True:
        weight_shape = self.weights.shape
        flat_weights_rev = self.weights.reshape((weight_shape[1], weight_shape[0], weight_shape[2], weight_shape[3]))
        self.flat_weights_rev = flat_weights_rev.reshape(self.num_channels, -1)
    else:
        weight_shape = self.weights_rev.shape
        flat_weights_rev = self.weights_rev.reshape((weight_shape[1], weight_shape[0], weight_shape[2], weight_shape[3]))
        self.flat_weights_rev = flat_weights_rev.reshape(self.num_channels,-1)
    out = self.flat_weights_rev @ self.X_col_rev   # @ denote matmul(a, b)
    self.activations_rev = out.reshape(self.batch_size, self.num_channels, self.output_size, self.output_size)
    return self.act_fn(self.activations_rev)

  def update_gradient(self,e):
    fn_deriv = self.act_fn.deriv(self.activations)
    e = e * fn_deriv
    self.dout = e.reshape(self.batch_size,self.num_filters,-1)
    dW = self.dout @ self.X_col.permute(0,2,1)
    dW = torch.sum(dW,dim=0)
    dW = dW.reshape((self.num_filters,self.num_channels,self.kernel_size,self.kernel_size))
    self.grad["weights"] = dW

  def update_gradient_rev(self,e):
    fn_deriv = self.act_fn.deriv(self.activations_rev)
    e = e * fn_deriv
    self.dout_rev = e.reshape(self.batch_size, self.num_channels, -1)
    dW = self.dout_rev @ self.X_col_rev.permute(0, 2, 1)
    dW = torch.sum(dW, dim=0)
    dW = dW.reshape((self.num_filters, self.num_channels,  self.kernel_size, self.kernel_size))
    self.grad_rev["weights_rev"] = dW

  def backward(self,e):
    fn_deriv = self.act_fn.deriv(self.activations)
    e = e * fn_deriv
    self.dout = e.reshape(self.batch_size,self.num_filters,-1)
    dX_col = self.flat_weights.T @ self.dout
    dX = self.fold(dX_col)
    return torch.clamp(dX,-50,50)

  def backward_rev(self,e):
    fn_deriv = self.act_fn.deriv(self.activations_rev)
    e = e * fn_deriv
    self.dout_rev = e.reshape(self.batch_size,self.num_channels,-1)
    dX_col_rev = self.flat_weights_rev.T @ self.dout_rev
    dX = self.fold(dX_col_rev)
    return torch.clamp(dX,-50,50)

  def _reset_grad(self):
    self.grad = {"weights": None, "bias": None}

  def _reset_grad_rev(self):
    self.grad_rev = {"weights_rev": None, "bias_rev": None}


class ConFullNet(object):
    def __init__(self, weight_share, in_size, out_size, num_filters, batch_size, act_function, use_bias=False, is_forward=False):
        self.in_size = in_size
        self.out_size = out_size
        self.num_filters = num_filters
        self.weight_share = weight_share

        if self.weight_share == True:
            self.weights = torch.empty((self.num_filters * self.in_size * self.in_size, self.out_size)).normal_(mean=0.0, std=0.05).to(DEVICE)
            self.bias = torch.zeros((self.out_size)).to(DEVICE)
        else:
            self.weights = torch.empty((self.num_filters * self.in_size * self.in_size, self.out_size)).normal_(mean=0.0, std=0.05).to(DEVICE)
            self.bias = torch.zeros((self.out_size)).to(DEVICE)
            self.weights_rev = torch.empty((self.num_filters * self.in_size * self.in_size, self.out_size)).normal_(mean=0.0, std=0.05).to(DEVICE)
            self.bias_rev = torch.zeros((self.out_size)).to(DEVICE)
        #################################################
        self.batch_size = batch_size
        self.is_forward = is_forward
        self.grad = {"weights": None, "bias": None}
        self.grad_rev = {"weights_rev": None, "bias_rev": None}
        self.input = None
        self.input_reverse = None
        self.act_fn = act_function
        self.use_bias = use_bias
    def forward(self, input):
        self.input = torch.flatten(input, 1)  ##add
        out = self.act_fn(torch.matmul(self.input, self.weights))
        if self.use_bias:
            out = out + self.bias
        return out
    def forward_rev(self, input):
        self.input_reverse = input.clone()
        if self.weight_share == True:
            x = self.act_fn(torch.matmul(self.input_reverse, self.weights.T))
        else:
            x = self.act_fn(torch.matmul(self.input_reverse, self.weights_rev.T))
        out = x.reshape(self.batch_size, self.num_filters, self.in_size, self.in_size)
        if self.use_bias:
            out = out + self.bias.T
        return out

    def backward(self, err):
        fn_deriv = self.act_fn.deriv(torch.matmul(self.input, self.weights))
        x = torch.matmul(err * fn_deriv, self.weights.T)
        out = x.reshape(self.batch_size, self.num_filters, self.in_size, self.in_size)
        return out

    def backward_rev(self, err):
        err = torch.flatten(err, 1)  ##add
        if self.weight_share == True:
            fn_deriv = self.act_fn.deriv(torch.matmul(self.input_reverse, self.weights.T))
            out = torch.matmul(err * fn_deriv, self.weights)
        else:
            fn_deriv = self.act_fn.deriv(torch.matmul(self.input_reverse, self.weights_rev.T))
            out = torch.matmul(err * fn_deriv, self.weights_rev)
        return out
    def update_gradient(self, err):
        fn_deriv = self.act_fn.deriv(torch.matmul(self.input, self.weights))
        dU = torch.matmul(self.input.T, err * fn_deriv)
        self.grad["weights"] = dU
        if self.use_bias:
            self.grad["bias"] = torch.sum(err, axis=0)
    def update_gradient_rev(self, err):
        err = torch.flatten(err, 1)  ##add
        if self.weight_share == True:
            fn_deriv = self.act_fn.deriv(torch.matmul(self.input_reverse, self.weights.T))
        else:
            fn_deriv = self.act_fn.deriv(torch.matmul(self.input_reverse, self.weights_rev.T))
        dU = torch.matmul(self.input_reverse.T, err * fn_deriv)
        self.grad_rev["weights_rev"] = dU.T
        if self.use_bias:
            self.grad_rev["bias_rev"] = torch.sum(err, axis=0)

    def _reset_grad(self):
        self.grad = {"weights": None, "bias": None}
    def _reset_grad_rev(self):
        self.grad_rev = {"weights_rev": None, "bias_rev": None}

def set_seed(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

