# Copyright 2022 by Senhui Qiu, Ulster University.
# All rights reserved.
import torch
from pypc import utils
from torch import nn
import time

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

        self.decay_dy = cf.decay_dy

        self.weight_share = cf.weight_share

        for l in range (self.l_layers):
            layer = PC_Layer(weight_share=cf.weight_share, in_size=cf.nodes[l], out_size=cf.nodes[l + 1], act_function = cf.act_function, use_bias=cf.use_bias)
            self.layers.append(layer)


    def Define_PC_Param(self):
        self.Y_Pre = [[] for _ in range(self.l_nodes)]
        self.Error = [[] for _ in range(self.l_nodes)]
        self.test_class_Error = [[] for _ in range(self.l_nodes)]
        self.Y_Pre_rev = [[] for _ in range(self.l_nodes)]
        self.Error_rev = [[] for _ in range(self.l_nodes)]
        self.Y = [[] for _ in range(self.l_nodes)]


    def Set_Y_train(self,):
        self.Y_train = utils.set_tensor(torch.zeros(self.batch_size, self.nodes[-1]).to(DEVICE))


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
            torch.cuda.synchronize()    # <---------------- extra line                
        end = time.time()
        self.class_time = end - start


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
        self.Set_target(label_batch)
        self.Set_input(img_batch)
        self.Set_Y_train()
        self.Propagate_Y()
        self.train_updates_class(cf.Train_iteration, label_batch, fixed_preds=cf.fixed_Y_Pre_train)
        self.update_grads_double()
        # self.update_grads()
        self.get_all_loss()
        self.get_all_loss_rev()


    def test_class(self, img_batch, label_batch, cf):
        self.Propagate_class(img_batch)
        self.pre_class = self.Y[-1]  # get predictive classification results
        self.Y[0] = torch.zeros(self.Y[0].shape)#add
        self.Y[0] = self.Y[0].normal_(mean=0, std=cf.init_std).to(DEVICE)  # add
        self.dif_layers_reconstruction(img_batch)
        self.test_class_Error[0] = label_batch - self.pre_class
        self.Y[0] = img_batch  # add
        self.get_all_test_class_loss()
        self.get_all_loss()
        self.get_all_loss_rev()



    def train_updates_class(self, n_iters, label_batch, fixed_preds=False):
        for n in range(1, self.l_nodes):
            self.Y_Pre[n] = self.layers[n - 1].forward(self.Y[n - 1])
            self.Error[n] = self.Y[n] - self.Y_Pre[n]

        for n in range(self.l_nodes-1, 0, -1):
            self.Y_Pre_rev[n-1] = self.layers[n - 1].forward_rev(self.Y[n])
            self.Error_rev[n-1] = self.Y[n-1] - self.Y_Pre_rev[n-1]

        for itr in range(n_iters):
            ##################get Y[0] predictive value########################
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
            self.loss_list.append(loss)

    def get_all_test_class_loss(self):
        self.test_class_loss_list = []
        for l in range (1,self.l_nodes):
            loss = torch.sum(self.test_class_Error[0] ** 2).item()
            self.test_class_loss_list.append(loss)

    def get_all_loss_rev(self):
        self.loss_list_rev = []
        for l in range (0,self.l_nodes-1):
            loss = torch.sum(self.Error_rev[l] ** 2).item()
            self.loss_list_rev.append(loss)

    @property
    def params(self):
        return self.layers


class PC_Layer(object):
    def __init__(self, weight_share, in_size, out_size, act_function, use_bias=False, is_forward=False):

        self.in_size = in_size
        self.out_size = out_size

        weights = torch.empty((self.in_size, self.out_size)).normal_(mean=0.0, std=0.05).to(DEVICE)
        bias = torch.zeros((self.out_size)).to(DEVICE)
        self.weights = utils.set_tensor(weights)
        self.bias = utils.set_tensor(bias)


        weights_rev = torch.empty((self.in_size, self.out_size)).normal_(mean=0.0, std=0.05).to(DEVICE)
        bias_rev = torch.zeros((self.out_size)).to(DEVICE)
        self.weights_rev = utils.set_tensor(weights_rev)
        self.bias_rev = utils.set_tensor(bias_rev)

        self.weight_share = weight_share
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
        if self.weight_share == True:
            out = self.act_fn(torch.matmul(self.input_reverse, self.weights.T))
        else:
            out = self.act_fn(torch.matmul(self.input_reverse, self.weights_rev.T))
        if self.use_bias:
            out = out + self.bias.T
        return out


    def backward(self, err):
        fn_deriv = self.act_fn.deriv(torch.matmul(self.input, self.weights))
        out = torch.matmul(err * fn_deriv, self.weights.T)
        return out

    def backward_rev(self, err):
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
        if self.weight_share == True:
            fn_deriv = self.act_fn.deriv(torch.matmul(self.input_reverse, self.weights.T))
            dU = torch.matmul(self.input_reverse.T, err * fn_deriv)
            self.grad_rev["weights_rev"] = dU
            if self.use_bias:
                self.grad_rev["bias_rev"] = torch.sum(err, axis=0)
        else:
            fn_deriv = self.act_fn.deriv(torch.matmul(self.input_reverse, self.weights_rev.T))
            dU = torch.matmul(self.input_reverse.T, err * fn_deriv)
            self.grad_rev["weights_rev"] = dU
            if self.use_bias:
                self.grad_rev["bias_rev"] = torch.sum(err, axis=0)

    def _reset_grad(self):
        self.grad = {"weights": None, "bias": None}

    def _reset_grad_rev(self):
        self.grad_rev = {"weights_rev": None, "bias_rev": None}


