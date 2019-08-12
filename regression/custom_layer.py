import torch
import numpy as np
from torch.autograd import Function
import torch.nn as nn
from torch.nn.parameter import Parameter
import ipdb
from torch.nn import init
import math
import torch.nn.functional as F
import numbers
from copy import copy
# # from meta_neural_network_architectures import extract_top_level_dict
# def extract_sub_level_parameters(params, sub_level=None):
#     if sub_level is None:
#         assert("Sub-level should not be None [expected type int]")
#     sub_params=OrderedDict()
#     for k, v in params.items():
#         if(str(sub_level) in k):
#             layers_dir = k.split(".")
#             sub_params[layers_dir[-1]] = v
#     print("{} -> {}".format(sub_params.keys(), sub_params.values()))
#     return params


# Inherit from Function
class LinearFunction(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, local_M_context, weight, M_matrix_gen_weight, bias=None, M_matrix_gen_bias=None, backprop=False):
        forward_flag = torch.tensor([True ^ backprop])
        ctx.save_for_backward(input, local_M_context, weight, M_matrix_gen_weight, bias, M_matrix_gen_bias, forward_flag)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, local_M_context, weight, M_matrix_gen_weight, bias, M_matrix_gen_bias, forward_flag = ctx.saved_tensors

        grad_input = grad_local_M_context = grad_weight = grad_bias = grad_M_matrix_gen_weight = grad_M_matrix_gen_bias = None
        backprop_grad = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.

        # print("forward_flag: {}".format(forward_flag[0]))
        M_matrix = None
        # print("local_M_context.size: {}".format(local_M_context.size()))
        # print("M_matrix_gen_weight.size: {}".format(M_matrix_gen_weight.size()))
        if forward_flag[0]:
            M_matrix = F.linear(local_M_context, weight=M_matrix_gen_weight, bias=M_matrix_gen_bias).view(weight.size()[0], weight.size()[0])

        # compute gradient of INPUT
        if ctx.needs_input_grad[0]: 
            grad_input = grad_output.mm(weight)

        # compute gradient of WEIGHT
        if ctx.needs_input_grad[1]:
            # print("forward_flag: {}".format(forward_flag[0]))
            if forward_flag[0]:
                grad_weight = grad_output.t().mm(input)
                # print("M_matrix.size(): {}".format(M_matrix.size()))
                # print("grad_weight.size(): {}".format(grad_weight.size()))
                grad_weight = M_matrix.mm(grad_weight)
            else:
                grad_weight = grad_output.t().mm(input) #*2

        # compute gradient of BIAS
        if bias is not None and ctx.needs_input_grad[2]:
            if forward_flag[0]:
                grad_bias = grad_output.sum(0)#.squeeze(0)
                # print("grad_bias.size : {}".format(grad_bias.size()))
                grad_bias = grad_bias.unsqueeze(1).t().mm(M_matrix).squeeze(1)
                # print("grad_bias.size : {}".format(grad_bias.size()))
            else:
                # print("! forward")
                grad_bias = grad_output.sum(0)#.squeeze(0)

        grad_M_matrix_gen_weight =  torch.zeros_like(M_matrix_gen_weight)
        grad_M_matrix_gen_bias = torch.zeros_like(M_matrix_gen_bias)
        grad_local_M_context = torch.zeros_like(local_M_context)
        forward_flag[0] = 0
        return grad_input ,grad_local_M_context, grad_weight, grad_M_matrix_gen_weight, grad_bias, grad_M_matrix_gen_bias, backprop_grad

class MetaLinearLayer_dual(nn.Module):
    def __init__(self, input_shape, num_filters, num_M_context_variable = 5, use_bias=True, backprop=False):
        super(MetaLinearLayer_dual, self).__init__()

        # forward parameters
        c = input_shape
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.ones(num_filters, c))
        self.num_M_context_variable = num_M_context_variable
        # print(".num_filters: {}".format(num_filters))
        if(num_M_context_variable is not None):
            self.M_matrix_gen_weight = nn.Parameter(torch.ones(num_filters * num_filters, self.num_M_context_variable))
            nn.init.xavier_uniform_(self.M_matrix_gen_weight)
        nn.init.xavier_uniform_(self.weight)
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(num_filters))
            self.M_matrix_gen_bias = nn.Parameter(torch.zeros( num_filters * num_filters))
        self.backprop = backprop

        # parameters for producing M-matrix
        self.local_M_context = nn.Parameter(torch.ones(self.num_M_context_variable))
        self.local_M_context_copy = self.local_M_context.clone().detach().cpu().numpy()

    # def reset_parameters(self):
    #     super(MetaLinearLayer_dual, self).reset_parameters()
    #     init.kaiming_uniform_(self.weight_back, a=math.sqrt(5))

    def forward(self, x, params=None, num_step=None):
        """
        Forward propagates by applying a linear function (Wx + b). If params are none then internal params are used.
        Otherwise passed params will be used to execute the function.
        :param x: Input data batch, in the form (b, f)
        :param params: A dictionary containing 'weights' and 'bias'. If params are none then internal params are used.
        Otherwise the external are used.
        :return: The result of the linear function.
        """

        # diff = self.local_M_context_copy - self.local_M_context.detach().cpu().numpy()
        # print("diff: {}".format(diff))

        
        # print("######## \t DUAL LAYER FORWARD PASS \t ##############")
        # for k,v in params.items():
        #     print("{}".format(k))

        if params is None:
            if self.use_bias:
                weight, bias = self.weight, self.bias
                M_matrix_gen_weight, M_matrix_gen_bias = self.M_matrix_gen_weight, self.M_matrix_gen_bias
            else:
                weight = self.weight
                M_matrix_gen_weight = self.M_matrix_gen_weight
                bias = None
                M_matrix_gen_bias = None
            local_M_context = self.local_M_context

        else:
            weight, bias = params["weight"], params["bias"]
            M_matrix_gen_weight, M_matrix_gen_bias = params["M_matrix_gen_weight"], params["M_matrix_gen_bias"]
            # print(x.shape)
            local_M_context = params["local_M_context"]
        out = LinearFunction.apply(x, local_M_context, weight, M_matrix_gen_weight, bias, M_matrix_gen_bias, self.backprop)
        return out


#### Linear testing
# inp_tensor1 = torch.randn((10,10))
# inp_tensor2 = inp_tensor1.clone()
# inp_tensor1.requires_grad_()
# inp_tensor2.requires_grad_()
# weight = torch.randn((1,10))
# weight.requires_grad_()
# out = LinearFunction.apply(inp_tensor1, weight, weight)
# out.sum().backward()
# layer1 = LinearLayer(10, 1)
# out = layer1(inp_tensor2)
# ipdb.set_trace()
# out.sum().backward(create_graph=True)
# inp_tensor2.grad.sum().backward()

#### Conv testing
# inp_tensor1 = torch.randn((10,3,10,10))
# inp_tensor2 = inp_tensor1.clone()
# inp_tensor1.requires_grad_()
# inp_tensor2.requires_grad_()
# # weight = torch.randn((1,10))
# # weight.requires_grad_()
# # out = LinearFunction.apply(inp_tensor1, weight, weight)
# # out.sum().backward()
# layer1 = MetaConv2dLayer(3, 2, 3, 1, 1, True)
# out = layer1(inp_tensor2)
# # ipdb.set_trace()
# grads = torch.autograd.grad((out**2).sum(), [inp_tensor2, layer1.weight], create_graph=True)
# grads[1].sum().backward()


