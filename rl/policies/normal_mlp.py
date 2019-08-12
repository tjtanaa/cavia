import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from policies.policy import Policy, weight_init


class NormalMLPPolicy(Policy):
    """Policy network based on a multi-layer perceptron (MLP), with a 
    `Normal` distribution output, with trainable standard deviation. This 
    policy network can be used on tasks with continuous action spaces (eg. 
    `HalfCheetahDir`). The code is adapted from 
    https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/sandbox/rocky/tf/policies/maml_minimal_gauss_mlp_policy.py
    """

    def __init__(self, input_size, output_size, hidden_sizes=(),
                 nonlinearity=F.relu, init_std=1.0, min_std=1e-6):
        super(NormalMLPPolicy, self).__init__(input_size=input_size, output_size=output_size)
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.min_log_std = math.log(min_std)
        self.num_layers = len(hidden_sizes) + 1

        layer_sizes = (input_size,) + hidden_sizes
        for i in range(1, self.num_layers):
            self.add_module('layer{0}'.format(i),
                            nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
        self.mu = nn.Linear(layer_sizes[-1], output_size)

        self.sigma = nn.Parameter(torch.Tensor(output_size))
        self.sigma.data.fill_(math.log(init_std))
        self.apply(weight_init)

    def forward(self, input, params=None):

        if params is None:
            params = OrderedDict(self.named_parameters())

        output = input
        for i in range(1, self.num_layers):
            output = F.linear(output,
                              weight=params['layer{0}.weight'.format(i)],
                              bias=params['layer{0}.bias'.format(i)])
            output = self.nonlinearity(output)
        mu = F.linear(output, weight=params['mu.weight'],
                      bias=params['mu.bias'])
        scale = torch.exp(torch.clamp(params['sigma'], min=self.min_log_std))

        return Normal(loc=mu, scale=scale)


# class CaviaMLPPolicy(Policy, nn.Module):
#     """CAVIA network based on a multi-layer perceptron (MLP), with a
#     `Normal` distribution output, with trainable standard deviation. This
#     policy network can be used on tasks with continuous action spaces (eg.
#     `HalfCheetahDir`).
#     """

#     def __init__(self, input_size, output_size, device, hidden_sizes=(), num_context_params=10,
#                  nonlinearity=F.relu, init_std=1.0, min_std=1e-6):
#         super(CaviaMLPPolicy, self).__init__(input_size, output_size)
#         self.input_size = input_size
#         self.output_size = output_size
#         self.device = device

#         self.hidden_sizes = hidden_sizes
#         self.nonlinearity = nonlinearity
#         self.min_log_std = math.log(min_std)
#         self.num_layers = len(hidden_sizes) + 1
#         self.context_params = []

#         layer_sizes = (input_size,) + hidden_sizes
#         self.add_module('layer{0}'.format(1), nn.Linear(layer_sizes[0] + num_context_params, layer_sizes[1]))
#         for i in range(2, self.num_layers):
#             self.add_module('layer{0}'.format(i), nn.Linear(layer_sizes[i - 1], layer_sizes[i]))

#         self.num_context_params = num_context_params
#         self.context_params = torch.zeros(self.num_context_params, requires_grad=True).to(self.device)

#         self.mu = nn.Linear(layer_sizes[-1], output_size)
#         self.sigma = nn.Parameter(torch.Tensor(output_size))
#         self.sigma.data.fill_(math.log(init_std))
#         self.apply(weight_init)

#     def forward(self, input, params=None):

#         # if no parameters are given, use the standard ones
#         if params is None:
#             params = OrderedDict(self.named_parameters())

#         # concatenate context parameters to input
#         output = torch.cat((input, self.context_params.expand(input.shape[:-1] + self.context_params.shape)),
#                            dim=len(input.shape) - 1)

#         # forward through FC Layer
#         for i in range(1, self.num_layers):
#             output = F.linear(output, weight=params['layer{0}.weight'.format(i)],
#                               bias=params['layer{0}.bias'.format(i)])

#         # last layer outputs mean; scale is a learned param independent of the input
#         mu = F.linear(output, weight=params['mu.weight'], bias=params['mu.bias'])
#         scale = torch.exp(torch.clamp(params['sigma'], min=self.min_log_std))

#         return Normal(loc=mu, scale=scale)

#     def update_params(self, loss, step_size, first_order=False, params=None):
#         """Apply one step of gradient descent on the loss function `loss`, with
#         step-size `step_size`, and returns the updated parameters of the neural
#         network.
#         """

#         # take the gradient wrt the context params
#         grads = torch.autograd.grad(loss, self.context_params, create_graph=not first_order)[0]

#         # set correct computation graph
#         if not first_order:
#             self.context_params = self.context_params - step_size * grads
#         else:
#             self.context_params = self.context_params - step_size * grads.detach()

#         return OrderedDict(self.named_parameters())

#     def reset_context(self):
#         self.context_params = torch.zeros(self.num_context_params, requires_grad=True).to(self.device)


class CaviaMLPPolicy(Policy, nn.Module):
    """CAVIA network based on a multi-layer perceptron (MLP), with a
    `Normal` distribution output, with trainable standard deviation. This
    policy network can be used on tasks with continuous action spaces (eg.
    `HalfCheetahDir`).
    """

    def __init__(self, input_size, output_size, device, hidden_sizes=(), num_context_params=10,
                 nonlinearity=F.relu, init_std=1.0, min_std=1e-6):
        super(CaviaMLPPolicy, self).__init__(input_size, output_size)
        self.input_size = input_size
        self.output_size = output_size
        self.device = device

        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.min_log_std = math.log(min_std)
        self.num_layers = len(hidden_sizes) + 1
        self.context_params = []

        layer_sizes = (input_size,) + hidden_sizes
        self.add_module('layer{0}'.format(1), nn.Linear(layer_sizes[0] + num_context_params, layer_sizes[1]))
        for i in range(2, self.num_layers):
            self.add_module('layer{0}'.format(i), nn.Linear(layer_sizes[i - 1], layer_sizes[i]))

        self.num_context_params = num_context_params
        self.context_params = torch.zeros(self.num_context_params, requires_grad=True).to(self.device)

        self.mu = nn.Linear(layer_sizes[-1], output_size)
        self.sigma = nn.Parameter(torch.Tensor(output_size))
        self.sigma.data.fill_(math.log(init_std))

        self.frozen_identity = torch.eye(self.num_context_params, requires_grad=False).to(self.device)
        # self.frozen_identity.requires_grad = False

        # saved M_matrix for further visualization/ analysis
        self.M_saved = OrderedDict()
        self.F_saved = OrderedDict()
        self.L_saved= OrderedDict()
        self.R_saved= OrderedDict()
        self.grad_saved = OrderedDict()
        self.i_saved = 0
        self.g = None
        self.gcontext = None
        self.norm_ggT = []
        self.norm_masked_ggT = []
        self.norm_fim_inverse =[]
        self.norm_norm_fim_inverse =[]
        self.apply(weight_init)

    def forward(self, input, params=None):

        # if no parameters are given, use the standard ones
        if params is None:
            params = OrderedDict(self.named_parameters())

        context_params = self.context_params.unsqueeze(0).mm(self.frozen_identity).expand(input.shape[:-1] + self.context_params.shape)
        self.gcontext = context_params
        self.gcontext.retain_grad()

        output = torch.cat((input, context_params),
                           dim=len(input.shape) - 1)

        # # concatenate context parameters to input
        # output = torch.cat((input, self.context_params.expand(input.shape[:-1] + self.context_params.shape)),
        #                    dim=len(input.shape) - 1)

        # forward through FC Layer
        for i in range(1, self.num_layers):
            output = F.linear(output, weight=params['layer{0}.weight'.format(i)],
                              bias=params['layer{0}.bias'.format(i)])

        # last layer outputs mean; scale is a learned param independent of the input
        mu = F.linear(output, weight=params['mu.weight'], bias=params['mu.bias'])
        scale = torch.exp(torch.clamp(params['sigma'], min=self.min_log_std))

        return Normal(loc=mu, scale=scale)

    def update_params(self, loss, step_size, first_order=False, params=None):
        """Apply one step of gradient descent on the loss function `loss`, with
        step-size `step_size`, and returns the updated parameters of the neural
        network.
        """

        # take the gradient wrt the context params
        grads = torch.autograd.grad(loss, self.context_params, create_graph=not first_order)[0]
        b = self.context_params.size()
        if(self.gcontext.grad.dim() > 2):
            m, n, c = self.gcontext.grad.size()
            g_reshaped = self.gcontext.grad.view(m*n,1,c)
            expected_ggT = torch.bmm(g_reshaped.permute(0,2,1), g_reshaped).detach() 
            expected_ggT = expected_ggT.mean(0) - grads.unsqueeze(1).permute(1,0).mm(grads.unsqueeze(1))
        else:
            n, c = self.gcontext.grad.size()
            g_reshaped = self.gcontext.grad.view(n,1,c)
            expected_ggT = torch.bmm(g_reshaped.permute(0,2,1), g_reshaped).detach()
            expected_ggT = expected_ggT.mean(0) - grads.unsqueeze(1).permute(1,0).mm(grads.unsqueeze(1))

        mask = False
        norm_expected_ggT = expected_ggT/ (torch.norm(expected_ggT) + 1e-8)
        masked_expected_ggT = None
        if mask:
            mask_gt = torch.gt(norm_expected_ggT, 1e-8)
            masked_expected_ggT = norm_expected_ggT * mask_gt.float()
        if not mask:
            masked_expected_ggT = norm_expected_ggT     

        fim = masked_expected_ggT
        fim_inverse = torch.pinverse(fim).detach()
        norm_fim_inverse = fim_inverse / (torch.norm(fim_inverse) + 1e-8)
        # print("mask: {} norm_ggT: {} norm_masked_ggT: {} fim_inverse: {} norm_fim_inverse: {}".format(mask, torch.norm(expected_ggT),\
        #                  torch.norm(masked_expected_ggT), torch.norm(fim_inverse), torch.norm(norm_fim_inverse))) 

        self.norm_ggT.append(torch.norm(expected_ggT).detach().cpu().data.numpy())
        self.norm_masked_ggT.append(torch.norm(masked_expected_ggT).detach().cpu().data.numpy())
        self.norm_fim_inverse.append(torch.norm(fim_inverse).detach().cpu().data.numpy())
        self.norm_norm_fim_inverse.append(torch.norm(norm_fim_inverse).detach().cpu().data.numpy())

        # set correct computation graph
        # if not first_order:
        #     self.context_params = self.context_params - step_size * grads
        # else:
        #     self.context_params = self.context_params - step_size * grads.detach()

        M = norm_fim_inverse.detach()
        if not first_order:
            self.context_params = self.context_params - step_size * M.mm(grads.unsqueeze(1)).squeeze(1)
        else:
            self.context_params = self.context_params - step_size * M.mm(grads.detach().unsqueeze(1)).squeeze(1)


        self.M_saved[str(self.i_saved)] = fim.detach().cpu().numpy()
        self.grad_saved[str(self.i_saved)] = grads.detach().cpu().numpy()
        self.i_saved = self.i_saved + 1
        
        return OrderedDict(self.named_parameters())


    def reset_context(self):
        self.context_params = torch.zeros(self.num_context_params, requires_grad=True).to(self.device)
        self.M_saved = OrderedDict()
        self.F_saved = OrderedDict()
        self.L_saved= OrderedDict()
        self.R_saved= OrderedDict()
        self.grad_saved = OrderedDict()
        self.i_saved = 0
        self.norm_ggT = []
        self.norm_masked_ggT = []
        self.norm_fim_inverse =[]
        self.norm_norm_fim_inverse =[]


class CustomCaviaMLPPolicy(Policy, nn.Module):
    """CAVIA network based on a multi-layer perceptron (MLP), with a
    `Normal` distribution output, with trainable standard deviation. This
    policy network can be used on tasks with continuous action spaces (eg.
    `HalfCheetahDir`).
    """

    def __init__(self, input_size, output_size, device, hidden_sizes=(), num_context_params=10,
                 nonlinearity=F.relu, init_std=1.0, min_std=1e-6):
        super(CustomCaviaMLPPolicy, self).__init__(input_size, output_size)
        self.input_size = input_size
        self.output_size = output_size
        self.device = device

        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.min_log_std = math.log(min_std)
        self.num_layers = len(hidden_sizes) + 1
        self.context_params = []

        layer_sizes = (input_size,) + hidden_sizes
        self.add_module('layer{0}'.format(1), nn.Linear(layer_sizes[0] + num_context_params, layer_sizes[1]))
        for i in range(2, self.num_layers):
            self.add_module('layer{0}'.format(i), nn.Linear(layer_sizes[i - 1], layer_sizes[i]))

        self.num_context_params = num_context_params
        self.context_params = torch.zeros(self.num_context_params, requires_grad=True).to(self.device)

        self.mu = nn.Linear(layer_sizes[-1], output_size)
        self.sigma = nn.Parameter(torch.Tensor(output_size))
        self.sigma.data.fill_(math.log(init_std))

        # self.M_layer = nn.Linear(self.num_context_params , self.num_context_params * self.num_context_params)
       
        # # convert context_params (+) grad_context_params to M-matrix (meta curvature)
        # self.M_layers = nn.ModuleList()
        # self.M_layers.append(nn.Linear(1 * num_context_params , 1* num_context_params))
        # self.M_layers.append(nn.Linear(1 * num_context_params , num_context_params * num_context_params))

        # self.M_layers.append(nn.Linear(1 * num_context_params , num_context_params * num_context_params))

        self.expected_wwT = None
        self.blend_ratio = 0.8
        self.frozen_identity = torch.eye(self.num_context_params, requires_grad=False).to(self.device)
        # self.frozen_identity.requires_grad = False
        self.identityL = None # nn.Parameter(torch.eye(self.num_context_params, requires_grad=True))
        self.identityR = None # nn.Parameter(torch.eye(self.num_context_params, requires_grad=True))

        # saved M_matrix for further visualization/ analysis
        self.M_saved = OrderedDict()
        self.F_saved = OrderedDict()
        self.L_saved= OrderedDict()
        self.R_saved= OrderedDict()
        self.grad_saved = OrderedDict()
        self.i_saved = 0
        self.g = None
        self.gcontext = None
        self.norm_ggT = []
        self.norm_masked_ggT = []
        self.norm_fim_inverse =[]
        self.norm_norm_fim_inverse =[]

        self.apply(weight_init)

    def forward(self, input, params=None):

        # if no parameters are given, use the standard ones
        if params is None:
            params = OrderedDict(self.named_parameters())

        # # concatenate context parameters to input
        # output = torch.cat((input, self.context_params.expand(input.shape[:-1] + self.context_params.shape)),
        #                    dim=len(input.shape) - 1)

        # concatenate context parameters to 
        context_params = self.context_params.unsqueeze(0).mm(self.frozen_identity).expand(input.shape[:-1] + self.context_params.shape)
        self.gcontext = context_params
        self.gcontext.retain_grad()

        output = torch.cat((input, context_params),
                           dim=len(input.shape) - 1)


        # forward through FC Layer
        for i in range(1, self.num_layers):
            output = F.linear(output, weight=params['layer{0}.weight'.format(i)],
                              bias=params['layer{0}.bias'.format(i)])
            if i == 1:
                # output.requires_grad=True
                self.g = output
                self.g.retain_grad()

        # last layer outputs mean; scale is a learned param independent of the input
        mu = F.linear(output, weight=params['mu.weight'], bias=params['mu.bias'])
        scale = torch.exp(torch.clamp(params['sigma'], min=self.min_log_std))

        return Normal(loc=mu, scale=scale)

    # def M_forward(self, grad_context_params):
    #     # print(grad_context_params.size())
    #     # print(self.context_params.size())
    #     # exit()
    #     # x = torch.cat((grad_context_params, self.context_params), dim=0)
    #     # print(x.size())
    #     # exit()
    #     x = grad_context_params
    #     for k in range(len(self.M_layers) - 1):
    #         x = F.relu(self.M_layers[k](x))
    #     y = self.M_layers[-1](x)
        
    #     return y

    def update_params(self, loss, step_size, first_order=False, params=None):
        """Apply one step of gradient descent on the loss function `loss`, with
        step-size `step_size`, and returns the updated parameters of the neural
        network.
        """
# ---------------------- M - matrix -----------------
        # take the gradient wrt the context params
        grads = torch.autograd.grad(loss, self.context_params, create_graph=not first_order)[0]
        # print("context_params.grad: {}".format(grads))
        b = self.context_params.size()
        # M = self.M_forward(grads)
        # M = M.view(*b,*b)
        # print("M.size : {}".format(M.size()))
        # print(grads.unsqueeze(1).size())

        # self.M_saved[str(self.i_saved)] = M.detach().cpu().numpy()
        # self.grad_saved[str(self.i_saved)] = grads.detach().cpu().numpy()
        # self.i_saved = self.i_saved + 1
        # # set correct computation graph
        # if not first_order:
        #     self.context_params = self.context_params - step_size * M.mm(grads.unsqueeze(1)).squeeze(1)
        # else:
        #     self.context_params = self.context_params - step_size * M.mm(grads.detach().unsqueeze(1)).squeeze(1)

        M = norm_fim_inverse.detach()
        # set correct computation graph
        if not first_order:
            self.context_params = self.context_params - step_size * M.mm(grads.unsqueeze(1)).squeeze(1)
        else:
            self.context_params = self.context_params - step_size * M.mm(grads.detach().unsqueeze(1)).squeeze(1)

        self.M_saved[str(self.i_saved)] = M.detach().cpu().numpy()

        self.F_saved[str(self.i_saved)] = fim.detach().cpu().numpy()
        self.grad_saved[str(self.i_saved)] = grads.detach().cpu().numpy()
        self.i_saved = self.i_saved + 1
        # exit()


        return OrderedDict(self.named_parameters())


    def reset_context(self):
        self.context_params = torch.zeros(self.num_context_params, requires_grad=True).to(self.device)
        self.M_saved= OrderedDict()
        self.F_saved= OrderedDict()
        self.L_saved= OrderedDict()
        self.R_saved= OrderedDict()
        self.grad_saved = OrderedDict()
        self.i_saved = 0
        self.norm_ggT = []
        self.norm_masked_ggT = []
        self.norm_fim_inverse =[]
        self.norm_norm_fim_inverse =[]

        



# class CustomCaviaMLPPolicy(Policy, nn.Module):
#     """CAVIA network based on a multi-layer perceptron (MLP), with a
#     `Normal` distribution output, with trainable standard deviation. This
#     policy network can be used on tasks with continuous action spaces (eg.
#     `HalfCheetahDir`).
#     """

#     def __init__(self, input_size, output_size, device, hidden_sizes=(), num_context_params=10,
#                  nonlinearity=F.relu, init_std=1.0, min_std=1e-6):
#         super(CustomCaviaMLPPolicy, self).__init__(input_size, output_size)
#         self.input_size = input_size
#         self.output_size = output_size
#         self.device = device

#         self.hidden_sizes = hidden_sizes
#         self.nonlinearity = nonlinearity
#         self.min_log_std = math.log(min_std)
#         self.num_layers = len(hidden_sizes) + 1
#         self.context_params = []

#         layer_sizes = (input_size,) + hidden_sizes
#         self.add_module('layer{0}'.format(1), nn.Linear(layer_sizes[0] + num_context_params, layer_sizes[1]))
#         for i in range(2, self.num_layers):
#             self.add_module('layer{0}'.format(i), nn.Linear(layer_sizes[i - 1], layer_sizes[i]))

#         self.num_context_params = num_context_params
#         self.context_params = torch.zeros(self.num_context_params, requires_grad=True).to(self.device)

#         self.mu = nn.Linear(layer_sizes[-1], output_size)
#         self.sigma = nn.Parameter(torch.Tensor(output_size))
#         self.sigma.data.fill_(math.log(init_std))

#         # self.M_layer = nn.Linear(self.num_context_params , self.num_context_params * self.num_context_params)
       
#         # # convert context_params (+) grad_context_params to M-matrix (meta curvature)
#         # self.M_layers = nn.ModuleList()
#         # self.M_layers.append(nn.Linear(1 * num_context_params , 1* num_context_params))
#         # self.M_layers.append(nn.Linear(1 * num_context_params , num_context_params * num_context_params))

#         # self.M_layers.append(nn.Linear(1 * num_context_params , num_context_params * num_context_params))

#         self.expected_wwT = None
#         self.blend_ratio = 0.8
#         self.frozen_identity = torch.eye(self.num_context_params, requires_grad=False).to(self.device)
#         # self.frozen_identity.requires_grad = False
#         self.identityL = None # nn.Parameter(torch.eye(self.num_context_params, requires_grad=True))
#         self.identityR = None # nn.Parameter(torch.eye(self.num_context_params, requires_grad=True))

#         # saved M_matrix for further visualization/ analysis
#         self.M_saved = OrderedDict()
#         self.F_saved = OrderedDict()
#         self.L_saved= OrderedDict()
#         self.R_saved= OrderedDict()
#         self.grad_saved = OrderedDict()
#         self.i_saved = 0
#         self.g = None
#         self.gcontext = None
#         self.norm_ggT = []
#         self.norm_masked_ggT = []
#         self.norm_fim_inverse =[]
#         self.norm_norm_fim_inverse =[]

#         self.apply(weight_init)

#     def forward(self, input, params=None):

#         # if no parameters are given, use the standard ones
#         if params is None:
#             params = OrderedDict(self.named_parameters())

#         # # concatenate context parameters to input
#         # output = torch.cat((input, self.context_params.expand(input.shape[:-1] + self.context_params.shape)),
#         #                    dim=len(input.shape) - 1)

#         # concatenate context parameters to 
#         context_params = self.context_params.unsqueeze(0).mm(self.frozen_identity).expand(input.shape[:-1] + self.context_params.shape)
#         self.gcontext = context_params
#         self.gcontext.retain_grad()

#         output = torch.cat((input, context_params),
#                            dim=len(input.shape) - 1)


#         # forward through FC Layer
#         for i in range(1, self.num_layers):
#             output = F.linear(output, weight=params['layer{0}.weight'.format(i)],
#                               bias=params['layer{0}.bias'.format(i)])
#             if i == 1:
#                 # output.requires_grad=True
#                 self.g = output
#                 self.g.retain_grad()

#         # last layer outputs mean; scale is a learned param independent of the input
#         mu = F.linear(output, weight=params['mu.weight'], bias=params['mu.bias'])
#         scale = torch.exp(torch.clamp(params['sigma'], min=self.min_log_std))

#         return Normal(loc=mu, scale=scale)

#     # def M_forward(self, grad_context_params):
#     #     # print(grad_context_params.size())
#     #     # print(self.context_params.size())
#     #     # exit()
#     #     # x = torch.cat((grad_context_params, self.context_params), dim=0)
#     #     # print(x.size())
#     #     # exit()
#     #     x = grad_context_params
#     #     for k in range(len(self.M_layers) - 1):
#     #         x = F.relu(self.M_layers[k](x))
#     #     y = self.M_layers[-1](x)
        
#     #     return y

#     def update_params(self, loss, step_size, first_order=False, params=None):
#         """Apply one step of gradient descent on the loss function `loss`, with
#         step-size `step_size`, and returns the updated parameters of the neural
#         network.
#         """
# # ---------------------- M - matrix -----------------
#         # take the gradient wrt the context params
#         grads = torch.autograd.grad(loss, self.context_params, create_graph=not first_order)[0]
#         # print("context_params.grad: {}".format(grads))
#         b = self.context_params.size()
#         # M = self.M_forward(grads)
#         # M = M.view(*b,*b)
#         # print("M.size : {}".format(M.size()))
#         # print(grads.unsqueeze(1).size())

#         # self.M_saved[str(self.i_saved)] = M.detach().cpu().numpy()
#         # self.grad_saved[str(self.i_saved)] = grads.detach().cpu().numpy()
#         # self.i_saved = self.i_saved + 1
#         # # set correct computation graph
#         # if not first_order:
#         #     self.context_params = self.context_params - step_size * M.mm(grads.unsqueeze(1)).squeeze(1)
#         # else:
#         #     self.context_params = self.context_params - step_size * M.mm(grads.detach().unsqueeze(1)).squeeze(1)

# # # --------------- Fisher (s)
#         if(self.gcontext.grad.dim() > 2):
#             m, n, c = self.gcontext.grad.size()
#             # print("self.gcontext.grad.size() : {}".format(self.gcontext.grad.size()))
#             g_reshaped = self.gcontext.grad.view(m*n,1,c)
#             expected_ggT = torch.bmm(g_reshaped.permute(0,2,1), g_reshaped).detach() 
#             # expected_ggT = torch.bmm(self.gcontext.grad.permute(0,2,1), self.gcontext.grad).detach()  #/ batch_size.to(self.device)
#             # print("expected_ggT,size:{}".format(expected_ggT.size()))
#             expected_ggT = expected_ggT.mean(0)- grads.unsqueeze(1).permute(1,0).mm(grads.unsqueeze(1))
#         else:
#             n, c = self.gcontext.grad.size()
#             g_reshaped = self.gcontext.grad.view(n,1,c)
#             expected_ggT = torch.bmm(g_reshaped.permute(0,2,1), g_reshaped).detach() 
#             expected_ggT = expected_ggT.mean(0)- grads.unsqueeze(1).permute(1,0).mm(grads.unsqueeze(1))
#             # expected_ggT = torch.matmul(self.gcontext.grad.permute(1,0), self.gcontext.grad) #/ batch_size.to(self.device)
#             # expected_ggT = expected_ggT.mean(0)
        
#         mask = False
#         # print("mask: {}".format(mask))
#         # print("norm_ggT: {}".format(torch.norm(expected_ggT)))
#         # print("expected_ggT.size: {}".format(expected_ggT.size()))
#         norm_expected_ggT = expected_ggT/ (torch.norm(expected_ggT) + 1e-8)
#         masked_expected_ggT = None
#         if mask:
#             mask_gt = torch.gt(norm_expected_ggT, 1e-8)
#             masked_expected_ggT = norm_expected_ggT * mask_gt.float()
#         # print("ggT gt(1e-8): {}".format(torch.gt(expected_ggT/ torch.norm(expected_ggT), 1e-8)))
#         if not mask:
#             masked_expected_ggT = norm_expected_ggT
#         # print("norm_ggT: {} norm_masked_ggT: {}".format(torch.norm(expected_ggT), torch.norm(masked_expected_ggT)))        



#         fim = masked_expected_ggT
#         fim_inverse = torch.inverse(fim).detach()
#         norm_fim_inverse = fim_inverse / (torch.norm(fim_inverse) + 1e-8)
#         # print("mask: {} norm_ggT: {} norm_masked_ggT: {} fim_inverse: {} norm_fim_inverse: {}".format(mask, torch.norm(expected_ggT),\
#                         #  torch.norm(masked_expected_ggT), torch.norm(fim_inverse), torch.norm(norm_fim_inverse))) 
#         # print("norm_fim_inverse: {}".format(norm_fim_inverse))

#         # M = self.identityL.mm(norm_fim_inverse.mm(self.identityR))
#         M = norm_fim_inverse.detach()
#         # print("M.mm(grads.unsqueeze(1)).squeeze(1).size: {}".format(M.mm(grads.unsqueeze(1)).squeeze(1).size()))
#         # set correct computation graph
#         if not first_order:
#             self.context_params = self.context_params - step_size * M.mm(grads.unsqueeze(1)).squeeze(1)
#         else:
#             self.context_params = self.context_params - step_size * M.mm(grads.detach().unsqueeze(1)).squeeze(1)

#         self.M_saved[str(self.i_saved)] = M.detach().cpu().numpy()
#         # self.L_saved[str(self.i_saved)] = self.identityL.detach().cpu().numpy()
#         # self.R_saved[str(self.i_saved)] = self.identityR.detach().cpu().numpy()
#         self.F_saved[str(self.i_saved)] = fim.detach().cpu().numpy()
#         self.grad_saved[str(self.i_saved)] = grads.detach().cpu().numpy()
#         self.i_saved = self.i_saved + 1
#         # exit()


#         return OrderedDict(self.named_parameters())


#     def reset_context(self):
#         self.context_params = torch.zeros(self.num_context_params, requires_grad=True).to(self.device)
#         self.M_saved= OrderedDict()
#         self.F_saved= OrderedDict()
#         self.L_saved= OrderedDict()
#         self.R_saved= OrderedDict()
#         self.grad_saved = OrderedDict()
#         self.i_saved = 0
#         self.norm_ggT = []
#         self.norm_masked_ggT = []
#         self.norm_fim_inverse =[]
#         self.norm_norm_fim_inverse =[]

        
# class CaviaMLPPolicy(Policy, nn.Module):
#     """CAVIA network based on a multi-layer perceptron (MLP), with a
#     `Normal` distribution output, with trainable standard deviation. This
#     policy network can be used on tasks with continuous action spaces (eg.
#     `HalfCheetahDir`).
#     """

#     def __init__(self, input_size, output_size, device, hidden_sizes=(), num_context_params=10,
#                  nonlinearity=F.relu, init_std=1.0, min_std=1e-6):
#         super(CaviaMLPPolicy, self).__init__(input_size, output_size)
#         self.input_size = input_size
#         self.output_size = output_size
#         self.device = device

#         self.hidden_sizes = hidden_sizes
#         self.nonlinearity = nonlinearity
#         self.min_log_std = math.log(min_std)
#         self.num_layers = len(hidden_sizes) + 1
#         self.context_params = []

#         layer_sizes = (input_size,) + hidden_sizes
#         self.add_module('layer{0}'.format(1), nn.Linear(layer_sizes[0] + num_context_params, layer_sizes[1]))
#         for i in range(2, self.num_layers):
#             self.add_module('layer{0}'.format(i), nn.Linear(layer_sizes[i - 1], layer_sizes[i]))

#         self.num_context_params = num_context_params
#         self.context_params = torch.zeros(self.num_context_params, requires_grad=True).to(self.device)

#         self.mu = nn.Linear(layer_sizes[-1], output_size)
#         self.sigma = nn.Parameter(torch.Tensor(output_size))
#         self.sigma.data.fill_(math.log(init_std))
#         self.apply(weight_init)

#     def forward(self, input, params=None):

#         # if no parameters are given, use the standard ones
#         if params is None:
#             params = OrderedDict(self.named_parameters())

#         # concatenate context parameters to input
#         output = torch.cat((input, self.context_params.expand(input.shape[:-1] + self.context_params.shape)),
#                            dim=len(input.shape) - 1)

#         # forward through FC Layer
#         for i in range(1, self.num_layers):
#             output = F.linear(output, weight=params['layer{0}.weight'.format(i)],
#                               bias=params['layer{0}.bias'.format(i)])

#         # last layer outputs mean; scale is a learned param independent of the input
#         mu = F.linear(output, weight=params['mu.weight'], bias=params['mu.bias'])
#         scale = torch.exp(torch.clamp(params['sigma'], min=self.min_log_std))

#         return Normal(loc=mu, scale=scale)

#     def update_params(self, loss, step_size, first_order=False, params=None):
#         """Apply one step of gradient descent on the loss function `loss`, with
#         step-size `step_size`, and returns the updated parameters of the neural
#         network.
#         """

#         # take the gradient wrt the context params
#         grads = torch.autograd.grad(loss, self.context_params, create_graph=not first_order)[0]

#         # set correct computation graph
#         if not first_order:
#             self.context_params = self.context_params - step_size * grads
#         else:
#             self.context_params = self.context_params - step_size * grads.detach()

#         return OrderedDict(self.named_parameters())

#     def reset_context(self):
#         self.context_params = torch.zeros(self.num_context_params, requires_grad=True).to(self.device)
