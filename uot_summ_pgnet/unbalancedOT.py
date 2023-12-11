# -*- coding: utf-8 -*-
#pylint: skip-file

import torch
import time
# import time


def compute_sinkhorn_loss(cost_matrices,
                          masked_scaling_factors,
                          src_mask=None, tgt_mask=None,
                          epsilon=1.0, tau=1.0,
                          block_padding=5.0):
    # print("src_mask", src_mask)
    # print("tgt_mask", tgt_mask)
    # print("masked_scaling_factors", masked_scaling_factors)

    x_y_mask = src_mask.unsqueeze(2) * tgt_mask.unsqueeze(1)

    # print("block_padding", block_padding)
    # torch.set_printoptions(profile="full")
    # print("x_y_mask", x_y_mask.size())
    # print("x_y_mask", x_y_mask)
    # print("cost_matrices 1", cost_matrices)

    # Wasserstein cost function
    # c_m = (1 - cost_matrices) * x_y_mask + (1.0 - x_y_mask) * 2.0
    cost_matrices = cost_matrices * x_y_mask + (1.0 - x_y_mask) * block_padding
    # print("cost_matrices 2", cost_matrices)

    # positivetime = time.time()
    plan_positive = sinkhorn_batched(cost_matrices, masked_scaling_factors, tgt_mask,
                                     is_balanced=False, epsilon=epsilon, tau=tau)
    # print("positive time:", time.time() - positivetime)

    # todo masked?
    plan_positive = plan_positive * x_y_mask
    # print("plan_positive", plan_positive.size())
    # print("plan_positive", plan_positive)

    ot_cost_positive = torch.sum(plan_positive * cost_matrices, (1, 2))  # todo masked?

    src_marginal = torch.sum(plan_positive, 2, keepdim=False)  # x marginal

    # masked_rest_factors = torch.max(1 - masked_scaling_factors, torch.zeros_like(masked_scaling_factors)) * src_mask
    # normalized_masked_rest_factors = masked_rest_factors / torch.sum(masked_rest_factors, 1, keepdim=True)  # normalize
    # normalized_tgt_mask = tgt_mask / torch.sum(tgt_mask, 1, keepdim=True)  # normalize

    # print("normalized_masked_rest_factors", normalized_masked_rest_factors)
    # print("normalized_tgt_mask", normalized_tgt_mask)
    # negativetime = time.time()
    # plan_negative = sinkhorn_batched(cost_matrices, normalized_masked_rest_factors, normalized_tgt_mask,
    #                                  is_balanced=True, epsilon=epsilon)
    # print("negative time:", time.time() - negativetime)

    # plan_negative = plan_negative * x_y_mask
    # print("plan_negative", plan_negative)

    # ot_cost_negative = torch.sum(plan_negative * cost_matrices, (1, 2))  # todo masked?

    # print("ot_cost_positive", ot_cost_positive)
    # print("src_marginal", src_marginal)
    # print("ot_cost_negative", ot_cost_negative)

    # print("^^^^^^^^^^^^^^^^^^^^")
    # return plan_positive, ot_cost_positive, src_marginal, ot_cost_negative
    return plan_positive, ot_cost_positive, src_marginal


# batched version of unbalanced sinkhorn
def sinkhorn_batched(cost_matrix,
                     mu, nu,
                     is_balanced=False,
                     epsilon=1.0, tau=1.0, max_num_iter=300):
    # print("epsilon", epsilon)
    # print("tau", tau)

    # print("mu", mu)
    # print("nu", nu)

    mu = mu + 1e-37  # add 1e-37 to avoid nan and inf
    nu = nu + 1e-37

    def M(u, v):
        # "Modified cost for logarithmic updates"
        # "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-cost_matrix + u.unsqueeze(2) + v.unsqueeze(1)) / epsilon

    def lse(A):
        # "log-sum-exp"
        return torch.log(torch.exp(A).sum(2, keepdim=True) + 1e-6)  # add 10^-6 to prevent NaN

    # Actual Sinkhorn loop ......................................................................
    u, v, err_1 = 0.0 * mu, 0.0 * nu, 0.0
    for i in range(max_num_iter):
        u1 = u  # useful to check the update
        if is_balanced:
            u = epsilon * (torch.log(mu) - lse(M(u, v)).squeeze()) + u  # balanced
        else:
            u = (epsilon * (torch.log(mu) - lse(M(u, v)).squeeze()) + u) * tau / (tau + epsilon)  # new 4 unbalanced
        v = epsilon * (torch.log(nu) - lse(M(u, v).transpose(1, 2)).squeeze()) + v

        # print("torch.exp(M(u, v))", torch.exp(M(u, v)))
        # err = (torch.exp(M(u, v)).sum(1, keepdim=False) - nu).abs().sum(dim=1).max().item()
        err_1 = (u - u1).abs().sum(dim=1).max().item()
        if err_1 < 1e-3:  # todo The termination threshold should be small enough!!!
            break
    # print("iteration", i)
    # print("err_1", err_1)
    pi = torch.exp(M(u, v))  # Transport plan pi = diag(a)*K*diag(b) todo: check

    if torch.isnan(pi).sum().item() > 0:  # handles NaN error
        print("Error! In this batch, certain sinkhorn optimization problem failed! Simply set the value to 0.")
        pi = torch.where(torch.isnan(pi), torch.zeros_like(pi), pi)

    # print("x marginal", torch.sum(pi, 2, keepdim=False))
    # print("y marginal", torch.sum(pi, 1, keepdim=False))
    # print("Transport plan pi", pi)

    # print("---------------")
    return pi


######################### unused #########################################
######################### unused #########################################
######################### unused #########################################
######################### unused #########################################
def cost_matrix_cosine_batched_gpu(x, y):
    # # Returns the matrix of $|x_i-y_j|^p$.
    batch_size = x.size(0)
    c_m = []
    cos = torch.nn.CosineSimilarity(dim=2, eps=1e-8)

    for i in range(batch_size):
        # c_m_i = 1 - cos(x[i].unsqueeze(1).repeat(1, y[i].size(0), 1),
        #                 y[i].unsqueeze(0).repeat(x[i].size(0), 1, 1))
        # print("c_m_i", c_m_i)
        # print("---------------")
        # c_m += [c_m_i]
        c_m += [1 - cos(x[i].unsqueeze(1).repeat(1, y[i].size(0), 1),
                        y[i].unsqueeze(0).repeat(x[i].size(0), 1, 1))]

    c_m = torch.stack(c_m).view(batch_size, *c_m[0].size())

    return c_m


def cost_matrix_batched_gpu(x, y, p=2):
    # # Returns the matrix of $|x_i-y_j|^p$.
    batch_size = x.size(0)
    c_m = []
    for i in range(batch_size):
        c_m += [torch.sum((torch.abs(x[i].unsqueeze(1) - y[i].unsqueeze(0))) ** p, 2)]

    c_m = torch.stack(c_m).view(batch_size, *c_m[0].size())

    return c_m


def cost_matrix_batched_gpu_normalized(x, y, x_len, y_len, p=2, scale_factor=10000.0):
    # # Returns the matrix of $|x_i-y_j|^p$.
    batch_size = x.size(0)
    c_m = []
    for i in range(batch_size):
        c_m_i = torch.sum((torch.abs(x[i].unsqueeze(1) - y[i].unsqueeze(0))) ** p, 2)
        # print("c_m_i", c_m_i)
        # print("scaled c_m_i", (c_m_i * scale_factor) / (c_m_i[:x_len[i], :y_len[i]].sum()))
        c_m += [(c_m_i * scale_factor) / (c_m_i[:x_len[i], :y_len[i]].sum())]  # normalize

    c_m = torch.stack(c_m).view(batch_size, *c_m[0].size())
    return c_m


def cost_matrix_batched_cpu(x, y, p=2, device=None):
    # # Returns the matrix of $|x_i-y_j|^p$.
    x_col = x.unsqueeze(2).to(torch.device("cpu"))
    y_lin = y.unsqueeze(1).to(torch.device("cpu"))
    c_m = torch.sum((torch.abs(x_col - y_lin)) ** p, 3)

    return c_m.to(device)  # torch.device("cuda")


'''
        torch.set_printoptions(profile="full")
        print("mu", mu)
        print("nu", nu)
        print("u", u)
        print("v", v)
        print("pi", pi)

        print(cost_matrix)
        torch.set_printoptions(profile="default")
        
    print("c_m", c_m.size())
    print("c_m", c_m)

    plan_positive = sinkhorn_batched(c_m, masked_scaling_factors, tgt_mask,
                                     is_balanced=False, epsilon=epsilon, tau=tau)
    print("plan_positive", plan_positive)
    ot_cost_positive = torch.sum(plan_positive * c_m, (1, 2))  # todo masked?
    src_marginal = torch.sum(plan_positive, 2, keepdim=False)  # x marginal
    print("src_marginal", src_marginal)

    rest_factors = 1 - masked_scaling_factors
    print("rest_factors", rest_factors)
    masked_rest_factors = rest_factors * src_mask
    print("masked_rest_factors", masked_rest_factors)
    normalized_masked_rest_factors = masked_rest_factors / torch.sum(masked_rest_factors, 1, keepdim=True) # normalize
    print("normalized_masked_rest_factors", torch.sum(normalized_masked_rest_factors, 1, keepdim=True))
    normalized_tgt_mask = tgt_mask / torch.sum(tgt_mask, 1, keepdim=True)  # normalize
    print("normalized_tgt_mask", normalized_tgt_mask)

    plan_negative = sinkhorn_batched(c_m, normalized_masked_rest_factors, normalized_tgt_mask,
                                     is_balanced=True, epsilon=epsilon)
    print("plan_negative", torch.sum(plan_negative, 1, keepdim=False))
    print("plan_negative", plan_negative)

    ot_cost_negative = torch.sum(plan_negative * c_m, (1, 2))  # todo masked?

    print("^^^^^^^^^^^^^^^^^^^^")
    
    c_m = c_m / torch.sum(c_m, (1, 2),  keepdim=True)

    # print("x_mask", x_mask)
    # print("y_mask", y_mask)
    # print("x_y_mask", x_y_mask)
    # print("x_mask", x_mask.size())
    # print("y_mask", y_mask.size())
    # print("x_y_mask", x_y_mask.size())
    # print("c_m", c_m.size())
    # print("c_m", c_m)

    # c_m_sum = torch.sum(c_m, (1, 2),  keepdim=True)
    # print("c_m_sum", c_m_sum.size())
    # print("c_m_sum", c_m_sum)

    # print("---------------")
    
    # handles NaN error
    if torch.isnan(cost).sum().item() > 0:
        print("Error! In this batch, certain sinkhorn optimization problem failed! Simply set the value to 0.")
        return torch.where(torch.isnan(cost), torch.zeros_like(cost), cost)
    else:
        return cost

    print("mu", mu.size())
    print("nu", nu.size())
        print("i", i)
        print("err, marginal difference", err)
        print("err_1, variable difference", err_1)
        print("---------------")
    print("Transport plan pi", pi.size())
    print("Transport plan pi", pi)
    # todo: whether the mass = y side -> false
    print("Transport plan pi", pi.sum((1, 2), keepdim=True))

    print("cost", cost)
    print("x marginal", torch.sum(pi, 2, keepdim=False).size())
    print("x marginal", torch.sum(pi, 2, keepdim=False))
    print("y marginal", torch.sum(pi, 1, keepdim=False))
    
    def cost_matrix_cosine_batched_gpu(x, y, p=2):
    # # Returns the matrix of $|x_i-y_j|^p$.
    batch_size = x.size(0)
    c_m = []
    cos = torch.nn.CosineSimilarity(dim=2, eps=1e-8)

    for i in range(batch_size):
        print("x[i]", x[i].size(0))
        print("y[i]", y[i].size(0))

        print("x[i]", x[i].unsqueeze(1).size())
        print("y[i]", y[i].unsqueeze(0).size())

        print("x[i]", x[i].unsqueeze(1).repeat(1, y[i].size(0), 1).size())
        print("y[i]", y[i].unsqueeze(0).repeat(x[i].size(0), 1, 1).size())

        print("x[i] - y[i]", (x[i].unsqueeze(1) - y[i].unsqueeze(0)).size())

        print("cos", cos(x[i].unsqueeze(1).repeat(1, y[i].size(0), 1),
                         y[i].unsqueeze(0).repeat(x[i].size(0), 1, 1)).size())

        print("cos", (1 - cos(x[i].unsqueeze(1).repeat(1, y[i].size(0), 1),
                      y[i].unsqueeze(0).repeat(x[i].size(0), 1, 1))).size())

        print("---------------")
        c_m += [torch.sum((torch.abs(x[i].unsqueeze(1) - y[i].unsqueeze(0))) ** p, 2)]

    c_m = torch.stack(c_m).view(batch_size, *c_m[0].size())

    return c_m
    
        print("self._epsilo", epsilon)
    print("self._tau_sinkhorn", tau)
'''
