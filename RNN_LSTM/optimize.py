# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2016-11-06 21:04:19
# @Last Modified by:   AaronLai
# @Last Modified time: 2016-11-06 21:24:52

import theano.tensor as T
from theano.ifelse import ifelse


def sgd(parameters, grads, lr, minibatch, batchsize, auxis, caches):
    updates = []
    update_batch = ifelse(T.lt(minibatch, batchsize - 1), 0, 1)

    for ix in range(len(grads)):
        # update parameters if reaching batchsize
        move = -(lr / batchsize) * (caches[ix] + grads[ix])
        updates.append((parameters[ix], parameters[ix] + move * update_batch))
        new_cache = (caches[ix] + grads[ix]) * (1 - update_batch)
        updates.append((caches[ix], new_cache))

    return updates


def momentum(parameters, grads, lr, minibatch, batchsize,
             momentum, caches, moment=0.95):
    """theano update, optimized by Momentum"""
    updates = []
    update_batch = ifelse(T.lt(minibatch, batchsize - 1), 0, 1)

    for ix in range(len(grads)):
        move = - (lr / batchsize) * (grads[ix] + caches[ix])
        direction = moment * momentum[ix] + move

        # update parameters if reaching batchsize
        new_param = parameters[ix] + direction * update_batch
        updates.append((parameters[ix], new_param))

        # remember the move if updating parameters
        new_mom = momentum[ix] * (1 - update_batch) + direction * update_batch
        updates.append((momentum[ix], new_mom))

        # accumulate gradients if not reaching batchsize
        new_cache = (caches[ix] + grads[ix]) * (1 - update_batch)
        updates.append((caches[ix], new_cache))

    return updates


def NAG(parameters, grads, lr, minibatch, batchsize,
        real_pos, caches, moment=0.95):
    """theano update, optimized by NAG"""
    updates = []
    update_batch = ifelse(T.lt(minibatch, batchsize - 1), 0, 1)

    for ix in range(len(grads)):
        move = -(lr / batchsize) * (caches[ix] + grads[ix])
        real = parameters[ix] + move
        spy = real + moment * (real - real_pos[ix])

        # update parameters to spy position if reaching batchsize
        new_param = spy * update_batch + parameters[ix] * (1 - update_batch)
        updates.append((parameters[ix], new_param))

        # remember the real position if moved parameters
        new_realpos = real * update_batch + real_pos[ix] * (1 - update_batch)
        updates.append((real_pos[ix], new_realpos))

        # accumulate gradients if not reaching batchsize
        new_cache = (caches[ix] + grads[ix]) * (1 - update_batch)
        updates.append((caches[ix], new_cache))

    return updates


def RMSProp(parameters, grads, lr, minibatch, batchsize,
            sigma_square, caches, alpha=0.9, const=1e-2):
    """theano update, optimized by RMSProp"""
    updates = []
    update_batch = ifelse(T.lt(minibatch, batchsize - 1), 0, 1)

    for ix in range(len(grads)):
        move = (grads[ix] + caches[ix]) / batchsize
        factor = sigma_square[ix] * alpha + (1 - alpha) * (move**2)
        step = -lr * move / (T.sqrt(factor) + const)

        # update parameters to spy position if reaching batchsize
        new_param = (parameters[ix] + step) * update_batch
        new_param += parameters[ix] * (1 - update_batch)
        updates.append((parameters[ix], new_param))

        # remember the scaling factors if reaching batchsize
        new_sig = factor * update_batch + sigma_square[ix] * (1 - update_batch)
        updates.append((sigma_square[ix], new_sig))

        # accumulate gradients if not reaching batchsize
        new_cache = (caches[ix] + grads[ix]) * (1 - update_batch)
        updates.append((caches[ix], new_cache))

    return updates
