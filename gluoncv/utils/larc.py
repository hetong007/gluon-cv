"""LARC gradient rescaling and clipping with SGD or NAG"""
from mxnet.optimizer import SGD

class SGD_LARC(optimizer.SGD):
    r"""LARC gradient rescaling and clippint with SGD"""
    def __init__(self, trust_coefficient=0.02, clip=True, eps=1e-8, **kwargs):
        super(SGD_LARC, self).__init__(**kwargs)
        self.trust_coefficient = trust_coefficient
        self.clip = clip
        self.eps = eps

    def _update_impl(self, index, weight, grad, state, multi_precision=False):
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)

        weight_norm = weight.norm().asscalar()
        grad_norm = grad.norm().asscalar()

        adaptive_lr = 1.0
        if weight_norm != 0 and grad_norm != 0:
            denom = grad_norm + param_norm * wd + self.eps
            adaptive_lr = self.trust_coefficient * weight_norm / denom
            if self.clip:
                adaptive_lr = min(adaptive_lr/lr, 1)

        kwargs = {'rescale_grad': self.rescale_grad * adaptive_lr}
        if self.momentum > 0:
            kwargs['momentum'] = self.momentum
        if self.clip_gradient:
            kwargs['clip_gradient'] = self.clip_gradient

        if not multi_precision:
            if state is not None:
                sgd_mom_update(weight, grad, state, out=weight,
                               lazy_update=self.lazy_update, lr=lr, wd=wd, **kwargs)
            else:
                sgd_update(weight, grad, out=weight, lazy_update=self.lazy_update,
                           lr=lr, wd=wd, **kwargs)
        else:
            if state[0] is not None:
                mp_sgd_mom_update(weight, grad, state[0], state[1], out=weight,
                                  lr=lr, wd=wd, **kwargs)
            else:
                mp_sgd_update(weight, grad, state[1], out=weight,
                              lr=lr, wd=wd, **kwargs)
