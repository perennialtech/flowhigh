from torch.optim import Adam, AdamW


def separate_weight_decayable_params(params):
    wd_params, no_wd_params = [], []

    for param in params:
        param_list = no_wd_params if param.ndim < 2 else wd_params
        param_list.append(param)

    return wd_params, no_wd_params


def get_optimizer(
    params,
    lr=1e-4,
    wd=1e-2,
    betas=(0.9, 0.99),
    eps=1e-8,
    filter_by_requires_grad=True,
    group_wd_params=True,
):
    params = list(params)

    if filter_by_requires_grad:
        params = [param for param in params if param.requires_grad]

    if wd <= 0:
        return Adam(params, lr=lr, betas=betas, eps=eps)

    if group_wd_params:
        wd_params, no_wd_params = separate_weight_decayable_params(params)
        params = [
            {"params": wd_params},
            {"params": no_wd_params, "weight_decay": 0},
        ]

    return AdamW(params, lr=lr, weight_decay=wd, betas=betas, eps=eps)
