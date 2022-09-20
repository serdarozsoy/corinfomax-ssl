import torch.optim as optim
from LRscheduler import LinearWarmupCosineAnnealingLR

def make_optimizer(model, args, pretrain):
    if pretrain:
        optim_name = args.pre_optimizer
        learning_rate = args.learning_rate
        momentum = args.momentum
        w_decay = args.w_decay
        params = model.parameters()
    else:
        optim_name = args.lin_optimizer
        learning_rate = args.lin_learning_rate
        momentum = args.lin_momentum
        w_decay = args.lin_w_decay
        params = model.parameters()

    if optim_name == "SGD":
        optimizer = optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=w_decay)
    elif optim_name == "Adam":
        optimizer = optim.Adam(params, lr=learning_rate)
    elif optim_name =="AdamW":
        optimizer = optim.AdamW(params, lr=learning_rate, weight_decay=w_decay)
    else:
        raise Exception("selected optimizer is not in list")
    return optimizer


    """
    if ext_opt == "LARS":
        optimizer = LARSWrapper(
                    optimizer,
                    eta=eta_lars,
                    clip=grad_clip_lars,
                    exclude_bias_n_norm=exclude_bias_n_norm,
                )
    """

    """
    elif sched_name == "lin_warmup_cos":
        scheduler = LinearWarmupCosineAnnealingLR(
                        optimizer,
                        warmup_epochs=warmup_epochs,
                        max_epochs=max_epochs,
                        warmup_start_lr=warmup_start_lr,
                        eta_min=min_lr,
                    )
    """


def make_scheduler(optimizer, args, pretrain):
    if pretrain:
        sched_name = args.pre_scheduler
        plateau_factor = args.plateau_factor
        plateau_patience = args.plateau_patience
        plateau_threshold = args.plateau_threshold
        epochs = args.epochs
        min_lr = args.min_lr
        step_size = args.step_size
        step_gamma = args.step_gamma
        step_list = args.step_list
        step_gamma = args.step_gamma
    else:
        sched_name = args.lin_scheduler
        plateau_factor = args.lin_plateau_factor
        plateau_patience = args.lin_plateau_patience
        plateau_threshold = args.lin_plateau_threshold
        epochs = args.lin_epochs
        min_lr = args.lin_min_lr
        step_size = args.lin_step_size
        step_gamma = args.lin_step_gamma
        step_list = args.lin_step_list
        step_gamma = args.lin_step_gamma

    if sched_name == "None":
        scheduler = "None"
    elif sched_name == "reduce_plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=plateau_factor, patience=plateau_patience, threshold=plateau_threshold)
    elif sched_name =="cos":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=min_lr)
    elif sched_name =="step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size, gamma=step_gamma) 
    elif sched_name =="multi_step":
        step_list  = [ int(x) for x in step_list ]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=step_list, gamma=step_gamma) 
    elif sched_name == "cos_warm_res":
        #scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, step_size)
        scheduler =optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.lin_min_lr, max_lr=args.lin_learning_rate, step_size_up=10, mode="triangular2")
    elif sched_name == "lin_warmup_cos":
        scheduler = LinearWarmupCosineAnnealingLR(
                        optimizer,
                        warmup_epochs=args.warmup_epochs,
                        max_epochs=epochs,
                        warmup_start_lr=args.warmup_start_lr,
                        eta_min=min_lr,
                    )
    else:
        raise Exception("selected scheduler is not in list")
    return scheduler

    #scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.95, patience=10, threshold=0.01)
    #scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=10, threshold=0.002)
    #scheduler = OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=len(train_loader), epochs=epochs)
