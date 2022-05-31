import torch
from torch import optim


def build_lr_scheduler(cfg,
                       args,
                       name='step',
                       optimizer=None,
                       resume=None):
    print('==============================')
    print('Lr Scheduler: {}'.format(name))

    if name == 'step':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, 
            milestones=cfg['epoch'][args.schedule]['lr_epoch']
            )

    if resume is not None:
        print('keep training: ', resume)
        checkpoint = torch.load(resume)
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("lr_scheduler")
        lr_scheduler.load_state_dict(checkpoint_state_dict)
                        
                                
    return optimizer
