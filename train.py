import os
import argparse
import yaml
import torch
import time
from tqdm import tqdm
import numpy as np
from transforms import get_transforms
from loss import dice_loss, fitness_test
from torchvision import models, datasets, transforms
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW, lr_scheduler
from pathlib import Path
from transforms import get_transforms
from utils.general import print_args, LOGGER, colorstr, one_cycle
from utils.torch_utils import select_device, de_parallel
from copy import deepcopy
from datetime import datetime

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]


def train(hyp, opt, device):
    epochs, batch_size, resume, weights = opt.epochs, opt.batch_size, opt.resume, opt.weights


    # Hyperparameters
    with open(hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    # TODO: Loggers
        
    
    # Model
    cuda = device.type != 'cpu'
    model = models.segmentation.deeplabv3_resnet101(pretrained=False, progress=True, num_classes=20).to(device)

    # Resume (load checkpoint and apply it to the model)
    if resume:
        ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
        LOGGER.info(colorstr('yellow', ('Resuming training from %s saved at %s (last epoch %d)') 
                                        % (weights, ckpt['date'], ckpt['epoch'])))
        model.load_state_dict(ckpt['model'].float().state_dict(), strict=False)  # load

    # Data-Parallel mode
    if cuda:
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
    
  
    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    LOGGER.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g1.append(v.weight)

    if opt.optimizer == 'Adam':
        optimizer = Adam(g0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    elif opt.optimizer == 'AdamW':
        optimizer = AdamW(g0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = SGD(g0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    
    optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})  # add g1 with weight_decay
    optimizer.add_param_group({'params': g2})  # add g2 (biases)
    LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
                f"{len(g0)} weight (no decay), {len(g1)} weight, {len(g2)} bias")
    del g0, g1, g2

    # Scheduler
    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if resume:
        start_epoch = ckpt['epoch'] + 1
        optimizer.load_state_dict(ckpt['optimizer'])
        del ckpt
        


    # Dataset
    img_transforms = get_transforms(True)
    train_dataset = datasets.Cityscapes(root='./data/cityscapes', split = 'train', transforms=img_transforms)
    val_dataset = datasets.Cityscapes(root='./data/cityscapes', split = 'val')

    # Data Loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, drop_last=True, pin_memory=True)

    num_batches = len(train_loader)
    it = iter(train_loader)
    x, y = it.next()

    # Start training
    t0 = time.time()
    num_warmup = max(round(hyp['warmup_epochs'] * num_batches), 1000)  # number of warmup iterations
    last_opt_step = -1
    scaler = torch.cuda.amp.GradScaler()

    
      
    for epoch in range(start_epoch, epochs):
        model.train()

        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%10s' * 3) % ('Epoch', 'gpu_mem', 'loss'))
        pbar = tqdm(pbar, total=num_batches, bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}')  # progressive bar
        optimizer.zero_grad()
        for i, (imgs, targets) in pbar:
            num_iters = i + num_batches * epoch   # number of integrated batches
            imgs = imgs.to(device, non_blocking=True).float()

            # warmup
            if num_iters <= num_warmup:
                xi = [0, num_warmup]
                accumulate = max(1, np.interp(num_iters, xi, [1, num_batches/batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = np.interp(num_iters, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(num_iters, xi, [hyp['warmup_momentum'], hyp['momentum']])
                
            # Forward
            with torch.cuda.amp.autocast(enabled=cuda):
                preds = model(imgs)
                loss = dice_loss(targets.to(device), preds['out'])
            
            # Backward
            scaler.scale(loss).backward()

            # Optimizer
            if num_iters - last_opt_step >= accumulate:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                last_opt_step = num_iters


            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
            pbar.set_description(('%10s' * 2 + ' %10.4f') % (f'{epoch}/{epochs - 1}', mem, loss.item()))

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]   # for loggers
        scheduler.step()

        # Save model
        ckpt = {'model': deepcopy(de_parallel(model)).half(), 'optimizer': optimizer.state_dict(), \
                'epoch': epoch, 'opt': opt, 'loss': loss.item(), \
                'date': datetime.now().isoformat()}
        torch.save(ckpt, weights)



def main(opt):
    print_args(FILE.stem, opt)
    # Resume
    if opt.resume: # resume an interrupted run
        ckpt_file = opt.resume if isinstance(opt.resume, str) else 'weights.pt'  # specified or most recent path
        assert os.path.isfile(ckpt_file), 'ERROR: --resume checkpoint does not exist'
        opt.weights, opt.resume = ckpt_file, True  # reinstate
        
        

    device = select_device(opt.device, batch_size=opt.batch_size)
    train(opt.hyp, opt, device)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'weights.pt', help='initial weights path')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)