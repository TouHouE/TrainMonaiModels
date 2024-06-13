import torch
import wandb
from torch import nn
from torch import optim
from tqdm.auto import tqdm
import factory
from monai.losses import DiceCELoss
import os
import datetime as dt
import accelerate as ACC
accelerator: ACC.Accelerator
# import wandb


def get_now(t0):
    # t0 = dt.datetime.now()
    return f'{t0:%Y-%m-%d %H:%M:%S}'


def start_training(config):
    global accelerator
    accelerator = ACC.Accelerator(mixed_precision='bf16')
    model: nn.Module = factory.get_model(config)
    optimizer: optim.Optimizer = factory.get_optimizer(model, config)
    loss_func = DiceCELoss(sigmoid=True)
    loader_map = factory.get_loader(config)
    model, optimizer = accelerator.prepare(model, optimizer)
    for key, value in loader_map.items():
        loader_map[key] = accelerator.prepare(value)
    tot_epoch = config['epoch']
    config['rank'] = 1 - int(accelerator.is_main_process)
    is_rank0 = getattr(config, 'rank', 0) == 0

    run = None
    if 'wandb' in config['logs']['logger'] and is_rank0:
        import wandb
        run = wandb.init(
            project=config['logs']['project'],
            dir=config['logs']['exp_dir'],
            name=config['logs']['exp_dir']
        )

    best_vdloss = 1e+10

    for epoch in range(tot_epoch):
        start_t = dt.datetime.now()
        tloss = train_epoch(model, loader_map['train'], optimizer, loss_func, epoch, run, config)
        end_train = dt.datetime.now()
        vdloss = do_val(model, loader_map['val'], loss_func, epoch, config)
        end_val = dt.datetime.now()

        if run is not None and is_rank0:
            train_dur = end_train - start_t
            val_dur = end_val - end_train
            total = end_val - start_t
            run.log({
                'train step time': train_dur.microseconds / 100000,
                'val step time': val_dur.microseconds / 100000,
                'total epoch time': total.microseconds / 100000,
                'val loss(dice)': vdloss,
            })


        if best_vdloss > vdloss and is_rank0:
            torch.save(accelerator.unwrap_model(model).state_dict(), os.path.join(config['logs']['exp_dir'], 'best-model.pt'))
            best_vdloss = vdloss


def train_epoch(model: nn.Module, loader, optimizer:optim.Optimizer, loss_func: nn.Module, epoch: int, wandb_logger, config):
    lcfg: dict = config['logs']
    bar: enumerate | tqdm = enumerate(loader)
    if 'tqdm' in lcfg['logger']:
        bar = tqdm(bar, total=len(loader), desc=f'Epoch:[{epoch}/{config["epoch"]}]|')
    epoch_loss = .0
    model.train()

    for idx, pack in bar:
        image = pack['image']
        label = pack['label']

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        with accelerator.autocast():
            pred = model(image)
            loss = loss_func(pred, label)
        accelerator.backward(loss)
        # loss.backward()
        optimizer.step()

        iter_loss = loss.item()
        epoch_loss += iter_loss

        if isinstance(bar, tqdm):
            bar.set_postfix({
                'iter loss': iter_loss,
                'epoch loss': epoch_loss / (idx + 1)
            })

    optimizer.zero_grad()
    return epoch_loss

@torch.no_grad()
def do_val(model: nn.Module, loader, loss_func, epoch: int, config):
    model.eval()
    lcfg: dict = config['logs']
    bar: enumerate | tqdm = enumerate(loader)
    if 'tqdm' in lcfg['logger']:
        bar = tqdm(bar, total=len(loader), desc='validation')
    val_loss = .0

    for idx, pack in bar:
        image = pack['image']
        label = pack['label']

        pred = model(image)
        iter_loss = loss_func(pred, label)
        val_loss += iter_loss.item()

        if isinstance(bar, tqdm):
            bar.set_postfix({
                'iter loss(DSC)': iter_loss.item(),
                'val loss(DSC)': val_loss / (idx + 1)
            })
    return val_loss
