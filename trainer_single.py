import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import Gaze360
import model

import cv2
import argparse
import os
import yaml
import tqdm
import logging
from time import gmtime, strftime
from shutil import copy

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        optimizers: torch.optim.Optimizer,
        loss,
        log_dir,
        gpu_id: int,
        save_every: int, 
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(f"cuda:{gpu_id}")
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.ge_optimizer = optimizers['ge_optimizer']
        self.ga_optimizer = optimizers['ga_optimizer']
        self.de_optimizer = optimizers['de_optimizer']
        self.ge_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.ge_optimizer, patience=10)
        self.ga_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.ga_optimizer, patience=10)
        self.de_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.de_optimizer, patience=10)
        self.geloss_op = loss["geloss_op"]
        self.deloss_op =  loss["deloss_op"]
        self.log_dir = log_dir
        self.save_every = save_every
        self.epoch = 0
        self.iter = 0
        self.train_loss = [None]
        self.eval_loss = [None]

    def _save_checkpoint(self, epoch, log_dir):
        
        ge_optimizer = self.ge_optimizer
        ga_optimizer = self.ga_optimizer
        de_optimizer = self.de_optimizer
        # Save model
        torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'ge_optimizer_state_dict': ge_optimizer.state_dict(),
                'ga_optimizer_state_dict': ga_optimizer.state_dict(),
                'de_optimizer_state_dict': de_optimizer.state_dict(),
                'loss': self.train_loss[-1],
                }, os.path.join(log_dir, f'{epoch+1}.pth.tar'))
        
    def _run_batch(self, source, target, train=True):
        if train:
            self.ge_optimizer.zero_grad()
            self.ga_optimizer.zero_grad()
            self.de_optimizer.zero_grad()
            
            # Forward prop
            gaze_pred, img_pred = self.model(source)
            
            for param in self.model.deconv.parameters():
                        param.requires_grad=False

            # loss calculation and backprop
            geloss = self.geloss_op(gaze_pred, img_pred, target, source)
            geloss.backward(retain_graph=True)


            for param in self.model.deconv.parameters():
                param.requires_grad=True


            for param in self.model.feature.parameters():
                param.requires_grad = False

            deloss = self.deloss_op(img_pred, source)
            deloss.backward()

            for param in self.model.feature.parameters():
                param.requires_grad=True

            self.ge_optimizer.step()
            self.ga_optimizer.step()
            self.de_optimizer.step()
            self.ge_scheduler.step(geloss)
            self.ga_scheduler.step(geloss)
            self.de_scheduler.step(deloss)

            return geloss, deloss
        
        else:
            self.model.eval()
            # Forward prop
            gaze_pred, img_pred = self.model(source)
            geloss = self.geloss_op(gaze_pred, img_pred, target, source)
            deloss = self.deloss_op(img_pred, source)

            self.model.train()
            return geloss, deloss


    def _run_epoch(self, epoch, pbar, writer):
        
        # Training epoch -----------------------------------------------------
        for items in tqdm.tqdm(self.train_loader):
            self.iter += 1
            source = items["image"]; targets = items["gaze_dir"]
            source = source.to(f"cuda:{self.gpu_id}")
            targets = targets.to(f"cuda:{self.gpu_id}")
            geloss, deloss = self._run_batch(source, targets)
            pbar.set_postfix({"geloss": f"{geloss:.2f}", "deloss": f"{deloss:.2f}"})
            
            # Tensorboard
            writer.add_scalar("Train GELoss", geloss.detach().item(), self.iter)
            writer.add_scalar("Train DELoss", deloss.detach().item(), self.iter)

        # Validation Set -----------------------------------------------------
        for items in self.eval_loader:
            source = items["image"]; targets = items["gaze_dir"]
            source = source.to(f"cuda:{self.gpu_id}")
            targets = targets.to(f"cuda:{self.gpu_id}")
            eval_geloss, eval_deloss = self._run_batch(source, targets, train=False)
            pbar.set_postfix({"eval_geloss": f"{eval_geloss:.2f}", "eval_deloss": f"{eval_deloss:.2f}"})

            # Tensorboard
            writer.add_scalar("Eval GELoss", eval_geloss.detach().item(), self.epoch)
            writer.add_scalar("Eval DELoss", eval_deloss.detach().item(), self.epoch)
        

    def train(self, total_epochs: int):
        writer = SummaryWriter(log_dir=log_dir)
        pbar = tqdm.trange(total_epochs, desc='Training')
        for epoch in pbar:
            self.epoch = epoch
            self._run_epoch(epoch, pbar=pbar, writer=writer)
            
            if epoch % self.save_every == 0:
                self._save_checkpoint(self.epoch, self.log_dir)

    def _test_train(self):
        return NotImplementedError("Need to be Implemented")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._save_checkpoint(self.epoch, self.log_dir)


### Helper Functions ---------------------------------------------------------------
def load_train_objs(data_path, lr, total_epochs, batch_size, save_every, pretrained):
    
    train_dataset = Gaze360(data_path, mode="train")
    eval_dataset = Gaze360(data_path, mode="eval")
    logger.debug("===> Model building <===")
    net = model.Model()
    net.train()
    net.cuda()

    if type(pretrained)!=type(None):
        if os.path.exists(pretrained): #TODO: Modify to get last epoch, optim, ...
            net.load_state_dict(torch.load(pretrained), strict=False)

    logger.debug("optimizer building")
    optimizers={}

    optimizers["ge_optimizer"] = optim.Adam(net.feature.parameters(),
             lr=lr, betas=(0.9,0.95))

    optimizers["ga_optimizer"] = optim.Adam(net.gazeEs.parameters(), 
             lr=lr, betas=(0.9,0.95))

    optimizers["de_optimizer"] = optim.Adam(net.deconv.parameters(), 
            lr=lr, betas=(0.9,0.95))
    
    return train_dataset, eval_dataset, net, optimizers

def load_losses(map, w1, w2):
    attentionmap = cv2.imread(map, 0)/255
    attentionmap = torch.from_numpy(attentionmap).type(torch.FloatTensor)
    geloss_op = model.Gelossop(attentionmap, w1, w2)
    deloss_op = model.Delossop()
    return {"geloss_op":geloss_op, "deloss_op":deloss_op}


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True
    )

## main function ---------------------------------------------------------------------------
def main(device, log_dir, config):
    dataset, eval_dataset, model, optimizers = load_train_objs(**config["train_params"])
    loss = load_losses(**config["loss_params"])
    
    train_params = config["train_params"]
    train_loader = prepare_dataloader(dataset, train_params["batch_size"])
    eval_loader =  prepare_dataloader(eval_dataset, train_params["batch_size"])
    
    with Trainer(model, train_loader, eval_loader, optimizers, loss, log_dir, device, train_params["save_every"]) as trainer:
        trainer.train(train_params["total_epochs"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train gaze estimation model-single GPU')
    parser.add_argument('--gpu_id', default=0, type=int, help='GPU ID to train the model on')
    parser.add_argument('--log_dir', default="logs/" ,type=str, help='Path to save the logs')
    parser.add_argument('--config', default="./configs/gaze360.yml", type=str, help='Config file path')
    args = parser.parse_args()
    
    with open(args.config) as f:
            configs = yaml.safe_load(f)

    # Log Dir
    log_dir = os.path.join(args.log_dir, os.path.basename(args.config).split('.')[0])
    log_dir += ' ' + strftime("%d_%m_%y_%H.%M.%S", gmtime()) + '_e:' + str(configs["train_params"]["total_epochs"]) + '_lr:' + str(configs["train_params"]["lr"])
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(args.config))):
        copy(args.config, log_dir)

    #device = 0  # shorthand for cuda:0
    main(args.gpu_id, log_dir, configs)