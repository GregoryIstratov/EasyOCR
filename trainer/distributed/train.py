import argparse
import os
import sys
import time
import torch.backends.cudnn as cudnn
import yaml

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "TextRecognitionDataGenerator/")
sys.path.insert(0, "TextRecognitionDataGenerator/server")

from dataset import GenDatasetLocal, GenDatasetRemote, RangedGenDataset, Dataloader, OCRDataset


import numpy as np

import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler

from torch.nn.parallel import DistributedDataParallel as DDP

from model import Model
from utils import AttrDict

from utils import CTCLabelConverter, AttnLabelConverter, Averager
from modules.prediction import Attention
from test import validation, validation2

from torch.distributed.elastic.multiprocessing.errors import record

import random
import logging
import logger as lg
from threading import Thread
from TextRecognitionDataGenerator.trdg_server import launch_trdg_server

def get_config(file_path):
    with open(file_path, 'r', encoding="utf8") as stream:
        opt = yaml.safe_load(stream)
    opt = AttrDict(opt)
    opt.character = opt.number + opt.symbol + opt.lang_char
    opt.tg_settings['symbols'] = opt.symbol
    opt.tg_settings['max_len'] = opt.batch_max_length
    opt.tg_settings['height'] = opt.imgH
    opt = AttrDict(dict(opt))
    os.makedirs(f'./saved_models/{opt.experiment_name}', exist_ok=True)
    return opt

# On Windows platform, the torch.distributed package only
# supports Gloo backend, FileStore and TcpStore.
# For FileStore, set init_method parameter in init_process_group
# to a local file. Example as follow:
# init_method="file:///f:/libtmp/some_file"
# dist.init_process_group(
#    "gloo",
#    rank=rank,
#    init_method=init_method,
#    world_size=world_size)
# For TcpStore, same way as on Linux.

def setup(world_rank, world_size, single:bool):
    cudnn.benchmark = True
    cudnn.deterministic = False
    
    # initialize the process group
    dist.init_process_group("nccl", init_method="env://", rank=world_rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
@record
def run(rank, world_rank, world_size, single:bool, opt):
    print(f"Running DDP on world_rank: {world_rank} local_rank {rank}.")
    
    def is_master():
        return world_rank == 0
    
    if single:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '12355'
        setup(rank, world_size, single)
    else:
        setup(world_rank, world_size, single)
    
    torch.cuda.set_device(rank)
    
    if is_master():
        logger: logging.Logger = lg.create_logger("stdout", lg.LogLevel.INFO, filename=f"./saved_models/{opt.experiment_name}/log_output.log")
        logger.info("Running in distributer parallel mode")
    
    if not opt.train_data:
        if opt.tg_distributed:
            gen_ds = GenDatasetRemote(opt)
        else:
            gen_ds = GenDatasetLocal(opt)
            
        train_dataset = Dataloader(opt, RangedGenDataset(gen_ds, opt.epoch_size // world_size), workers=opt.workers, prefetch_factor=opt.prefetch_train)
            
    else:
        train_dataset = Dataloader(opt, OCRDataset(root=opt.train_data, opt=opt), workers=opt.workers, prefetch_factor=opt.prefetch_train)
    
    if not opt.valid_data:
        valid_dataset = Dataloader(opt, RangedGenDataset(GenDatasetLocal(opt), opt.valid_items_count), workers=4, prefetch_factor=opt.prefetch_valid)
    else:            
        valid_dataset = Dataloader(opt, OCRDataset(root=opt.valid_data, opt=opt), workers=4, prefetch_factor=opt.prefetch_valid)
    
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)    

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)

    if opt.saved_model != '':
        if opt.new_prediction:
            if opt.Prediction == 'CTC':
                model.Prediction = nn.Linear(model.SequenceModeling_output, opt.num_class)
            elif opt.Prediction == 'Attn':
                model.Prediction = Attention(model.SequenceModeling_output, opt.hidden_size, opt.num_class)
            #model.Prediction = nn.Linear(model.SequenceModeling_output, len(pretrained_dict['module.Prediction.weight']))  
        
        model = model.to(rank) 
        
        if opt.new_prediction:
            if opt.Prediction == 'CTC':
                model.Prediction = nn.Linear(model.module.SequenceModeling_output, opt.num_class)
            elif opt.Prediction == 'Attn':
                model.Prediction = Attention(model.module.SequenceModeling_output, opt.hidden_size, opt.num_class)            
            #model.module.Prediction = nn.Linear(model.module.SequenceModeling_output, opt.num_class)  
            for name, param in model.module.Prediction.named_parameters():
                if 'bias' in name:
                    init.constant_(param, 0.0)
                elif 'weight' in name:
                    init.kaiming_normal_(param)
            model = model.to(rank) 
            
        # freeze some layers
        try:
            if opt.freeze_FeatureFxtraction:
                for param in model.module.FeatureExtraction.parameters():
                    param.requires_grad = False
            if opt.freeze_SequenceModeling:
                for param in model.module.SequenceModeling.parameters():
                    param.requires_grad = False
            if opt.freeze_Prediction:
                for param in model.module.Prediction.parameters():
                    param.requires_grad = False
        except:
            pass
    else:
        # weight initialization
        for name, param in model.named_parameters():
            if 'localization_fc2' in name:
                print(f'Skip {name} as it is already initialized')
                continue
            try:
                if 'bias' in name:
                    init.constant_(param, 0.0)
                elif 'weight' in name:
                    init.kaiming_normal_(param)
            except Exception as e:  # for batchnorm.
                if 'weight' in name:
                    param.data.fill_(1)
                continue
        model = model.to(rank)       
        
    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
        
    if is_master():
        logger.info(f'Trainable params num : {sum(params_num)}')

    if opt.saved_model != '':
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        pretrained_dict = torch.load(opt.saved_model, map_location=map_location)
    #model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # create model and move it to GPU with id rank
    
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            
    ddp_model = DDP(model, device_ids=[rank], output_device=rank)
          
    if is_master():
        logger.info(f'Loading pretrained model from {opt.saved_model}')
    
    if opt.saved_model != '':
        ddp_model.load_state_dict(pretrained_dict, strict=False)
            
    """ setup loss """
    if 'CTC' in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(rank)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(rank)  # ignore [GO] token = ignore index 0
    # loss averager
    loss_avg = Averager()
    
    # setup optimizer
    match opt.optim:
        case 'adamw':
            lr = 1e-3 * opt.lr
            optimizer = optim.AdamW(filtered_parameters, lr=lr)
        case 'adam':
            #optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
            lr = 1e-3 * opt.lr
            optimizer = optim.Adam(filtered_parameters, lr=lr)
        case 'adadelta':
            optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps, weight_decay=0.00001)
    
    if is_master():
        logger.info(f"Optimizer: \n{optimizer}")
        
    if opt.sched_enabled:
        if is_master():
            logger.info(f"Creating scheduler MulLR factor: {opt.sched_MulLR_factor}")
            
        lambda1 = lambda epoch: opt.sched_MulLR_factor
        scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda1)  
    
    ddp_model.train() 
    #print(f"[R{rank}] DDP Model: \n{ddp_model}")    
    
    """ start training """
    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    ep = 0
    scaler = GradScaler()
    total_trained: int = 0
    valid_before = opt.saved_model != ''
    
    for ep in range(opt.epochs):
        if opt.sched_enabled and ep != 0:
            scheduler.step()
            if is_master():
                logger.info(f"[{ep}/{opt.epochs}] Scheduler step, new lr: {optimizer.param_groups[0]['lr']:0.7f}")
        
        def train():
            amp = opt.amp
            t1=time.time()
            nonlocal loss_avg
            nonlocal total_trained
            tm_it_st = time.time()
            num_batches = len(train_dataset)
            if is_master():
                logger.info(f"Staring new epoch [{ep+1:02d}/{opt.epochs}], batches={num_batches}")
            items_trained = 0
            for i, (image_tensors, labels) in enumerate(train_dataset):
                total_trained = total_trained + image_tensors.size(0)                
                items_trained = items_trained + image_tensors.size(0)                
                
                # train part
                optimizer.zero_grad(set_to_none=True)
                
                text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
                
                if amp:
                    with autocast():
                        image = image_tensors.to(rank)
                        batch_size = image.size(0)

                        if 'CTC' in opt.Prediction:
                            preds = ddp_model(image, text).log_softmax(2)
                            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                            preds = preds.permute(1, 0, 2)
                            torch.backends.cudnn.enabled = False
                            cost = criterion(preds, text.to(rank), preds_size.to(rank), length.to(rank))
                            torch.backends.cudnn.enabled = True
                        else:
                            preds = ddp_model(image, text[:, :-1])  # align with Attention.forward
                            target = text[:, 1:]  # without [GO] Symbol
                            cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
                    scaler.scale(cost).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), opt.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    image = image_tensors.to(rank)
                    batch_size = image.size(0)
                    if 'CTC' in opt.Prediction:
                        preds = ddp_model(image, text).log_softmax(2)
                        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                        preds = preds.permute(1, 0, 2)
                        torch.backends.cudnn.enabled = False
                        cost = criterion(preds, text.to(rank), preds_size.to(rank), length.to(rank))
                        torch.backends.cudnn.enabled = True
                    else:
                        preds = ddp_model(image, text[:, :-1])  # align with Attention.forward
                        target = text[:, 1:]  # without [GO] Symbol
                        cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
                    cost.backward()
                    torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), opt.grad_clip) 
                    optimizer.step()
                loss_avg.add(cost)
    
                if i % opt.status_every_batch == 0 or i == (num_batches - 1):
                    tm_now = time.time()
                    elapsed = tm_now - tm_it_st
                    elapsed_ep_st = tm_now - t1 
                    tm_it_st = tm_now
                    items_sec = items_trained / elapsed_ep_st * world_size
                    if is_master():
                        logger.info(f"[ep {ep+1:03d}/{opt.epochs:03d}][{i+1:03d}/{num_batches:03d}] items trained (total: {total_trained * world_size:06d} ep: {items_trained * world_size:06d}) loss: {loss_avg.val():0.4f} lr: {optimizer.param_groups[0]['lr']:0.7f} time: {elapsed:0.5f} sec items/sec: {items_sec:0.1f}")
            
            if is_master():        
                epoch_elapsed = time.time() - t1
                items_sec = items_trained / epoch_elapsed * world_size
                logger.info(f'Epoch {ep+1}/{opt.epochs} end, training time: {epoch_elapsed:0.5f} items itrained: {items_trained * world_size:06d} items/sec: {items_sec:0.1f}')                 


        def validate():
            nonlocal best_accuracy
            nonlocal best_norm_ED
            nonlocal loss_avg
            # validation part
            logger.info("Begin validation...")
            elapsed_time = time.time() - start_time
            t1=time.time()
            # for log
            with open(f'./saved_models/{opt.experiment_name}/log_train.txt', 'a', encoding="utf8") as log:
                ddp_model.eval()
                with torch.no_grad():
                    # valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels,\
                    # infer_time, length_of_data = validation(ddp_model, criterion, valid_loader, converter, opt, rank)
                    valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels,\
                    infer_time, length_of_data = validation2(ddp_model, criterion, valid_dataset, converter, opt, rank)
                ddp_model.train()

                # training loss and validation loss
                loss_log = f'[ep {ep+1}/{opt.epochs}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
                loss_avg.reset()

                current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_norm_ED":17s}: {current_norm_ED:0.4f}'
                
                torch.save(ddp_model.state_dict(), f'./saved_models/{opt.experiment_name}/current.pth')

                # keep best accuracy model (on valid dataset)
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    torch.save(ddp_model.state_dict(), f'./saved_models/{opt.experiment_name}/best_accuracy.pth')
                if current_norm_ED > best_norm_ED:
                    best_norm_ED = current_norm_ED
                    torch.save(ddp_model.state_dict(), f'./saved_models/{opt.experiment_name}/best_norm_ED.pth')
                best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_norm_ED":17s}: {best_norm_ED:0.4f}'

                loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
                #logger.info(loss_model_log)
                log.write(loss_model_log + '\n')

                # show some predicted results
                dashed_line = '-' * 80
                head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
                
                show_number = min(opt.val_show_number, len(labels))
                
                start = random.randint(0,len(labels) - show_number )    
                for gt, pred, confidence in zip(labels[start:start+show_number], preds[start:start+show_number], confidence_score[start:start+show_number]):
                    if 'Attn' in opt.Prediction:
                        gt = gt[:gt.find('[s]')]
                        pred = pred[:pred.find('[s]')]

                    predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
                predicted_result_log += f'{dashed_line}'
                logger.info(f"\n{loss_model_log}\n{predicted_result_log}")
                log.write(predicted_result_log + '\n')
                logger.info(f'validation time: {time.time()-t1}')
                
        if is_master() and valid_before:
            valid_before = False
            validate()
            
        train()
        
        if is_master():
            validate()
            save_path = f'./saved_models/{opt.experiment_name}/ep_{ep+1}.pth'
            logger.info(f"Saving to {save_path}")
            torch.save(ddp_model.state_dict(), save_path)

    cleanup()
    
    
def run_trdg_dbg_server(tg_settings: dict):
    print("Creating trdg dbg server...")
    launch_trdg_server(tg_settings)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()    
    
    opt = get_config(args.config)
    
    LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    WORLD_RANK = int(os.environ['RANK'])
    
    if WORLD_RANK == 0 and opt.tg_dbg_server:
        tg_set = dict(opt.tg_settings)
        t = Thread(target=run_trdg_dbg_server, args=(tg_set,))
        t.start()
    
    run(rank=LOCAL_RANK, world_rank=WORLD_RANK, world_size=WORLD_SIZE, single=False, opt=opt)