import os
import sys
import time
import random
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
from torch.cuda.amp import autocast, GradScaler
import numpy as np

from utils import CTCLabelConverter, AttnLabelConverter, Averager
from dataset import GenDatasetLocal, GenDatasetRemote, RangedGenDataset, Dataloader, OCRDataset, hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from model import Model
from test import validation, validation2
from modules.prediction import Attention
import logging
import logger as lg

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def count_parameters(model, logger: logging.Logger):
    logger.info("Modules, Parameters")
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        #table.add_row([name, param])
        total_params+=param
        logger.info(f"{name}: {param}")
    logger.info(f"Total Trainable Params: {total_params}")
    return total_params

def train(opt):
    logger: logging.Logger = lg.create_logger("stdout", lg.LogLevel.INFO, filename=f"./saved_models/{opt.experiment_name}/log_output.log")
    
    """ dataset preparation """
    if not opt.data_filtering_off:
        logger.info('Filtering the images containing characters which are not in opt.character')
        logger.info('Filtering the images whose label is longer than opt.batch_max_length')
    
    # if not opt.select_data:
    #     train_dataset = Generated_Dataset(opt)
    # else:
    #     train_dataset = Batch_Balanced_Dataset(opt)
    
    if not opt.train_data:
        if opt.tg_distributed:
            gen_ds = GenDatasetRemote(opt)
        else:
            gen_ds = GenDatasetLocal(opt)
            
        train_dataset = Dataloader(opt, RangedGenDataset(gen_ds, opt.epoch_size), workers=opt.workers)
            
    else:
        train_dataset = Dataloader(opt, OCRDataset(root=opt.train_data, opt=opt), workers=opt.workers, prefetch_factor=64)
    
    if not opt.valid_data:
        valid_dataset = Dataloader(opt, RangedGenDataset(GenDatasetLocal(opt), opt.valid_items_count), workers=4)
    else:            
        valid_dataset = Dataloader(opt, OCRDataset(root=opt.valid_data, opt=opt), workers=4, prefetch_factor=64)
    
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
        pretrained_dict = torch.load(opt.saved_model)
        if opt.new_prediction:
            if opt.Prediction == 'CTC':
                model.Prediction = nn.Linear(model.SequenceModeling_output, opt.num_class)
            elif opt.Prediction == 'Attn':
                model.Prediction = Attention(model.SequenceModeling_output, opt.hidden_size, opt.num_class)
            #model.Prediction = nn.Linear(model.SequenceModeling_output, len(pretrained_dict['module.Prediction.weight']))  
        
        model = torch.nn.DataParallel(model).to(device) 
        logger.info(f'loading pretrained model from {opt.saved_model}')
        if opt.FT:
            model.load_state_dict(pretrained_dict, strict=False)
        else:
            model.load_state_dict(pretrained_dict)
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
            model = model.to(device) 
    else:
        # weight initialization
        for name, param in model.named_parameters():
            if 'localization_fc2' in name:
                logger.info(f'Skip {name} as it is already initialized')
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
        model = torch.nn.DataParallel(model).to(device)
    
    model.train() 
    logger.info("Model:")
    logger.info(model)
    count_parameters(model, logger)
    
    """ setup loss """
    if 'CTC' in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
    # loss averager
    loss_avg = Averager()

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
    
    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    logger.info(f'Trainable params num :  {sum(params_num)}')
    # [logger.info(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

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
    logger.info("Optimizer:")
    logger.info(optimizer)
    
    if opt.sched_enabled:
        logger.info(f"Creating scheduler MulLR factor: {opt.sched_MulLR_factor}")
        lambda1 = lambda epoch: opt.sched_MulLR_factor
        scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda1)     

    """ final options """
    # logger.info(opt)
    with open(f'./saved_models/{opt.experiment_name}/opt.txt', 'a', encoding="utf8") as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        logger.info(opt_log)
        opt_file.write(opt_log)

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
            logger.info(f"[{ep}/{opt.epochs}] Scheduler step, new lr: {optimizer.param_groups[0]['lr']:0.7f}")    
                
        def train():
            amp = opt.amp
            t1=time.time()
            nonlocal loss_avg
            nonlocal total_trained
            tm_it_st = time.time()
            num_batches = len(train_dataset)
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
                        image = image_tensors.to(device)
                        batch_size = image.size(0)

                        if 'CTC' in opt.Prediction:
                            preds = model(image, text).log_softmax(2)
                            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                            preds = preds.permute(1, 0, 2)
                            torch.backends.cudnn.enabled = False
                            cost = criterion(preds, text.to(device), preds_size.to(device), length.to(device))
                            torch.backends.cudnn.enabled = True
                        else:
                            preds = model(image, text[:, :-1])  # align with Attention.forward
                            target = text[:, 1:]  # without [GO] Symbol
                            cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
                    scaler.scale(cost).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    image = image_tensors.to(device)
                    batch_size = image.size(0)
                    if 'CTC' in opt.Prediction:
                        preds = model(image, text).log_softmax(2)
                        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                        preds = preds.permute(1, 0, 2)
                        torch.backends.cudnn.enabled = False
                        cost = criterion(preds, text.to(device), preds_size.to(device), length.to(device))
                        torch.backends.cudnn.enabled = True
                    else:
                        preds = model(image, text[:, :-1])  # align with Attention.forward
                        target = text[:, 1:]  # without [GO] Symbol
                        cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
                    cost.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip) 
                    optimizer.step()
                loss_avg.add(cost)
    
                if i % 4 == 0 or i == (num_batches - 1):
                    tm_end = time.time()
                    elapsed = tm_end - tm_it_st
                    tm_it_st = time.time()
                    logger.info(f"[ep {ep+1:03d}/{opt.epochs:03d}][{i+1:03d}/{num_batches:03d}] items trained (total: {total_trained:06d} ep: {items_trained:06d}) loss: {loss_avg.val():0.4f} lr: {optimizer.param_groups[0]['lr']:0.7f} time: {elapsed:0.5f} sec")
                    
            logger.info(f'training time: {time.time()-t1}')                 

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
                model.eval()
                with torch.no_grad():
                    # valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels,\
                    # infer_time, length_of_data = validation(model, criterion, valid_loader, converter, opt, device)
                    valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels,\
                    infer_time, length_of_data = validation2(model, criterion, valid_dataset, converter, opt, device)
                model.train()

                # training loss and validation loss
                loss_log = f'[ep {ep+1}/{opt.epochs}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
                loss_avg.reset()

                current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_norm_ED":17s}: {current_norm_ED:0.4f}'
                
                torch.save(model.state_dict(), f'./saved_models/{opt.experiment_name}/current.pth')

                # keep best accuracy model (on valid dataset)
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    torch.save(model.state_dict(), f'./saved_models/{opt.experiment_name}/best_accuracy.pth')
                if current_norm_ED > best_norm_ED:
                    best_norm_ED = current_norm_ED
                    torch.save(model.state_dict(), f'./saved_models/{opt.experiment_name}/best_norm_ED.pth')
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
                
        if valid_before:
            valid_before = False
            validate()
            
        train()
        validate()
        # save model per 1e+4 iter.
        torch.save(model.state_dict(), f'./saved_models/{opt.experiment_name}/ep_{ep+1}.pth')

