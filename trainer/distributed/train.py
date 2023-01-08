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

from dataset import AlignCollate, Batch_Balanced_Dataset, Generated_Dataset, hierarchical_dataset, GenericDataset, GenDatasetLocal, GenDatasetRemote


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

cudnn.benchmark = True
cudnn.deterministic = False

def get_config(file_path):
    with open(file_path, 'r', encoding="utf8") as stream:
        opt = yaml.safe_load(stream)
    opt = AttrDict(opt)
    opt.character = opt.number + opt.symbol + opt.lang_char
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

    # initialize the process group
    dist.init_process_group("nccl", init_method="env://", rank=world_rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
@record
def run(rank, world_rank, world_size, single:bool, opt):
    print(f"Running DDP on world_rank: {world_rank} local_rank {rank}.")
    
    if single:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '12355'
        setup(rank, world_size, single)
    else:
        setup(world_rank, world_size, single)
    
    torch.cuda.set_device(rank)
    
    if opt.select_data != False:
        opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    
    if not opt.select_data:
        if opt.tg_distributed and rank % 2 == 1:
            train_dataset = GenericDataset(opt, GenDatasetRemote(opt), workers=opt.workers)
        else:
            train_dataset = GenericDataset(opt, GenDatasetLocal(opt), workers=opt.workers)            
            
    else:
        train_dataset = Batch_Balanced_Dataset(opt)    
    
    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, contrast_adjust=opt.contrast_adjust)
    
    if rank == 0:
        if not opt.valid_data:
            valid_dataset = GenericDataset(opt, GenDatasetLocal(opt), workers=4)
        else:            
            valid_dataset, valid_dataset_log = hierarchical_dataset(root=opt.valid_data, opt=opt)
            valid_loader = torch.utils.data.DataLoader(
                valid_dataset, batch_size=min(32, opt.batch_size),
                shuffle=True,  # 'True' to check training progress with validation function.
                num_workers=4, prefetch_factor=512,
                collate_fn=AlignCollate_valid, pin_memory=True)     
    
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)    

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print(f'[R{rank}] model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)

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
    print(f'[R{rank}] Trainable params num : ', sum(params_num))             

    if opt.saved_model != '':
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        pretrained_dict = torch.load(opt.saved_model, map_location=map_location)
    #model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # create model and move it to GPU with id rank
    ddp_model = DDP(model, device_ids=[rank], output_device=rank)        
    print(f'[R{rank}] loading pretrained model from {opt.saved_model}')
    
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
    print(f"[R{rank}] Optimizer: \n{optimizer}")
    
    ddp_model.train() 
    #print(f"[R{rank}] DDP Model: \n{ddp_model}")    
    
    """ start training """
    start_iter = 0

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    i = start_iter

    scaler = GradScaler()
    t1= time.time()
        
    valid_enabled = opt.saved_model != ''
    tm_it_st = time.time()
    
    show_number = 15
    
    while(True):
        # print(f"[R{rank}] Train loop begin sync")
        # dist.barrier()
        # print(f"[R{rank}] Train loop syncronized")
        
        # train part
        optimizer.zero_grad(set_to_none=True)
        
        if i > 0 or not valid_enabled:
            with autocast():
                image_tensors, labels = train_dataset.get_batch()
                image = image_tensors.to(rank)
                text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
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
            loss_avg.add(cost)
        
        if i % 100 == 0:
            tm_end = time.time()
            elapsed = tm_end - tm_it_st
            tm_it_st = time.time()
            print(f"[W{world_rank}R{rank}][{i}/{opt.num_iter}] Loss: {loss_avg.val()} time: {elapsed:0.5f} sec")

        # validation part
        if (rank == 0 and world_rank == 0 and i % opt.valInterval == 0 and valid_enabled):
            print('training time: ', time.time()-t1)
            t1=time.time()
            elapsed_time = time.time() - start_time
            # for log
            with open(f'./saved_models/{opt.experiment_name}/log_train.txt', 'a', encoding="utf8") as log:
                ddp_model.eval()
                with torch.no_grad():
                    valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels,\
                    infer_time, length_of_data = validation2(ddp_model, criterion, valid_dataset, converter, opt, rank, 25)
                ddp_model.train()

                # training loss and validation loss
                loss_log = f'[{i}/{opt.num_iter}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
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
                print(loss_model_log)
                log.write(loss_model_log + '\n')

                # show some predicted results
                dashed_line = '-' * 80
                head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
                
                show_number = min(show_number, len(labels))
                
                start = random.randint(0,len(labels) - show_number )    
                for gt, pred, confidence in zip(labels[start:start+show_number], preds[start:start+show_number], confidence_score[start:start+show_number]):
                    if 'Attn' in opt.Prediction:
                        gt = gt[:gt.find('[s]')]
                        pred = pred[:pred.find('[s]')]

                    predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
                predicted_result_log += f'{dashed_line}'
                print(predicted_result_log)
                log.write(predicted_result_log + '\n')
                print('validation time: ', time.time()-t1)
                t1=time.time()
                tm_it_st = time.time()
                
        valid_enabled = True
        # save model per 1e+4 iter.
        if (i + 1) % 5000 == 0:            
            CHECKPOINT_PATH = f'./saved_models/{opt.experiment_name}/iter_{i+1}.pth'
            if rank == 0:
                # All processes should see same parameters as they all start from same
                # random parameters and gradients are synchronized in backward passes.
                # Therefore, saving it in one process is sufficient.
                torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)
            
        if i == opt.num_iter:
            print('end the training')
            break
        i += 1    
        
        # print(f"[R{rank}] Syncronizing at loop end")
        # dist.barrier()

    cleanup()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--single", type=bool, default=False)
    args = parser.parse_args()    
    #opt = get_config("config_files/en_filtered_config.yaml")
    #opt = get_config("config_files/en_filtered_config_ft.yaml")
    #opt = get_config("config_files/orig_config_ft.yaml")
    #opt = get_config("config_files/orig_config_ft2.yaml")
    #opt = get_config("config_files/vgg_lstm_ctc.yaml")
    #opt = get_config("config_files/resnet_none_attn.yaml")
    #opt = get_config("config_files/resnet_lstm_attn.yaml")
    
    opt = get_config(args.config)
    
    if args.single:
        n_gpus = torch.cuda.device_count()
        assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
        world_size = n_gpus
        mp.spawn(run,
                args=(0, world_size, args.single, opt),
                nprocs=world_size,
                join=True)
    else:
        LOCAL_RANK = int(os.environ['LOCAL_RANK'])
        WORLD_SIZE = int(os.environ['WORLD_SIZE'])
        WORLD_RANK = int(os.environ['RANK'])
        run(rank=LOCAL_RANK, world_rank=WORLD_RANK, world_size=WORLD_SIZE, single=False, opt=opt)