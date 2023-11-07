import os
import sys
import re
import six
import math
import torch
import pandas  as pd
import random as rnd
from typing import Generic, TypeVar

from natsort import natsorted
from PIL import Image, ImageFilter
import numpy as np
from torch.utils.data import Dataset, IterableDataset, ConcatDataset, Subset
from torch._utils import _accumulate
import torchvision.transforms as transforms
from TextRecognitionDataGenerator.generate import Generator as TextGen
from TextRecognitionDataGenerator.server.client import Client as TextGenDistClient
from TextRecognitionDataGenerator.trdg.aug import build_augmentation_pipeline
import multiprocessing as mp

from PIL import Image

def contrast_grey(img):
    high = np.percentile(img, 90)
    low  = np.percentile(img, 10)
    return (high-low)/(high+low), high, low

def adjust_contrast_grey(img, target = 0.4):
    contrast, high, low = contrast_grey(img)
    if contrast < target:
        img = img.astype(int)
        ratio = 200./(high-low)
        img = (img - low + 25)*ratio
        img = np.maximum(np.full(img.shape, 0) ,np.minimum(np.full(img.shape, 255), img)).astype(np.uint8)
    return img


class Batch_Balanced_Dataset(object):

    def __init__(self, opt):
        """
        Modulate the data ratio in the batch.
        For example, when select_data is "MJ-ST" and batch_ratio is "0.5-0.5",
        the 50% of the batch is filled with MJ and the other 50% of the batch is filled with ST.
        """
        log = open(f'./saved_models/{opt.experiment_name}/log_dataset.txt', 'a')
        dashed_line = '-' * 80
        print(dashed_line)
        log.write(dashed_line + '\n')
        print(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}')
        log.write(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}\n')
        assert len(opt.select_data) == len(opt.batch_ratio)

        _AlignCollate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, contrast_adjust = opt.contrast_adjust)
        self.data_loader_list = []
        self.dataloader_iter_list = []
        batch_size_list = []
        Total_batch_size = 0
        for selected_d, batch_ratio_d in zip(opt.select_data, opt.batch_ratio):
            _batch_size = max(round(opt.batch_size * float(batch_ratio_d)), 1)
            print(dashed_line)
            log.write(dashed_line + '\n')
            _dataset, _dataset_log = hierarchical_dataset(root=opt.train_data, opt=opt, select_data=[selected_d])
            total_number_dataset = len(_dataset)
            log.write(_dataset_log)

            """
            The total number of data can be modified with opt.total_data_usage_ratio.
            ex) opt.total_data_usage_ratio = 1 indicates 100% usage, and 0.2 indicates 20% usage.
            See 4.2 section in our paper.
            """
            number_dataset = int(total_number_dataset * float(opt.total_data_usage_ratio))
            dataset_split = [number_dataset, total_number_dataset - number_dataset]
            indices = range(total_number_dataset)
            _dataset, _ = [Subset(_dataset, indices[offset - length:offset])
                           for offset, length in zip(_accumulate(dataset_split), dataset_split)]
            selected_d_log = f'num total samples of {selected_d}: {total_number_dataset} x {opt.total_data_usage_ratio} (total_data_usage_ratio) = {len(_dataset)}\n'
            selected_d_log += f'num samples of {selected_d} per batch: {opt.batch_size} x {float(batch_ratio_d)} (batch_ratio) = {_batch_size}'
            print(selected_d_log)
            log.write(selected_d_log + '\n')
            batch_size_list.append(str(_batch_size))
            Total_batch_size += _batch_size

            _data_loader = torch.utils.data.DataLoader(
                _dataset, batch_size=_batch_size,
                shuffle=True,
                num_workers=int(opt.workers), #prefetch_factor=2,persistent_workers=True,
                collate_fn=_AlignCollate, pin_memory=True)
            self.data_loader_list.append(_data_loader)
            self.dataloader_iter_list.append(iter(_data_loader))

        Total_batch_size_log = f'{dashed_line}\n'
        batch_size_sum = '+'.join(batch_size_list)
        Total_batch_size_log += f'Total_batch_size: {batch_size_sum} = {Total_batch_size}\n'
        Total_batch_size_log += f'{dashed_line}'
        opt.batch_size = Total_batch_size

        print(Total_batch_size_log)
        log.write(Total_batch_size_log + '\n')
        log.close()

    def get_batch(self):
        balanced_batch_images = []
        balanced_batch_texts = []

        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            try:
                image, text = next(data_loader_iter)
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except StopIteration:
                self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                image, text = next(self.dataloader_iter_list[i])
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except ValueError:
                pass

        balanced_batch_images = torch.cat(balanced_batch_images, 0)

        return balanced_batch_images, balanced_batch_texts
    
class GenDatasetBase(IterableDataset):
    def __init__(self) -> None:
        super().__init__()
        
    def __iter__(self):
        raise NotImplementedError        
    
class GenDatasetLocal(GenDatasetBase):

    def __init__(self, opt):
        super().__init__()
        self.tg_settings = opt.tg_settings

    def __iter__(self):
        gen = TextGen(opt=dict(self.tg_settings))
        
        return gen
    
class GenDatasetRemote(GenDatasetBase):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    def __iter__(self):
        gen = TextGenDistClient(self.opt.tg_distributed)
        gen.connect()
        
        return gen    
    
    
class RangedGenDataset(Dataset):
    def __init__(self, gen_dataset: GenDatasetBase, n_samples: int) -> None:
        super().__init__()
        self.ds = gen_dataset
        self.iter = iter(self.ds)
        self.n_samples = n_samples
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        return next(self.iter)
        
DatasetT = GenDatasetLocal | GenDatasetRemote | Dataset

class IterDataloader(object):

    def __init__(self, opt, dataset:DatasetT, workers:int = 4, prefetch_factor: int = 2, shuffle: bool = False):

        _AlignCollate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, contrast_adjust = opt.contrast_adjust)
        self.data_loader_list = []
        self.dataloader_iter_list = []

        self._data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=opt.batch_size,
                shuffle=shuffle,
                num_workers=workers, prefetch_factor=prefetch_factor, persistent_workers=True,
                collate_fn=_AlignCollate, pin_memory=True)

        self.data_loader_iter = iter(self._data_loader)
        
    def __iter__(self):
        return self.data_loader_iter
    
    def __next__(self):
        return self.get_batch()

    def get_batch(self):       
        try:
            image, text = next(self.data_loader_iter)
        except StopIteration:
            self.data_loader_iter = iter(self._data_loader)
            return self.get_batch()

        return image, text
    
class Dataloader(object):

    def __init__(self, opt, dataset:Dataset, workers:int = 4, prefetch_factor: int = 2, shuffle: bool = False):

        _AlignCollate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, contrast_adjust = opt.contrast_adjust)
        self.dataset = dataset

        self._data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=opt.batch_size,
                shuffle=shuffle,
                num_workers=workers, prefetch_factor=prefetch_factor, persistent_workers=True,
                collate_fn=_AlignCollate, pin_memory=True)

        self.data_loader_iter = iter(self._data_loader)
        
    def __len__(self):
        return len(self._data_loader)
    
    def __iter__(self):
        return iter(self._data_loader)

    # def get_batch(self):       
    #     try:
    #         image, text = next(self.data_loader_iter)
    #     except StopIteration:
    #         self.data_loader_iter = iter(self._data_loader)
    #         return self.get_batch()

    #     return image, text
        

def hierarchical_dataset(root, opt, select_data='/'):
    """ select_data='/' contains all sub-directory of root directory """
    dataset_list = []
    dataset_log = f'dataset_root:    {root}\t dataset: {select_data[0]}'
    print(dataset_log)
    dataset_log += '\n'
    for dirpath, dirnames, filenames in os.walk(root+'/'):
        if not dirnames:
            select_flag = False
            for selected_d in select_data:
                if selected_d in dirpath:
                    select_flag = True
                    break

            if select_flag:
                dataset = OCRDataset(dirpath, opt)
                sub_dataset_log = f'sub-directory:\t/{os.path.relpath(dirpath, root)}\t num samples: {len(dataset)}'
                print(sub_dataset_log)
                dataset_log += f'{sub_dataset_log}\n'
                dataset_list.append(dataset)

    concatenated_dataset = ConcatDataset(dataset_list)

    return concatenated_dataset, dataset_log




class GenDataset(IterableDataset):

    def __init__(self, opt):
        super(GenDataset).__init__()
        self.opt = opt
        self.counter = mp.Value('l', 0)
        self.id = rnd.randint(0, 256)

    def __iter__(self):
        gen = None
        i = None
        with self.counter.get_lock():
            i = int(self.counter.value)
            self.counter.value += 1
            
                
        if self.opt.tg_distributed and i % 2 == 0:
            print(f"[id={self.id} counter={i}] Creating new TextGenDistClient iter")
            gen = TextGenDistClient(self.opt.tg_distributed)
            gen.connect()
        else:
            print(f"[id={self.id} counter={i}] Creating new TextGen iter")
            gen = TextGen(max_len=self.opt.batch_max_length, blur=self.opt.tg_blur, random_blur=self.opt.tg_random_blur, 
                          length=self.opt.tg_words_len, 
                          height=self.opt.imgH, rgb=self.opt.rgb, sensitive=self.opt.sensitive,
                          aug_opts=dict(self.opt.tg_augs))
        
        return gen
    
class GenDatasetRanged(Dataset):

    def __init__(self, opt, count):
        super(GenDatasetRanged).__init__()
        self.ds = GenDataset(opt)
        self.count = count
        self.gen = iter(self.ds)

    def __len__(self):
        return self.count
    
    def __getitem__(self, index):
        return next(self.gen)

class OCRDataset(Dataset):

    def __init__(self, root, opt, enable_augs=False):

        self.root = root
        self.opt = opt
        self.enable_augs = enable_augs
        
        if self.enable_augs:
            self.aug_pipelines = build_augmentation_pipeline(opt.tg_settings['augs'])
                
        print(root)
        self.df = pd.read_csv(os.path.join(root,'labels.csv'), sep='^([^,]+),', engine='python', usecols=['filename', 'words'], keep_default_na=False)
        self.nSamples = len(self.df)     

        if self.opt.data_filtering_off:
            self.filtered_index_list = [index for index in range(self.nSamples)]
        else:
            self.filtered_index_list = []
            for index in range(self.nSamples):
                label = self.df.at[index,'words']
                try:
                    if len(label) > self.opt.batch_max_length:
                        continue
                except:
                    print(label)
                out_of_char = f'[^{self.opt.character}]'
                if re.search(out_of_char, label.lower()):
                    continue
                self.filtered_index_list.append(index)
            self.nSamples = len(self.filtered_index_list)
            
        # self.filtered_index_list = []
        # for index in range(self.nSamples):
        #     label = self.df.at[index,'words']

        #     if re.search('<+', label.lower()):
        #         continue
        #     self.filtered_index_list.append(index)
        # self.nSamples = len(self.filtered_index_list)               

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        index = self.filtered_index_list[index]
        img_fname = self.df.at[index,'filename']
        img_fpath = os.path.join(self.root, img_fname)
        label = self.df.at[index,'words']

        if self.opt.rgb:
            img = Image.open(img_fpath).convert('RGB')  # for color image
        else:
            img = Image.open(img_fpath).convert('L')

        if not self.opt.sensitive:
            label = label.upper()

        # We only train and evaluate on alphanumerics (or pre-defined character set in train.py)
        out_of_char = f'[^{self.opt.character}]'
        label = re.sub(out_of_char, '', label)
        
        #label = re.sub('\.', ' ', label)
        label = re.sub('\s+', ' ', label)
        #label = re.sub('["\',]+', '', label)
        label = label.strip()
        
        w, h = img.size
        ratio = w / float(h)
        if math.ceil(self.opt.imgH * ratio) > self.opt.imgW:
            resized_w = self.opt.imgW
        else:
            resized_w = math.ceil(self.opt.imgH * ratio)

        img = img.resize((resized_w, self.opt.imgH), Image.LINEAR)        
        
        if self.enable_augs:
            #img.save("test_orig.jpg", "JPEG")
            img = self.aug_pipelines["doc"].apply(img)
            #img.save("test.jpg", "JPEG")
            
        
        return (img, label)
    
class DocGenDataset(Dataset):
    doc: Dataset
    gen: Dataset
    doc_ratio: float
    
    def __init__(self, doc: Dataset, gen: Dataset, doc_ratio: float):
        self.doc = doc
        self.gen = gen
        self.doc_ratio = doc_ratio
        
    def __len__(self):
        return len(self.doc)
    
    def __getitem__(self, index):
        if rnd.random() < self.doc_ratio:
            return self.doc.__getitem__(index)
        else:
            return self.gen.__getitem__(index)

class ResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class NormalizePAD(object):

    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img


class AlignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False, contrast_adjust = 0.):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad
        self.contrast_adjust = contrast_adjust

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)

        if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
            resized_max_w = self.imgW
            input_channel = 3 if images[0].mode == 'RGB' else 1
            transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

            resized_images = []
            for image in images:
                w, h = image.size
                
                # if True:
                #     image = add_image_noise(image)
                #     # blur_fact = 2
                #     # gaussian_filter = ImageFilter.GaussianBlur(radius=blur_fact)
                #     # image = image.filter(gaussian_filter)         

                #### augmentation here - change contrast
                if self.contrast_adjust > 0:
                    image = np.array(image.convert("L"))
                    image = adjust_contrast_grey(image, target = self.contrast_adjust)
                    image = Image.fromarray(image, 'L')

                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)

                resized_image = image.resize((resized_w, self.imgH), Image.LINEAR)
                resized_images.append(transform(resized_image))
                # resized_image.save('./image_test/%d_test.jpg' % w)

            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

        else:
            transform = ResizeNormalize((self.imgW, self.imgH))
            image_tensors = [transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors, labels


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)
