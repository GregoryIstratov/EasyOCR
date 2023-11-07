import torch.nn as nn
from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention
from recognition_cnn import *
from torchvision import models, transforms

from pathlib import Path


class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()

        print(f"Loading model... opts: {opt}")

        self.opt = opt
        self.stages = {'Trans': opt.Transformation, 'Feat': opt.FeatureExtraction,
                       'Seq': opt.SequenceModeling, 'Pred': opt.Prediction}

        """ Transformation """
        if opt.Transformation == 'TPS':
            print(f"Transformation TPS")
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW), I_channel_num=opt.input_channel)
        else:
            print('No Transformation module specified')

        """ FeatureExtraction """
        if opt.FeatureExtraction == 'VGG':
            print(f"Feature extraction VGG")
            self.FeatureExtraction = VGG_FeatureExtractor(opt.input_channel, opt.output_channel)

        elif opt.FeatureExtraction == 'RCNN':
            print(f"Feature extraction RCNN")
            self.FeatureExtraction = RCNN_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'ResNet':
            print(f"Feature extraction ResNet")
            self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = opt.output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        #LP RESNET
        model = resnet()

        #updated_params = torch.load("/home/ivan/projects/license/backup/lp_down_pad1_rec_e16.pt").state_dict()
        #updated_params = models.resnet50(pretrained=True).state_dict()
        model_path = Path(opt.saved_model)
        
        if model_path.suffix == ".pt":
            updated_params = torch.load(model_path).state_dict()
        else:
            updated_params = torch.load(model_path)
        new_params = model.state_dict()
        new_params.update(updated_params)
        model.load_state_dict(new_params, strict=False)

        self.resnet_features = model

        print("OUTPUT", self.FeatureExtraction_output)

        """ Sequence modeling"""
        if opt.SequenceModeling == 'BiLSTM':
            print(f"Sequence Modeling BiLSTM")
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
                BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
            self.SequenceModeling_output = opt.hidden_size
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if opt.Prediction == 'CTC':
            print(f"Prediction CTC")
            self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class)
        elif opt.Prediction == 'Attn':
            print(f"Feature extraction Attn")
            self.Prediction = Attention(self.SequenceModeling_output, opt.hidden_size, opt.num_class)
        else:
            raise Exception('Prediction is neither CTC or Attn')

    def forward(self, input, text, is_train=True):
        """ Transformation stage """
        #print("-1", input.size())

        if(False):
            if not self.stages['Trans'] == "None":
                input = self.Transformation(input)

        #print("0", input.size())

        """ Feature extraction stage """
        #visual_feature = self.FeatureExtraction(input)

        #print(input.size())

        #input = input.expand(input.size(0), 3, input.size(2), input.size(3))


        #print(input.size())

        visual_feature = self.resnet_features(input)

        #print("1",visual_feature.size())

        #visual_feature = nn.AdaptiveAvgPool2d((None, 1))()
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]

        #print("2",visual_feature.size())

        visual_feature = visual_feature.squeeze(3)

        #print("3", visual_feature.size())

        """ Sequence modeling stage """
        if(True):
            if self.stages['Seq'] == 'BiLSTM':
                contextual_feature = self.SequenceModeling(visual_feature)
            else:
                contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        #contextual_feature = visual_feature

        """ Prediction stage """
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.opt.batch_max_length)

        return prediction
