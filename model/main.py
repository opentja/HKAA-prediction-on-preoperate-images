import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from trainer import Trainer
import argparse
import torch
from model import Network
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import numpy as np
import os
import torch.nn as nn
# from pytorch_model_summary import summary
from torchsummary import summary

def select_device(device='', apex=True, batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = 'Using CUDA ' + ('Apex ' if apex else '')  # apex for mixed precision https://github.com/NVIDIA/apex
        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)
            print("%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                  (s, i, x[i].name, x[i].total_memory / c))
    else:
        print('Using CPU')

    print('')  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu')

def main():
    parser = argparse.ArgumentParser(description='CNN Example')

    parser.add_argument('--foldPath', default='/infodev1/phi-data/shi/kneeX-ray/data/crossValidation/')
    parser.add_argument('--stageFoldPath',
                        default='/infodev1/phi-data/shi/kneeX-ray/data/crossValidation/')
    parser.add_argument('--flag', default='hip', help='indicate which data you want to train, hip, knee or ankle.')
    # parser.add_argument('--foldPath', default='/Users/shiyan/Documents/Mayo/landmarkDetection/data/crossValidation/')

    parser.add_argument('--batchSize', type=int, default=16, metavar='16',
                        help='input batch size for training (default: 16)')

    parser.add_argument('--epochs', type=int, default=48, metavar='48',
                        help='number of epochs to train (default:48)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='0.001',
                        help='learning rate.')
    parser.add_argument('--nin', type=int, default=1, metavar='1',
                        help='number of input channel')
    parser.add_argument('--fold', type=int, default=5)

    parser.add_argument('--fcNodes', type=int, default=1024)
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--bestModelFold', default=5, type=int)
    parser.add_argument('--dataSource', type=str, default='oai')
    parser.add_argument('--device', default='0')
    parser.add_argument('--predictImagePath', default='/infodev1/phi-data/shi/kneeX-ray/preoperate/')
    parser.add_argument('--predictionResultPath', default='/infodev1/phi-data/shi/kneeX-ray/preoperatePrediction/')
    args = parser.parse_args()
    print(args)
    # Device configuration
    device = select_device(args.device,batch_size=args.batchSize)

    # device = torch.device('cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu')
    #model = Network(args)
    if args.flag == 'hip':
        outChannels = 6
    if args.flag != 'hip':
        outChannels = 4
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=1, out_channels=outChannels, init_features=32, pretrained=False)
    if torch.cuda.device_count() > 1:
        # model = nn.DataParallel(model, device_ids=[2, 1, 0])
        # model.to(f'cuda:{model.device_ids[0:2]}')
        model = nn.DataParallel(model)
    model.to(device)
    # summary(model, (3, args.resolution, args.resolution))
    # print(summary(model, torch.zeros((args.batchSize, 3, 512, 512),dtype=float)))
    total = sum([param.nelement() for param in model.parameters()])
    # print('  + Number of params: %.2fM' % (total / 1e6))
    print('Number of parameters: ', total)
    # trainer object
    trainer = Trainer(model, device, args)
    # train model
    trainer.train()


if __name__ == '__main__':
    main()
