import torch
import torch.nn as nn
from layers import BasicBlock, BnReluConv, DeepWiseConv, Transition


class Network(nn.Module):
    def __init__(self, args):
        self.nin = args.nin
        if args.flag == 'hip':
            self.classNum = 6
        if args.flag != 'hip':
            self.classNum = 4
        super().__init__()
        self.basicBlock1 = BasicBlock(nin=self.nin, nout=128)
        # self.basicBlock2 = BasicBlock(nin=128, nout=256)
        self.conv1 = DeepWiseConv(nin=256, nout=512, kernelSize=3)
        # self.transition1 = Transition(nin=512, nout=256)
        self.transition1 = Transition(nin=512, nout=256)
        self.conv2 = DeepWiseConv(nin=256, nout=256, kernelSize=3)
        self.transition2 = Transition(nin=256, nout=self.classNum)

        # self.conv5 = DeepWiseConv(nin=128, nout=128, kernelSize=3)
        self.convTranspose2D1 = nn.ConvTranspose2d(self.classNum, self.classNum, 8, 4)
        self.convTranspose2D2 = nn.ConvTranspose2d(self.classNum, self.classNum, 4, 4)
        self.convTranspose2D3 = nn.ConvTranspose2d(self.classNum, self.classNum, 2, 2)



        # self.fc1 = nn.Sequential(
        #
        #     nn.Linear(7 * 7 * 128, args.fcNodes),
        #     nn.ReLU(),
        #     # nn.Dropout2d(p=0.5)
        # )
        # self.fc2 = nn.Sequential(
        #     nn.Linear(args.fcNodes, args.fcNodes),
        #     nn.ReLU(),
        #     # nn.Dropout2d(p=0.5)
        # )
        # if args.flag == 'hip':
        #     self.fc3 = nn.Sequential(
        #         nn.Linear(args.fcNodes, 12)
        #     )
        # else:
        #     self.fc3 = nn.Sequential(
        #         nn.Linear(args.fcNodes, 8)
        #     )

    def forward(self, x):

        out = self.basicBlock1(x)
        # print('out.shape of basicBlock1: ', out.shape)
        # out = self.basicBlock2(out)
        # print('out.shape of basicBlock2: ', out.shape)
        out = self.conv1(out)
        # print('out.shape of conv1: ', out.shape)
        out = self.transition1(out)
        # print('out.shape of transition1: ', out.shape)
        out = self.conv2(out)
        # print('out.shape of conv2: ', out.shape)
        out = self.transition2(out)
        # print('out.shape of transition2: ', out.shape)
        out = self.convTranspose2D1(out)
        # print('after transpose2D', out.shape)

        out = self.convTranspose2D2(out)
        # print('after transpose2D', out.shape)
        out = self.convTranspose2D3(out)
        # print('after transpose2D', out.shape)

        out = out.view(out.size(0), -1)
        # print(out.size)
        # out = self.fc1(out)
        # out = self.fc2(out)
        # out = self.fc3(out)
        return out
