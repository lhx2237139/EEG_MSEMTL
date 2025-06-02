import torch
import torch.nn as nn
import numpy as np

class Expert(nn.Module):
    def conv_block(self, in_chan, out_chan, kernel, step, pool):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan,
                      kernel_size=kernel, stride=step),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool)))

    def __init__(self, input_size, time_scale, spat_window, num_T, num_S, sampling_rate):
        super(Expert, self).__init__()
        self.pool = 8
        # 专家网络，学习共享特征
        self.MSFreq = self.conv_block(1, num_T, (1, int(time_scale * sampling_rate)), 1, self.pool)
        self.MSSpat1 = self.conv_block(num_T, num_S, (int(input_size[1]), spat_window[0]), 1, int(self.pool * 0.25))
        self.MSSpat2 = self.conv_block(num_T, num_S, (int(input_size[1] * spat_window[1]), 1), (int(input_size[1] * spat_window[1]), 1),
                                         int(self.pool * 0.25))
        self.MSSpat3 = self.conv_block(num_T, num_S, (int(input_size[1] * spat_window[2]), 1), (int(input_size[1] * spat_window[2]), 1),
                                         int(self.pool * 0.25))
        kernel = int(
            int((input_size[2] - time_scale * sampling_rate + 1) / self.pool) / (self.pool * 0.25))
        self.time_layer = self.conv_block(num_S, num_S, (1, kernel), 1, 1)
        self.BN_F = nn.BatchNorm2d(num_T)
        self.BN_S = nn.BatchNorm2d(num_S)
    def forward(self, x):
        y = self.MSFreq(x)
        y = self.BN_F(y)
        y1 = [self.MSSpat1(y),self.MSSpat2(y),self.MSSpat3(y)]
        y2 = torch.cat(y1, dim=2)
        y2 = self.BN_S(y2)
        y3 = self.time_layer(y2)
        return y3

class Tower(nn.Module):
    def conv_block(self, in_chan, out_chan, kernel, step, pool):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan,
                      kernel_size=kernel, stride=step),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool)))

    def __init__(self, num_S,hidden,dropout_rate,num_classes):
        super(Tower, self).__init__()
        self.spec_fuse = self.conv_block(num_S, num_S * 2, (1, 4), 1, 1)
        self.spat_fuse = self.conv_block(num_S * 2, num_S * 4, (7, 1), 1, 1)
        self.BN_spec_fusion = nn.BatchNorm2d(num_S * 2)
        self.BN_spat_fusion = nn.BatchNorm2d(num_S * 4)
        self.tower = nn.Sequential(
            nn.Linear(num_S * 4, hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden, num_classes)
        )
    def forward(self, x):
        y = self.spec_fuse(x)
        y = self.BN_spec_fusion(y)
        y = self.spat_fuse(y)
        y = self.BN_spat_fusion(y)
        y = torch.squeeze(y)
        y = self.tower(y)
        return y

class MSEMTL(nn.Module):
    def __init__(self, input_size, num_T, num_S, sampling_rate,hidden,dropout_rate,num_classes, num_tasks):
        super(MSEMTL, self).__init__()
        # 多尺度卷积窗
        time_window = [0.5, 0.25, 0.125, 0.083]
        spat_window = [1,0.5,0.25]
        # # 消冗实验
        # time_window = [0.5, 0.5, 0.5, 0.5]
        # spat_window = [1, 1, 1]
        # 专家集合
        self.experts = nn.ModuleList([Expert(input_size, time_scale, spat_window, num_T, num_S, sampling_rate) for time_scale in time_window])
        # 任务特定层，每个任务有自己的输出层
        self.task_specific_layers = nn.ModuleList([Tower(num_S,hidden,dropout_rate,num_classes) for _ in range(num_tasks)])

    def forward(self, x):
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(x)
            expert_outputs.append(expert_output)

        fuse_feature = torch.cat(expert_outputs,dim=-1)
        #print(fuse_feature.shape)
        y_emo = self.task_specific_layers[0](fuse_feature)
        y_cog = self.task_specific_layers[1](fuse_feature)
        out_emo = torch.cat((y_emo.unsqueeze(0), y_cog.unsqueeze(0)), dim=0)
        return out_emo