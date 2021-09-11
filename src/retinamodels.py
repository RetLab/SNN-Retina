'''
Retina models used to reproduce stimulus-response map of retina
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.rnn import LSTM




class SpikingNN(torch.autograd.Function):
    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        return input.gt(0).type(torch.cuda.FloatTensor)

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input <= 0.0] = 0
        return grad_input


def LIF_sNeuron(membrane_potential, threshold, l=1):
    ex_membrane = nn.functional.threshold(membrane_potential, threshold, 0)
    membrane_potential = membrane_potential - ex_membrane
    out = SpikingNN.apply(ex_membrane)
    membrane_potential = l * membrane_potential.detach() + membrane_potential - membrane_potential.detach()

    return membrane_potential, out


class cnn3dsnn_grating_mem(nn.Module):
    def __init__(self, channel):
        super(cnn3dsnn_grating_mem, self).__init__()
        self.conv1 = nn.Conv3d(1, channel[0], (3,15,15), 2)
        self.conv2 = nn.Conv3d(channel[0], channel[1], (3,15,15), 2)
        self.conv3 = nn.Conv3d(channel[1], channel[2], (3,15,15), 2)
        self.fc1 = nn.Linear(int(21632 * channel[2] / 16), 512)
        self.fc2 = nn.Linear(512, 54)

    def forward(self, x):

        membrane_f0 = torch.zeros(x.size(0), 512, device = x.device, requires_grad=True)
        membrane_f1 = torch.zeros(x.size(0), 54, device = x.device, requires_grad=True)
        # here the depth will as channel for avg_pool2d, so it is ok
        # breakpoint()
        x = F.avg_pool2d(x, 2)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)
        x_snnin = torch.flatten(x, 1)


        for i in range(30):
            rand_num = torch.rand(x_snnin.size(0), x_snnin.size(1), device = x_snnin.device)
            Poisson_d_input = (torch.abs(x_snnin)/2) > rand_num
            Poisson_d_input = torch.mul(Poisson_d_input.float(), torch.sign(x_snnin)).detach() + x_snnin - x_snnin.detach()

            membrane_f0 = membrane_f0 + self.fc1(Poisson_d_input)
            membrane_f0, out = LIF_sNeuron(membrane_f0, 1)

            membrane_f1 = membrane_f1 + self.fc2(out)

        return membrane_f1
        # return spike_sum

class cnn3dsnn_NI1_mem(nn.Module):
    def __init__(self, channel):
        super(cnn3dsnn_NI1_mem, self).__init__()
        self.conv1 = nn.Conv3d(1, channel[0], (3,15,15), 2)
        self.conv2 = nn.Conv3d(channel[0], channel[1], (3,15,15), 2)
        self.conv3 = nn.Conv3d(channel[1], channel[2], (3,15,15), 2)
        self.fc1 = nn.Linear(int(2048*channel[2]/64), 512)
        self.fc2 = nn.Linear(512, 38)

    def forward(self, x): # x with shape 10*30*256*256
        # here the depth will as channel for avg_pool2d, so it is ok
        membrane_f0 = torch.zeros(x.size(0), 512, device = x.device, requires_grad=True)
        membrane_f1 = torch.zeros(x.size(0), 38, device = x.device, requires_grad=True)

        x = F.avg_pool2d(x, 2)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)

        # Input of SNN block, then generate spike using Poisson distribution
        x_snnin = torch.flatten(x, 1)
        for i in range(30):
            rand_num = torch.rand(x_snnin.size(0), x_snnin.size(1), device = x_snnin.device)
            Poisson_d_input = (torch.abs(x_snnin)/2) > rand_num
            Poisson_d_input = torch.mul(Poisson_d_input.float(), torch.sign(x_snnin)) + x_snnin - x_snnin.detach()

            membrane_f0 = membrane_f0 + self.fc1(Poisson_d_input)
            membrane_f0, out = LIF_sNeuron(membrane_f0, 1)

            membrane_f1 = membrane_f1 + self.fc2(out)

        return membrane_f1


# SNN-LSTM Model
class snnlstm_grating(nn.Module):
    def __init__(self, channel):
        super(snnlstm_grating, self).__init__()
        self.conv1 = nn.Conv2d(1, channel[0], 15, 2)
        self.conv2 = nn.Conv2d(channel[0], channel[1], 15, 2)
        self.conv3 = nn.Conv2d(channel[1], channel[2], 15, 2)
        self.lstm = nn.LSTM(channel[2]*26*26,54,3,batch_first=True)
        self.channel = channel

    def forward(self, x):
        # here the depth will as channel for avg_pool2d, so it is ok
        membrane_c1 = torch.zeros(x.size(0), self.channel[0], 143, 143, device = x.device, requires_grad=True)
        membrane_c2 = torch.zeros(x.size(0), self.channel[1], 65, 65, device = x.device, requires_grad=True)
        membrane_c3 = torch.zeros(x.size(0), self.channel[2], 26, 26, device = x.device, requires_grad=True)
        x = F.avg_pool2d(x, 2)
        
        x_snnin = x.unsqueeze(1)
        for i in range(x_snnin.shape[2]):
            frame = x_snnin[:,:,i,:,:]

            membrane_c1 = membrane_c1 + self.conv1(frame)
            membrane_c1, out = LIF_sNeuron(membrane_c1, 1)

            membrane_c2 = membrane_c2 + self.conv2(out)
            membrane_c2, out = LIF_sNeuron(membrane_c2, 1)

            membrane_c3 = membrane_c3 + self.conv3(out)
            membrane_c3, out = LIF_sNeuron(membrane_c3, 1)
            # breakpoint()
            if i == 0:
                c3o = out.unsqueeze(1)
            else:
                c3o = torch.cat((c3o, out.unsqueeze(1)), 1)

        output, (hn, cn) = self.lstm(torch.flatten(c3o,2))

        return torch.sum(output,1)


class snnlstm_NI1(nn.Module):
    def __init__(self, channel):
        super(snnlstm_NI1, self).__init__()
        self.conv1 = nn.Conv2d(1, channel[0], 10, 2)
        self.conv2 = nn.Conv2d(channel[0], channel[1], 10, 2)
        self.conv3 = nn.Conv2d(channel[1], channel[2], 10, 2)
        self.lstm = nn.LSTM(int(2592*channel[2]/32),38,3,batch_first=True)
        self.channel = channel 

    def forward(self, x):
        # here the depth will as channel for avg_pool2d, so it is ok
        membrane_c1 = torch.zeros(x.size(0), self.channel[0], 60, 60, device = x.device, requires_grad=True)
        membrane_c2 = torch.zeros(x.size(0), self.channel[1], 26, 26, device = x.device, requires_grad=True)
        membrane_c3 = torch.zeros(x.size(0), self.channel[2], 9, 9, device = x.device, requires_grad=True)
        
        x = F.avg_pool2d(x, 2)
        x_snnin = x.unsqueeze(1)
        for i in range(x_snnin.shape[2]):
            frame = x_snnin[:,:,i,:,:]

            membrane_c1 = membrane_c1 + self.conv1(frame)
            membrane_c1, out = LIF_sNeuron(membrane_c1, 1)

            membrane_c2 = membrane_c2 + self.conv2(out)
            membrane_c2, out = LIF_sNeuron(membrane_c2, 1)

            membrane_c3 = membrane_c3 + self.conv3(out)
            membrane_c3, out = LIF_sNeuron(membrane_c3, 1)
            if i == 0:
                c3o = out.unsqueeze(1)
            else:
                c3o = torch.cat((c3o, out.unsqueeze(1)), 1)

        output, (hn, cn) = self.lstm(torch.flatten(c3o,2))

        return torch.sum(output,1)



class snn_grating_mem(nn.Module):
    def __init__(self, channel):
        super(snn_grating_mem, self).__init__()
        self.conv1 = nn.Conv2d(1, channel[0], 15, 2)
        self.conv2 = nn.Conv2d(channel[0], channel[1], 15, 2)
        self.conv3 = nn.Conv2d(channel[1], channel[2], 15, 2)
        self.fc1 = nn.Linear(int(10816*channel[2]/16), 512)
        self.fc2 = nn.Linear(512, 54)
        self.channel = channel

    def forward(self, x):

        membrane_c1 = torch.zeros(x.size(0), self.channel[0], 143, 143, device = x.device, requires_grad=True)
        membrane_c2 = torch.zeros(x.size(0), self.channel[1], 65, 65, device = x.device, requires_grad=True)
        membrane_c3 = torch.zeros(x.size(0), self.channel[2], 26, 26, device = x.device, requires_grad=True)
        membrane_f0 = torch.zeros(x.size(0), 512, device = x.device, requires_grad=True)
        membrane_f1 = torch.zeros(x.size(0), 54, device = x.device, requires_grad=True)
        
        x = F.avg_pool2d(x, 2)
        x_snnin = x.unsqueeze(1)
        for i in range(x_snnin.shape[2]):
            frame = x_snnin[:,:,i,:,:]

            membrane_c1 = membrane_c1 + self.conv1(frame)
            membrane_c1, out = LIF_sNeuron(membrane_c1, 1)

            membrane_c2 = membrane_c2 + self.conv2(out)
            membrane_c2, out = LIF_sNeuron(membrane_c2, 1)

            membrane_c3 = membrane_c3 + self.conv3(out)
            membrane_c3, out = LIF_sNeuron(membrane_c3, 1)
            
            x = torch.flatten(out, 1)

            membrane_f0 = membrane_f0 + self.fc1(x)
            membrane_f0, out = LIF_sNeuron(membrane_f0, 1)

            membrane_f1 = membrane_f1 + self.fc2(out)

        return membrane_f1
 

# Pure SNN for natural image
class snn_NI1_mem(nn.Module):
    def __init__(self, channel):
        super(snn_NI1_mem, self).__init__()
        self.conv1 = nn.Conv2d(1, channel[0], 15, 2)
        self.conv2 = nn.Conv2d(channel[0], channel[1], 15, 2)
        self.conv3 = nn.Conv2d(channel[1], channel[2], 15, 2)
        self.fc1 = nn.Linear(int(1024*channel[2]/64), 512)
        self.fc2 = nn.Linear(512, 38)
        self.channel = channel



    def forward(self, x):
        # here the depth will as channel for avg_pool2d, so it is ok
        membrane_c1 = torch.zeros(x.size(0), self.channel[0], 57, 57, device = x.device, requires_grad=True)
        membrane_c2 = torch.zeros(x.size(0), self.channel[1], 22, 22, device = x.device, requires_grad=True)
        membrane_c3 = torch.zeros(x.size(0), self.channel[2], 4, 4, device = x.device, requires_grad=True)
        membrane_f0 = torch.zeros(x.size(0), 512, device = x.device, requires_grad=True)
        membrane_f1 = torch.zeros(x.size(0), 38, device = x.device, requires_grad=True)

        x = F.avg_pool2d(x, 2)
        x_snnin = x.unsqueeze(1)
        for i in range(x_snnin.shape[2]):
            frame = x_snnin[:,:,i,:,:]

            membrane_c1 = membrane_c1 + self.conv1(frame)
            membrane_c1, out = LIF_sNeuron(membrane_c1, 1)

            membrane_c2 = membrane_c2 + self.conv2(out)
            membrane_c2, out = LIF_sNeuron(membrane_c2, 1)

            membrane_c3 = membrane_c3 + self.conv3(out)
            membrane_c3, out = LIF_sNeuron(membrane_c3, 1)

            x = torch.flatten(out, 1)

            membrane_f0 = membrane_f0 + self.fc1(x)
            membrane_f0, out = LIF_sNeuron(membrane_f0, 1)

            membrane_f1 = membrane_f1 + self.fc2(out)

        return membrane_f1
