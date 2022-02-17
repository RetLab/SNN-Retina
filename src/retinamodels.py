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
    # check exceed membrane potential and reset
    # membrane_potential  = nn.functional.threshold(membrane_potential,0,0)
    ex_membrane = nn.functional.threshold(membrane_potential, threshold, 0)
    membrane_potential = membrane_potential - ex_membrane
    # generate spike
    out = SpikingNN.apply(ex_membrane)
    # decay 
    # note: The detach function is used in original code,but in ASF backward doesn't need it.
    membrane_potential = l * membrane_potential.detach() + membrane_potential - membrane_potential.detach()

    return membrane_potential, out

# 3D CNN
class cnn3d_NI1(nn.Module):
    def __init__(self, channel):
        super(cnn3d_NI1, self).__init__()
        self.conv1 = nn.Conv3d(1, channel[0], (3,15,15), 2)
        self.conv2 = nn.Conv3d(channel[0], channel[1], (3,15,15), 2)
        self.conv3 = nn.Conv3d(channel[1], channel[2], (3,15,15), 2)
        self.fc1 = nn.Linear(int(2048 * channel[2] / 64), 512)
        self.fc2 = nn.Linear(512, 38)

    def forward(self, x):
        # here the depth will as channel for avg_pool2d, so it is ok
        x = F.avg_pool2d(x, 2)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

# deep cnn for grating -------------- still 99.9 channel 16-32
class cnn3d_grating(nn.Module):
    def __init__(self, channel):
        super(cnn3d_grating, self).__init__()
        self.conv1 = nn.Conv3d(1, channel[0], (3,15,15), 2)
        self.conv2 = nn.Conv3d(channel[0], channel[1], (3,15,15), 2)
        self.conv3 = nn.Conv3d(channel[1], channel[2], (3,15,15), 2)
        self.fc1 = nn.Linear(int(21632 * channel[2] / 16), 512)
        self.fc2 = nn.Linear(512, 54)

    def forward(self, x):
        # here the depth will as channel for avg_pool2d, so it is ok
        x = F.avg_pool2d(x, 2)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x




# 3D CNN + SNN
class cnn3dsnn_grating(nn.Module):
    def __init__(self, channel):
        super(cnn3dsnn_grating, self).__init__()
        self.conv1 = nn.Conv3d(1, channel[0], (3,15,15), 2)
        self.conv2 = nn.Conv3d(channel[0], channel[1], (3,15,15), 2)
        self.conv3 = nn.Conv3d(channel[1], channel[2], (3,15,15), 2)
        self.fc1 = nn.Linear(int(21632*channel[2]/16), 512)
        self.fc2 = nn.Linear(512, 54)

    def forward(self, x):

        membrane_f0 = torch.zeros(x.size(0), 512, device = x.device, requires_grad=True)
        membrane_f1 = torch.zeros(x.size(0), 54, device = x.device, requires_grad=True)
        spike_sum = torch.zeros(x.size(0), 54, device = x.device, requires_grad=True)
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
            membrane_f1, out = LIF_sNeuron(membrane_f1, 1)

            spike_sum = spike_sum + out 
        return spike_sum

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


# deep cnn for natural image
class cnn3dsnn_NI1(nn.Module):
    def __init__(self, channel):
        super(cnn3dsnn_NI1, self).__init__()
        self.conv1 = nn.Conv3d(1, channel[0], (3,15,15), 2)
        self.conv2 = nn.Conv3d(channel[0], channel[1], (3,15,15), 2)
        self.conv3 = nn.Conv3d(channel[1], channel[2], (3,15,15), 2)
        self.fc1 = nn.Linear(int(2048 * channel[2] / 64), 512)
        self.fc2 = nn.Linear(512, 38)

    def forward(self, x): # x with shape 10*30*256*256
        # here the depth will as channel for avg_pool2d, so it is ok
        membrane_f0 = torch.zeros(x.size(0), 512, device = x.device, requires_grad=True)
        membrane_f1 = torch.zeros(x.size(0), 38, device = x.device, requires_grad=True)
        spike_sum = torch.zeros(x.size(0), 38, device = x.device, requires_grad=True)

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
            membrane_f1, out = LIF_sNeuron(membrane_f1, 1)

            spike_sum = spike_sum + out 

        return spike_sum


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

'''cnn3dsnn model with feedback'''
# 1. feedback to first fc
# 3D CNN + SNN
class cnn3dsnn_grating_fb(nn.Module):
    def __init__(self, channel):
        super(cnn3dsnn_grating_fb, self).__init__()
        self.conv1 = nn.Conv3d(1, channel[0], (3,15,15), 2)
        self.conv2 = nn.Conv3d(channel[0], channel[1], (3,15,15), 2)
        self.conv3 = nn.Conv3d(channel[1], channel[2], (3,15,15), 2)
        self.fc1 = nn.Linear(int(21632*channel[2]/16), 512)
        self.fc2 = nn.Linear(512, 54)
        self.feedback = nn.Linear(54, 512)

    def forward(self, x):

        membrane_f0 = torch.zeros(x.size(0), 512, device = x.device, requires_grad=True)
        membrane_f1 = torch.zeros(x.size(0), 54, device = x.device, requires_grad=True)
        # spike_t = torch.zeros(x.size(0), 54, device = x.device, requires_grad=True)
        spike_sum = torch.zeros(x.size(0), 54, device = x.device, requires_grad=True)
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

            membrane_f0 = membrane_f0 + self.fc1(Poisson_d_input) + self.feedback(spike_sum/(i+1e-5))
            membrane_f0, out = LIF_sNeuron(membrane_f0, 1)

            membrane_f1 = membrane_f1 + self.fc2(out)
            membrane_f1, out = LIF_sNeuron(membrane_f1, 1)
            # spike_t = out 

            spike_sum = spike_sum + out 
        return spike_sum

# mem
class cnn3dsnn_grating_mem_fb(nn.Module):
    def __init__(self, channel):
        super(cnn3dsnn_grating_mem_fb, self).__init__()
        self.conv1 = nn.Conv3d(1, channel[0], (3,15,15), 2)
        self.conv2 = nn.Conv3d(channel[0], channel[1], (3,15,15), 2)
        self.conv3 = nn.Conv3d(channel[1], channel[2], (3,15,15), 2)
        self.fc1 = nn.Linear(int(21632 * channel[2] / 16), 512)
        self.fc2 = nn.Linear(512, 54)
        self.feedback = nn.Linear(54, 512)

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

            membrane_f0 = membrane_f0 + self.fc1(Poisson_d_input) + self.feedback(membrane_f1/(i+1e-5))
            membrane_f0, out = LIF_sNeuron(membrane_f0, 1)

            membrane_f1 = membrane_f1 + self.fc2(out)

        return membrane_f1
        # return spike_sum


# deep cnn for natural image
class cnn3dsnn_NI1_fb(nn.Module):
    def __init__(self, channel):
        super(cnn3dsnn_NI1_fb, self).__init__()
        self.conv1 = nn.Conv3d(1, channel[0], (3,15,15), 2)
        self.conv2 = nn.Conv3d(channel[0], channel[1], (3,15,15), 2)
        self.conv3 = nn.Conv3d(channel[1], channel[2], (3,15,15), 2)
        self.fc1 = nn.Linear(int(2048 * channel[2] / 64), 512)
        self.fc2 = nn.Linear(512, 38)
        self.feedback = nn.Linear(38, 512)

    def forward(self, x): # x with shape 10*30*256*256
        # here the depth will as channel for avg_pool2d, so it is ok
        membrane_f0 = torch.zeros(x.size(0), 512, device = x.device, requires_grad=True)
        membrane_f1 = torch.zeros(x.size(0), 38, device = x.device, requires_grad=True)
        spike_sum = torch.zeros(x.size(0), 38, device = x.device, requires_grad=True)

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

            membrane_f0 = membrane_f0 + self.fc1(Poisson_d_input) + self.feedback(spike_sum/(i+1e-5))
            membrane_f0, out = LIF_sNeuron(membrane_f0, 1)

            membrane_f1 = membrane_f1 + self.fc2(out)
            membrane_f1, out = LIF_sNeuron(membrane_f1, 1)

            spike_sum = spike_sum + out 

        return spike_sum


class cnn3dsnn_NI1_mem_fb(nn.Module):
    def __init__(self, channel):
        super(cnn3dsnn_NI1_mem_fb, self).__init__()
        self.conv1 = nn.Conv3d(1, channel[0], (3,15,15), 2)
        self.conv2 = nn.Conv3d(channel[0], channel[1], (3,15,15), 2)
        self.conv3 = nn.Conv3d(channel[1], channel[2], (3,15,15), 2)
        self.fc1 = nn.Linear(int(2048*channel[2]/64), 512)
        self.fc2 = nn.Linear(512, 38)
        self.feedback = nn.Linear(38, 512)

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

            membrane_f0 = membrane_f0 + self.fc1(Poisson_d_input) + self.feedback(membrane_f1/(i+1e-5))
            membrane_f0, out = LIF_sNeuron(membrane_f0, 1)

            membrane_f1 = membrane_f1 + self.fc2(out)

        return membrane_f1


# 2dcnnsnn
class cnn2dsnn_grating(nn.Module):
    def __init__(self, channel):
        super(cnn2dsnn_grating, self).__init__()
        self.conv1 = nn.Conv2d(1, channel[0], (15,15), 2)
        self.conv2 = nn.Conv2d(channel[0], channel[1], (15,15), 2)
        self.conv3 = nn.Conv2d(channel[1], channel[2], (15,15), 2)
        self.fc1 = nn.Linear(int(21632*channel[2]/32), 1024)
        self.fc2 = nn.Linear(1024, 54)

    def forward(self, x):

        membrane_f0 = torch.zeros(x.size(0), 1024, device = x.device, requires_grad=True)
        membrane_f1 = torch.zeros(x.size(0), 54, device = x.device, requires_grad=True)
        spike_sum = torch.zeros(x.size(0), 54, device = x.device, requires_grad=True)
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
            membrane_f1, out = LIF_sNeuron(membrane_f1, 1)

            spike_sum = spike_sum + out 
        # return membrane_f1
        return spike_sum


# conv2d + snn (Grating)
class cnn2dsnn_grating_mem(nn.Module):
    def __init__(self, channel):
        super(cnn2dsnn_grating_mem, self).__init__()
        self.conv1 = nn.Conv2d(1, channel[0], (15,15), 2)
        self.conv2 = nn.Conv2d(channel[0], channel[1], (15,15), 2)
        self.conv3 = nn.Conv2d(channel[1], channel[2], (15,15), 2)
        self.fc1 = nn.Linear(int(21632*channel[2]/32), 1024)
        self.fc2 = nn.Linear(1024, 54)

    def forward(self, x):

        membrane_f0 = torch.zeros(x.size(0), 1024, device = x.device, requires_grad=True)
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


class cnn2dsnn_NI1(nn.Module):
    def __init__(self, channel):
        super(cnn2dsnn_NI1, self).__init__()
        self.conv1 = nn.Conv2d(1, channel[0], 10, 2)
        self.conv2 = nn.Conv2d(channel[0], channel[1], 10, 2)
        self.conv3 = nn.Conv2d(channel[1], channel[2], 10, 2)
        self.fc1 = nn.Linear(int(2592*channel[2]/32), 512)
        self.fc2 = nn.Linear(512, 38)

    def forward(self, x):

        membrane_f0 = torch.zeros(x.size(0), 512, device = x.device, requires_grad=True)
        membrane_f1 = torch.zeros(x.size(0), 38, device = x.device, requires_grad=True)
        spike_sum = torch.zeros(x.size(0), 38, device = x.device, requires_grad=True)
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
            membrane_f1, out = LIF_sNeuron(membrane_f1, 1)

            spike_sum = spike_sum + out 
        # return membrane_f1
        return spike_sum

# conv2d + snn (Natural Image)
class cnn2dsnn_NI1_mem(nn.Module):
    def __init__(self, channel):
        super(cnn2dsnn_NI1_mem, self).__init__()
        self.conv1 = nn.Conv2d(1, channel[0], 10, 2)
        self.conv2 = nn.Conv2d(channel[0], channel[1], 10, 2)
        self.conv3 = nn.Conv2d(channel[1], channel[2], 10, 2)
        self.fc1 = nn.Linear(int(2592*channel[2]/32), 512)
        self.fc2 = nn.Linear(512, 38)

    def forward(self, x):

        membrane_f0 = torch.zeros(x.size(0), 512, device = x.device, requires_grad=True)
        membrane_f1 = torch.zeros(x.size(0), 38, device = x.device, requires_grad=True)
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

# 2dcnnlstm
class cnn2dlstm_grating(nn.Module):
    def __init__(self, channel):
        super(cnn2dlstm_grating, self).__init__()
        self.conv1 = nn.Conv2d(1, channel[0], 15, 2)
        self.conv2 = nn.Conv2d(channel[0], channel[1], 15, 2)
        self.conv3 = nn.Conv2d(channel[1], channel[2], 15, 2)
        self.lstm = nn.LSTM(channel[2]*26*26,54,3,batch_first=True)

    def forward(self, x):
        # here the depth will as channel for avg_pool2d, so it is ok
        # breakpoint()
        batch = x.size(0)
        steps = x.size(1)
        x = F.avg_pool2d(x, 2)
        x = torch.flatten(x, start_dim=0, end_dim=1)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = x.reshape(batch,steps, *x.shape[1:])
        x_lstm = torch.flatten(x, 2)
        output, (hn, cn) = self.lstm(x_lstm)

        return torch.sum(output,1)


class cnn2dlstm_NI1(nn.Module):
    def __init__(self, channel):
        super(cnn2dlstm_NI1, self).__init__()
        self.conv1 = nn.Conv2d(1, channel[0], 5, 2)
        self.conv2 = nn.Conv2d(channel[0], channel[1], 5, 2)
        self.conv3 = nn.Conv2d(channel[1], channel[2], 5, 2)
        self.lstm = nn.LSTM(int(6728*channel[2]/16),38,5,batch_first=True)

    def forward(self, x):
        # here the depth will as channel for avg_pool2d, so it is ok
        # breakpoint()
        batch = x.size(0)
        steps = x.size(1)
        x = F.avg_pool2d(x, 2)
        x = torch.flatten(x, start_dim=0, end_dim=1)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = x.reshape(batch,steps, *x.shape[1:])
        x_lstm = torch.flatten(x, 2)
        output, (hn, cn) = self.lstm(x_lstm)

        return torch.sum(output,1)


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
        # breakpoint()
        # batch = x.size(0)
        # steps = x.size(1)
        x = F.avg_pool2d(x, 2)
        # x = torch.flatten(x, start_dim=0, end_dim=1)
        
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
        # breakpoint()

        # batch = x.size(0)
        # steps = x.size(1)
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


# SNN
class snn_grating(nn.Module):
    def __init__(self, channel):
        super(snn_grating, self).__init__()
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
        spike_sum = torch.zeros(x.size(0), 54, device = x.device, requires_grad=True)
        # here the depth will as channel for avg_pool2d, so it is ok
        # breakpoint()
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
            membrane_f1, out = LIF_sNeuron(membrane_f1, 1)

            spike_sum = spike_sum + out 

        # return membrane_f1
        return spike_sum

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
        # here the depth will as channel for avg_pool2d, so it is ok
        # breakpoint()
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
class snn_NI1(nn.Module):
    def __init__(self, channel):
        super(snn_NI1, self).__init__()
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
        spike_sum = torch.zeros(x.size(0), 38, device = x.device, requires_grad=True)
        # breakpoint()

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
            membrane_f1, out = LIF_sNeuron(membrane_f1, 1)

            spike_sum = spike_sum + out 

        return spike_sum
 

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
        # breakpoint()

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


# LN model

class LN_grating(nn.Module):
    '''
    receive image as stimulus

    input: (150,150), size of a single image

    '''
    def __init__(self, channel=None):
        super(LN_grating, self).__init__()
        self.fc = nn.Linear(150*150, 54)

    def forward(self, x):
        x = F.avg_pool2d(x, 4)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.relu(x)
        return x

class LN_grating_clip(nn.Module):
    '''
    receive clip as stimulus

    input: (150,150,30), size of a single clip, 30 is the temporal dimension
    '''
    def __init__(self, channel=None):
        super(LN_grating_clip, self).__init__()
        self.fc = nn.Linear(150*150*30, 54)

    def forward(self, x):
        x = F.avg_pool2d(x, 4)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.relu(x)
        return x


class LN_NI1(nn.Module):
    '''
    receive image as stimulus

    input: (150,150), size of a single image

    '''
    def __init__(self, channel=None):
        super(LN_NI1, self).__init__()
        self.fc = nn.Linear(128*128, 38)

    def forward(self, x):
        x = F.avg_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.relu(x)
        return x

class LN_NI1_clip(nn.Module):
    '''
    receive clip as stimulus

    input: (150,150,30), size of a single clip, 30 is the temporal dimension
    '''
    def __init__(self, channel=None):
        super(LN_NI1_clip, self).__init__()
        self.fc = nn.Linear(128*128*30, 38)

    def forward(self, x):
        x = F.avg_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.relu(x)
        return x


class NIM_grating(nn.Module):
    '''
    receive image as stimulus

    input: (150,150), size of a single image

    '''
    def __init__(self, channel=None):
        super(NIM_grating, self).__init__()
        self.fc1 = nn.Linear(150*150, 1024)
        self.fc2 = nn.Linear(1024, 54)

    def forward(self, x):
        x = F.avg_pool2d(x, 4)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x

class NIM_grating_clip(nn.Module):
    '''
    receive clip as stimulus

    input: (150,150,30), size of a single clip, 30 is the temporal dimension
    '''
    def __init__(self, channel=None):
        super(NIM_grating_clip, self).__init__()
        self.fc1 = nn.Linear(150*150*30, 2048)
        self.fc2 = nn.Linear(2048, 54)

    def forward(self, x):
        x = F.avg_pool2d(x, 4)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x


class NIM_NI1(nn.Module):
    '''
    receive image as stimulus

    input: (150,150), size of a single image

    '''
    def __init__(self, channel=None):
        super(NIM_NI1, self).__init__()
        self.fc1 = nn.Linear(128*128, 1024)
        self.fc2 = nn.Linear(1024, 38)

    def forward(self, x):
        x = F.avg_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x

class NIM_NI1_clip(nn.Module):
    '''
    receive clip as stimulus

    input: (150,150,30), size of a single clip, 30 is the temporal dimension
    '''
    def __init__(self, channel=None):
        super(NIM_NI1_clip, self).__init__()
        self.fc1 = nn.Linear(128*128*30, 2048)
        self.fc2 = nn.Linear(2048, 38)

    def forward(self, x):
        x = F.avg_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x

# class NIM_clip(object):

#     def __init__(self)：
#         super(NIM, self).__init__()
#         self.subunit = nn.conv3d(1, channel[0], (3,15,15), 2)
#         self.linear = nn.Linear(, 54)
#     def forward(self, x):
#         x = F.avg_pool2d(x, 2)
#         x = self.subunit(x)
#         x = F.relu(x)
#         x = torch.flatten(x, 1)
#         x = self.linear(x)
#         x = F.relu(x)
#         return x

# confused by where to add the distribution and the meaning of the distribution (no distribution here, when adding no spike history)
class GLM_grating(nn.Module):
    """docstring for GLM_grating"""
    def __init__(self, neurons=54, dt=4):
        '''
        dt: the length of spike history
        '''
        super(GLM_grating, self).__init__()
        self.neurons = neurons
        self.dt = dt 
        self.l_weight = nn.Parameter(data=torch.rand([neurons, dt]).cuda())
        self.h_weight = nn.Parameter(data=torch.rand(dt).cuda())

        self.fc = nn.Linear(75*75, 54)
        # add link function and probability function later

    def forward(self, x):
        spike_count = torch.zeros(x.shape[0], self.neurons, device=x.device)
        spike_history = torch.zeros(x.shape[0], self.neurons, self.dt, device=x.device)


        # For real output, the stimulus is same for all time step
        x = F.avg_pool2d(x, 8)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        for i in range(30):    # 300 ms, a time step = 10 ms
            history_sum = spike_history * self.l_weight
            # pdb.set_trace()
            lateral_sum = torch.sum(history_sum, dim=(1, 2)).unsqueeze(1) - torch.sum(history_sum, dim=2)
            current_sum = x + lateral_sum + torch.sum(spike_history * self.h_weight, dim=2)
            # pdb.set_trace()

            rate = torch.exp(current_sum)
            _, spike_t = LIF_sNeuron(rate, rate.mean().detach())
            # spike_t = rate / self.dt

            spike_history = torch.roll(spike_history, -1, 2)
            spike_history[:,:,-1] = spike_t
            spike_count = spike_count + spike_t 
        return F.softplus(spike_count)

class GLM_grating_clip(nn.Module):
    """docstring for GLM_grating"""
    def __init__(self, neurons=54, dt=4):
        '''
        dt: the length of spike history
        '''
        super(GLM_grating_clip, self).__init__()
        self.neurons = neurons
        self.dt = dt 
        self.l_weight = nn.Parameter(data=torch.rand([neurons, dt]).cuda())
        self.h_weight = nn.Parameter(data=torch.rand(dt).cuda())

        self.fc = nn.Linear(75*75, 54)
        # add link function and probability function later

    def forward(self, x):
        spike_count = torch.zeros(x.shape[0], self.neurons, device=x.device)
        spike_history = torch.zeros(x.shape[0], self.neurons, self.dt, device=x.device)

        # For real output, the stimulus is same for all time step
        
        x = F.avg_pool2d(x, 8)

        for i in range(30):    # 300 ms, a time step = 10 ms
            frame = x[:,i]
            frame = torch.flatten(frame, 1)
            frame = self.fc(frame)
            history_sum = spike_history * self.l_weight
            lateral_sum = torch.sum(history_sum, dim=(1, 2)).unsqueeze(1) - torch.sum(history_sum, dim=2)
            current_sum = frame + lateral_sum + torch.sum(spike_history * self.h_weight, dim=2)
            rate = torch.exp(current_sum)
            _, spike_t = LIF_sNeuron(rate, rate.mean().detach())
            # spike_t = rate / self.dt

            spike_history = torch.roll(spike_history, -1, 2)
            spike_history[:,:,-1] = spike_t
            spike_count = spike_count + spike_t 
        return F.softplus(spike_count)



class GLM_NI1(nn.Module):
    """docstring for GLM_grating"""
    def __init__(self, neurons=38, dt=4):
        '''
        dt: the length of spike history
        '''
        super(GLM_NI1, self).__init__()
        self.neurons = neurons
        self.dt = dt 
        self.l_weight = nn.Parameter(data=torch.rand([neurons, dt]))
        self.h_weight = nn.Parameter(data=torch.rand(dt))

        self.fc = nn.Linear(64*64, 38)
        # add link function and probability function later

    def forward(self, x):
        spike_count = torch.zeros(x.shape[0], self.neurons, device=x.device)
        spike_history = torch.zeros(x.shape[0], self.neurons, self.dt, device=x.device)


        # For real output, the stimulus is same for all time step
        x = F.avg_pool2d(x, 4)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # pdb.set_trace()

        for i in range(30):    # 300 ms, a time step = 10 ms
            # pdb.set_trace()
            history_sum = spike_history * self.l_weight
            # pdb.set_trace()
            lateral_sum = torch.sum(history_sum, dim=(1, 2)).unsqueeze(1) - torch.sum(history_sum, dim=2)
            current_sum = x + lateral_sum + torch.sum(spike_history * self.h_weight, dim=2)
            rate = torch.exp(current_sum)
            _, spike_t = LIF_sNeuron(rate, rate.mean().detach())
            # spike_t = rate / self.dt

            spike_history = torch.roll(spike_history, -1, 2) 
            spike_history[:,:,-1] = spike_t
            spike_count = spike_count + spike_t 
        return F.softplus(spike_count)

class GLM_NI1_clip(nn.Module):
    """docstring for GLM_grating"""
    def __init__(self, neurons=38, dt=10):
        '''
        dt: the length of spike history
        '''
        super(GLM_NI1_clip, self).__init__()
        self.neurons = neurons
        self.dt = dt 
        self.l_weight = nn.Parameter(data=torch.rand([neurons, dt]))
        self.h_weight = nn.Parameter(data=torch.rand([dt]))

        self.fc = nn.Linear(64*64, 38)
        # add link function and probability function later

    def forward(self, x):
        spike_count = torch.zeros(x.shape[0], self.neurons, device=x.device)
        spike_history = torch.zeros(x.shape[0], self.neurons, self.dt, device=x.device)

        # For real output, the stimulus is same for all time step        
        x = F.avg_pool2d(x, 4)

        for i in range(30):    # 300 ms, a time step = 10 ms
            frame = x[:,i]
            frame = torch.flatten(frame, 1)
            frame = self.fc(frame)
            history_sum = spike_history * self.l_weight
            lateral_sum = torch.sum(history_sum, dim=(1, 2)).unsqueeze(1) - torch.sum(history_sum, dim=2)
            current_sum = frame + lateral_sum + torch.sum(spike_history * self.h_weight, dim=2)
            rate = torch.exp(current_sum)


            _, spike_t = LIF_sNeuron(rate, rate.mean().detach())
            # spike_t = F.hardsigmoid(current_sum) 

            # spike_t = rate / self.dt
            # spike_t = torch.poisson(rate)
            spike_history = torch.roll(spike_history, -1, 2)
            spike_history[:,:,-1] = spike_t
            spike_count = spike_count + spike_t 
        return F.softplus(spike_count)

