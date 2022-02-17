from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.rnn import LSTM
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader

from matplotlib.pyplot import MultipleLocator, axis
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import math
import os 
import time
import pdb
import random
from copy import deepcopy
from retinamodels import *
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def index_transfrom(index, augment):
    output = []
    for i in index:
        output.extend(range(i*augment, (i+1)*augment))
    return output



class RetinaData(Dataset):
    """
    Dataset of of retina data
    Data: Stimulus
    Target: Response of retina ganglion cells
    """
    
    def __init__(self, root, stimulus, train=True, transform=None, augment=10):
        """
        stimulus: type of stimulus (Gratings/NaturalImage1/NaturalImage2)
        """
        # self.root = os.path.expanduser(root)
        self.stimulus = stimulus
        # self.transform = transform
        self.augment = augment
        # self.target_transform = target_transform
        self.train = train

        stim, resp = get_data(root, self.stimulus, augment)

    
        

        # Here we select first 50 images as training set and left images as testing set
        if self.train:
            train_index = np.arange(50*augment)
            self.data = stim[train_index]
            self.targets = resp[train_index]
            print('load train dataset done!')

        else:
            test_index = np.arange(50*augment,60*augment)
            self.data = stim[test_index]
            self.targets = resp[test_index]
            print('load test dataset done!')

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        
        img = self.data[index]
        target = self.targets[index]
        
        
        img = torch.from_numpy(img).float()      
        target = torch.from_numpy(target).float()

        return (img, target)

    def __len__(self):
        return len(self.data) 
            
        




# Different path for bitahub and local machine

# Load Data
def get_data(DataPath, stimulus, augment=10,steps=30):
    """
    Loads data according to the specified stimulus
    augment: the times the samples will increase -> 60 samples to 600 samples
    Note the stimulus are normalized here. ToTensor transform is not needed anymore
    """
    # global DataPath
    # Stimulus and Response Path
    RespPath = os.path.join(DataPath, 'SpikeCount')
    StimPath = os.path.join(DataPath, 'Stimulus')
    # print(StimPath)
    if stimulus in ['NaturalImage1', 'NaturalImage2']:
        if augment == 1:
            stim = np.load(os.path.join(StimPath, 'NaturalImages.npy'))
            resp = np.load(os.path.join(RespPath, stimulus+'_count.npy'))
            stim = stim / 255
            # breakpoint()
        else:
            path = os.path.join(StimPath, 'NaturalImagesDA{}.npy'.format(augment))
            resp = np.load(os.path.join(RespPath, stimulus+'_count.npy'))
            resp = resp.repeat(augment,axis=0)
            if os.path.exists(path):
                stim = np.load(path)
            else:
                stim = np.load(os.path.join(StimPath, 'NaturalImages.npy'))
                stim = stim.repeat(augment,axis=0)
                stim = stim[:, np.newaxis, :, :].repeat(steps,axis=1)
            
                random_stim = np.random.randint(256,size=stim.shape, dtype=np.uint8)
                stim = (stim > random_stim)
                np.save(path, stim)

    elif stimulus == 'Gratings': 
        if augment == 1:
            stim = np.load(os.path.join(StimPath, 'Gratings.npy'))
            resp = np.load(os.path.join(RespPath, 'Gratings_count.npy'))
            # breakpoint()
            stim = stim / 255
        else:
            path = os.path.join(StimPath, 'GratingsDA{}.npy'.format(augment))
            resp = np.load(os.path.join(RespPath, 'Gratings_count.npy'))
            resp = resp.repeat([augment],axis=0)
            if os.path.exists(path):
                stim = np.load(path)
            else:
                stim = np.load(os.path.join(StimPath, 'Gratings.npy'))
                stim = stim.repeat([augment],axis=0)
                stim = stim[:, np.newaxis, :, :].repeat(steps,axis=1)
                random_stim = np.random.randint(256,size=stim.shape, dtype=np.uint8)
                stim = (stim > random_stim)
                np.save(path, stim)

    else:
        # load movie data (waiting for adding code)
        # stim = np.load(os.path.join(StimPath, 'NaturalImages.npy'))
        # resp = np.load(os.path.join(RespPath, stimulus+'_count.npy'))
        print('waiting for adding code')
        exit()
    return stim, resp 




    
def train(args, loss_f, model, device, train_loader, optimizer, epoch, k=1):
    model.train()
    train_loss = 0
    samples = 0
    right_samples = 0
    inrange_samples = 0
    samples_positive = 0
    right_samples_positive = 0
    # index = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # breakpoint()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # breakpoint()
        target_mean = target.mean(dim=2)
        target_max, _ = target.max(dim=2)
        target_min, _ = target.min(dim=2)
        mask = (target_max > 0)

        if args.weight != 0:
            loss = loss_f(output, target_mean) + args.weight * loss_f(output[mask], target_mean[mask])
        else:
            loss = loss_f(output, target_mean)
        # loss = F.mse_loss(output[output > target_max], target_max[output > target_max], reduction='sum') + F.mse_loss(output[output < target_min], target_min[output < target_min], reduction='sum')
        train_loss += loss
        output = torch.round(output)
        samples += output.numel()
        samples_positive += output[mask].numel()
        inrange_samples += ((output >= target_mean - k) & (output <= target_mean+k)).sum().item()
        right_samples += ((output >= target_min) & (output <= target_max)).sum().item()

        right_samples_positive += ((output[mask] >= target_min[mask]) & (output[mask] <= target_max[mask])).sum().item()       

        loss.backward()
        # pdb.set_trace()
        optimizer.step()

    print('Epoch:{} Training Average Loss:{:.2f}, Accuracy1:{:.3f}, Accuracy2:{:.3f}, Accuracy3:{:.3f}'.format(epoch, train_loss/(batch_idx+1), inrange_samples/samples, right_samples/samples, right_samples_positive/samples_positive))
    return

def test(args, loss_f, model, device, test_loader, k=1):
    model.eval()
    test_loss = 0
    samples = 0
    right_samples = 0
    samples_positive = 0
    inrange_samples = 0
    right_samples_positive = 0
    index = 0

    with torch.no_grad():
        for data, target in test_loader:
            # breakpoint()
#             print(data[0,0])
            data, target = data.to(device), target.to(device)
            output = model(data)
            # target.shape: batchsize * neurons * trails
            target_max, _ = target.max(dim=2)
            target_min, _ = target.min(dim=2)
            
            mask = (target_max > 0)
            target_mean = target.mean(dim=2)
            if args.weight != 0:
                loss = loss_f(output, target_mean) + args.weight * loss_f(output[mask], target_mean[mask])
            else:
                loss = loss_f(output, target_mean)
            # loss = F.mse_loss(output, target_mean,reduction='sum')
            output = torch.round(output)
            # acc = ((output >= target_min) & (output <= target_max)).sum().item() / output.numel()
            index += 1


            samples += output.numel()
            samples_positive += output[mask].numel()
            # breakpoint()
            right_samples += ((output >= target_min) & (output <= target_max)).sum().item()
            inrange_samples += ((output >= target_mean - k) & (output <= target_mean + k)).sum().item()
            right_samples_positive += ((output[mask] >= target_min[mask]) & (output[mask] <= target_max[mask])).sum().item()

            test_loss += loss 
        print('\tTesting Average Loss:{:.2f}, Accuracy1:{:.3f}, Accuracy2:{:.3f}, Accuracy3:{:.3f}'.format(test_loss/(index+1), inrange_samples/samples, right_samples/samples, right_samples_positive/samples_positive))
    return inrange_samples/samples, right_samples/samples, right_samples_positive/samples_positive, test_loss/(index+1)
        


    
def CrossCorrelation(x, y):
    time_range = int(y.shape[2] / 6)
    loss_list = [F.mse_loss(x, y)]
    for i in range(1, time_range):
        loss_list.append(F.mse_loss(x[:, :, i:], y[:, :, :(-1)*i]))
        loss_list.append(F.mse_loss(y[:, :, i:], x[:, :, :(-1)*i]))
    return loss_list

def plot_response(TaskPath, rows, columns, dataloader, model, args,device, train=False):
    Stimulus = ['grating', 'NI1', 'NI2']
#     pdb.set_trace()
    fig = plt.figure(figsize=(10*columns, 10*rows),facecolor='w')
    fig.suptitle('Movie{}  avground'.format(args.stim + 1))


    for index, (data, target) in enumerate(dataloader):
        output = model(data.to(device)) #[0:samples]
        output = torch.round(output).cpu().detach().numpy()
        
        output_shape = output.shape
        target = target.detach().numpy()
        
        for i in range(output_shape[0]):
            if train and ((index + 1) * args.batch_size + i + 1  > rows * columns):
                break
            fig.add_subplot(rows, columns, index * args.batch_size+i+1)
            predict = output[i]
            # breakpoint()
            actual = target[i]
            actual_mean = actual.mean(axis=1)
            # Note: the sort method change array itself and return None
            actual.sort(axis=-1)
            actual_max = actual[:,-1]
            actual_min = actual[:,0]
            plt.vlines(np.arange(predict.size), actual_min, actual_max,colors='red')
            plt.scatter(np.arange(predict.size), predict, c='b', label='predict')
            plt.scatter(np.arange(actual.shape[0]), actual_min, c='r', label='actual_min',marker='_')
            plt.scatter(np.arange(actual.shape[0]), actual_max, c='r', label='actual_max',marker='_')
            plt.scatter(np.arange(actual.shape[0]), actual_mean, c='g', label='actual_mean',marker='_')
            plt.yticks(np.arange(12))
            plt.title('Stimulus {}'.format(i))
            plt.legend() 
    i = 0
    if train:
        while True:
            i += 1
            # if args.mse:
            if args.clip:
                newname = os.path.join(TaskPath, 'rate_visualize', '{}_{}_clip_train_v{:d}.pdf'.format(args.arch, Stimulus[args.stim], i))
            else:
                newname = os.path.join(TaskPath, 'rate_visualize', '{}_{}_train_v{:d}.pdf'.format(args.arch, Stimulus[args.stim], i))
            if os.path.exists(newname):
                continue
            plt.savefig(newname)
            break
    else:
        while True:
            i += 1
            if args.clip:
                newname = os.path.join(TaskPath, 'rate_visualize', '{}_{}_clip_test_v{:d}.pdf'.format(args.arch, Stimulus[args.stim], i))
            else:
                newname = os.path.join(TaskPath, 'rate_visualize', '{}_{}_test_v{:d}.pdf'.format(args.arch, Stimulus[args.stim], i))
            if os.path.exists(newname):
                continue
            plt.savefig(newname)
            break     
    print(newname)

def UniqueName(path, extend_name):
    i = 1
    while os.path.exists(path + '_{}'.format(i) + extend_name):
        # path = path + '_{}'.format(i)
        i += 1
    return path + '_{}'.format(i) + extend_name

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Retina Model')
    parser.add_argument('--arch', help='Network used to model retina:1. 3dcnn 2. 3dcnnsnn 3. 2dcnnsnn 4. 2dcnnlstm 5. snnlstm 6. snn')
    parser.add_argument('--stim', type=int, default=1, metavar='N',
                        help='0:Gratings, 1:NaturalImage1, 2:NaturalImage2, else:movie(not finished)')
    parser.add_argument('--augment', type=int, default=1, metavar='N',
                        help='the times of samples')
    parser.add_argument('--batch-size', '-b', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--test-batch-size', '-bt', type=int, default=10, metavar='N',
                        help='input batch size for testing (default: 10)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 15)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.05)')
    parser.add_argument('--k', type=float, default=1, metavar='k',
                        help='the vary range of mean value (used to compute acc1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--local', action='store_true', default=False,
                    help='default in server')
    parser.add_argument('--clip', action='store_true', default=False,
                    help='use clip as input')
    parser.add_argument('--loss', default='mse', help='the type of loss function')
    parser.add_argument('--weight', type=int, default=0,metavar='N',
                    help='the weight to strength the positive term in loss')

    
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    # print(args.stimulus)
    Stimulus = ['grating', 'NI1', 'NI2']
    full_stimulus = ['Gratings', 'NaturalImage1', 'NaturalImage2', 'Movie1Exp1', 'Movie1Exp2', 'Movie2Exp1', 'Movie2Exp2']


    if args.arch.upper() not in ['LN','GLM', 'NIM']:
        print('{} is not supported now'.format(args.arch))
        exit(0)
    if args.clip:
        args.augment = 10
    

    stimulus = Stimulus[args.stim]
    print('\n\n'+'='*15 + 'settings' + '='*15)
    # print(args)
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    print('='*15 + 'settings' + '='*15 + '\n')
    print(stimulus)

    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # using path below need run the code at upper path
    if args.local:
        DataPath = 'C:\\data\\processed_data'
        TaskPath = os.path.join(DataPath, 'models')
        # TaskPath = '../processed_data/models'
    else:
        DataPath = '/model/retina_image/processed_data'
        TaskPath = '/model/retina_image/processed_data/Output'




    transform=transforms.Compose([
        transforms.ToTensor()
        ])

    if args.clip:
        model = eval('{}_{}_clip()'.format(args.arch, stimulus)).to(device)
    else:
        model = eval('{}_{}()'.format(args.arch, stimulus)).to(device)
        # pdb.set_trace()
        
    print(model)


    dataset1 = RetinaData(root=DataPath, stimulus=full_stimulus[args.stim], train=True,
                       transform=transform, augment=args.augment)
    dataset2 = RetinaData(root=DataPath, stimulus=full_stimulus[args.stim], train=False,
                       transform=transform, augment=args.augment)
    # stim_shape = dataset2.targets.shape
    
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=args.test_batch_size, num_workers=0)


    # pdb.set_trace()
    if args.loss == 'mse':
        loss_p = torch.nn.MSELoss(reduction='sum')
    elif args.loss == 'poisson':
        loss_p = torch.nn.PoissonNLLLoss(log_input=False)
    else:
        print('the loss is not supported now')
        exit(0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=5, gamma=args.gamma)
    start_time = time.time()
    loss1 = 1e10
    loss2 = 1e10

    # store loss and acc
    acc_l1 = 0
    acc_l2 = 0
    acc_l3 = 0
    acc_a1 = 0
    acc_a2 = 0
    acc_a3 = 0
    for epoch in range(1, args.epochs + 1):
        time1 = time.time()
        train(args, loss_p, model, device, train_loader, optimizer, epoch, k=args.k)
        acc_1, acc_2, acc_3, loss_v = test(args, loss_p, model, device, test_loader, k=args.k)
        # if loss_v < loss:
        if loss_v < loss1:
            loss1 = loss_v
            acc_l1 = acc_1
            acc_l2 = acc_2
            acc_l3 = acc_3
            best_state1 = deepcopy(model.state_dict())

        if acc_2 > acc_a2:
            loss2 = loss_v
            acc_a1 = acc_1
            acc_a2 = acc_2
            acc_a3 = acc_3
            best_state2 = deepcopy(model.state_dict()) 

        scheduler.step()
        time_t = time.time() - time1
        print('time used {:.1f}min in epoch {}'.format(time_t//60, epoch))

    time_used = time.time() - start_time
    print('time used {:.1f}min'.format(time_used//60))

    if args.save:
        # print('saving model(acc={:.3f}, loss={:.1f})...'.format(acc, loss))
        if args.clip:
            base_name = '{}_{}_clip'.format(args.arch, stimulus)
        else:
            base_name = '{}_{}'.format(args.arch, stimulus)

        rows = int(len(dataset2) / args.augment) # plot all the test data but only one train data
        columns = args.augment
        if acc_l1 == acc_a1 and loss1 == loss2:
            uname = UniqueName(os.path.join(TaskPath, 'model', base_name), '.pt')
            torch.save(best_state1, uname)
            print('best acc/loss model', uname,'\n','acc1:{:.4f}\nacc2:{:.4f}\nacc3:{:.4f}\nloss:{:.4f}'.format(acc_l1, acc_l2, acc_l3, loss1))
            model.load_state_dict(best_state1)
            plot_response(TaskPath=TaskPath, rows=rows, columns=columns, dataloader=train_loader, model=model, args=args,device=device, train=True)
            plot_response(TaskPath=TaskPath, rows=rows, columns=columns, dataloader=test_loader, model=model, args=args,device=device, train=False)
        else:
            uname1 = UniqueName(os.path.join(TaskPath, 'model', base_name), '.pt')
            torch.save(best_state1, uname1)
            uname2 = UniqueName(os.path.join(TaskPath, 'model', base_name), '.pt')
            torch.save(best_state2, uname2)
            print('best loss model', uname1,'\n','acc1:{:.4f}\nacc2:{:.4f}\nacc3:{:.4f}\nloss:{:.4f}'.format(acc_l1, acc_l2, acc_l3, loss1))
            print('best acc model', uname2,'\n','acc1:{:.4f}\nacc2:{:.4f}\nacc3:{:.4f}\nloss:{:.4f}'.format(acc_a1, acc_a2, acc_a3, loss2))
            model.load_state_dict(best_state1)
            plot_response(TaskPath=TaskPath, rows=rows, columns=columns, dataloader=train_loader, model=model, args=args,device=device, train=True)
            plot_response(TaskPath=TaskPath, rows=rows, columns=columns, dataloader=test_loader, model=model, args=args,device=device, train=False)
            model.load_state_dict(best_state2)
            plot_response(TaskPath=TaskPath, rows=rows, columns=columns, dataloader=train_loader, model=model, args=args,device=device, train=True)
            plot_response(TaskPath=TaskPath, rows=rows, columns=columns, dataloader=test_loader, model=model, args=args,device=device, train=False)

    print('Experiment finished!\n\n')

if __name__ == '__main__':
    main()



                        
