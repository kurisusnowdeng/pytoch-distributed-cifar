'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import time
import argparse

from models.vgg import *
from models.dpn import *
from models.lenet import *
from models.senet import *
from models.pnasnet import *
from models.densenet import *
from models.googlenet import *
from models.shufflenet import *
from models.shufflenetv2 import *
from models.resnet import *
from models.resnext import *
from models.preact_resnet import *
from models.mobilenet import *
from models.mobilenetv2 import *

# from utils import progress_bar

import horovod.torch as hvd


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--data_dir', type=str)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--model', type=str)
parser.add_argument('--model_dir', type=str)
parser.add_argument('--lr_decay_step_size', type=int, default=0)
parser.add_argument('--lr_decay_factor', type=float, default=1.0)
parser.add_argument('--save_frequency', type=int, default=100)
parser.add_argument('--num_epochs', type=int)
parser.add_argument('--device_ids', nargs='+', type=int)
args = parser.parse_args()

# Initialize Horovod
hvd.init()

# Pin GPU to be used to process local rank (one GPU per process)
torch.cuda.set_device(hvd.local_rank())


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
# Partition dataset among workers using DistributedSampler
train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=hvd.size(), rank=hvd.rank())

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, sampler=train_sampler)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
test_sampler = torch.utils.data.distributed.DistributedSampler(testset, num_replicas=hvd.size(), rank=hvd.rank())
testloader = torch.utils.data.DataLoader(testset, batch_size=100, sampler=test_sampler)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model: ' + args.model)
if args.model == 'vgg16':
    net = VGG('VGG16')
elif args.model == 'vgg19':
    net = VGG('VGG19')
elif args.model == 'resnet18':
    net = ResNet18()
elif args.model == 'resnet34':
    net = ResNet34()
elif args.model == 'resnet50':
    net = ResNet50()
elif args.model == 'resnet101':
    net = ResNet101()
elif args.model == 'resnet152':
    net = ResNet152()
# net = PreActResNet18()
elif args.model == 'googlenet':
    net = GoogLeNet()
# net = DenseNet121()
elif args.model == 'resnext':
    net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)

# net = net.to(device)
# if device == 'cuda':
#     if args.device_ids is not None:
#         net = torch.nn.DataParallel(net, device_ids=args.device_ids)
#     else:
#         net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True
net.cuda()
cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(model_dir), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(model_dir + '/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Broadcast parameters from rank 0 to all other processes.
hvd.broadcast_parameters(net.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

# Add Horovod Distributed Optimizer
optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=net.named_parameters())

if args.lr_decay_step_size > 0:
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step_size, gamma=args.lr_decay_factor)

# Training
def train(epoch):
    print('Number of workers: %d' % hvd.size())
    global_batch_size = hvd.size() * args.batch_size
    lr = scheduler.get_lr()[0] if args.lr_decay_step_size > 0 else args.lr
    print('\nEpoch: %d\nlr = %g' % (epoch, lr))
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    time_used = 0.0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        start = time.time()
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        end = time.time()

        time_used += end - start
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        print('[%d/%d] Loss: %.3f | Acc: %.3f%% (%d/%d) | Throughput: %.3f (images/sec)' 
            % (batch_idx + 1, len(trainloader), train_loss/(batch_idx+1),
            100.*correct/total, correct, total, global_batch_size/(end-start+1e-6)))
    
    print('\n[Epoch %d] Loss: %.3f | Acc: %.3f%% (%d/%d) | Throughput: %.3f (images/sec)' 
        % (epoch, train_loss/(len(trainloader)),
        100.*correct/total, correct, total, 
        global_batch_size*len(trainloader)/(time_used+1e-6)))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #     % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        print('\n[Evaluation] Loss: %.3f | Acc: %.3f%% (%d/%d)' 
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if (epoch == start_epoch + args.num_epochs - 1) or \
       (args.save_frequency > 0 and (epoch + 1) % args.save_frequency == 0):
    # if acc > best_acc:
        print('\nSaving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(args.model_dir):
            os.makedirs(args.model_dir)
        torch.save(state, args.model_dir + '/' + str(epoch) + '-ckpt.t7')
    if acc > best_acc:
        best_acc = acc


for epoch in range(start_epoch, start_epoch + args.num_epochs):
    train(epoch)
    test(epoch)
    if args.lr_decay_step_size > 0:
        scheduler.step()
