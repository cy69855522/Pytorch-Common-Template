# Personal python utility functions

# 1.images splitter
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torchvision.transforms import ToTensor, Resize, Compose
from PIL import Image
from sklearn.model_selection import train_test_split
import math


class ImageFolderSplitter:
    # images should be placed in folders like:
    # --root
    # ----root\dogs
    # ----root\dogs\image1.png
    # ----root\dogs\image2.png
    # ----root\cats
    # ----root\cats\image1.png
    # ----root\cats\image2.png
    # path: the root of the image folder
    def __init__(self, path, train_size=0.8, seed=66, drop_type=('txt', 'csv')):
        self.path = path
        self.train_size = train_size
        self.class2num = {}
        self.num2class = {}
        self.class_nums = {}
        self.data_x_path = []
        self.data_y_label = []
        self.x_train = []
        self.x_valid = []
        self.y_train = []
        self.y_valid = []
        self.total_number = 0
        for root, dirs, files in os.walk(path):
            if len(files) == 0 and len(dirs) > 1 and root == path:
                for i, dir1 in enumerate(dirs):
                    self.num2class[i] = dir1
                    self.class2num[dir1] = i
            elif len(files) > 1 and len(dirs) == 0:
                category = root.split('/')[-1]
                valid_paths = [file1 for file1 in files if not any(file1[-(1 + len(suffix)):] == '.' + suffix
                                                                   for suffix in drop_type)]
                if len(valid_paths) < 2:
                    print('!', root, dirs, files)
                    self.num2class.pop(self.class2num[category])
                    self.class2num.pop(category)
                    continue

                label = self.class2num[category]
                self.total_number += len(valid_paths)
                self.class_nums[label] = len(valid_paths)
                for file1 in valid_paths:
                    self.data_x_path.append(os.path.join(root, file1))
                    self.data_y_label.append(label)
            else:
                print('!', root, dirs, files)
                category = root.split('/')[-1]
                self.num2class.pop(self.class2num[category])
                self.class2num.pop(category)
                # raise RuntimeError("please check the folder structure!")
        print('Find %d classes %d images in total' % (len(self.num2class), self.total_number))
        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(self.data_x_path, self.data_y_label,
                                                                                  shuffle=True,
                                                                                  train_size=self.train_size,
                                                                                  test_size=1-self.train_size,
                                                                                  random_state=seed,
                                                                                  stratify=self.data_y_label)

    def getTrainingDataset(self):
        return self.x_train, self.y_train

    def getValidationDataset(self):
        return self.x_valid, self.y_valid


class DatasetFromFilename(Dataset):
    # x: a list of image file full path
    # y: a list of image categories
    def __init__(self, x, y, transforms=None):
        super(DatasetFromFilename, self).__init__()
        self.x = x
        self.y = y
        if transforms == None:
            self.transforms = ToTensor()
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = Image.open(self.x[idx])
        img = img.convert("RGB")
        return self.transforms(img), torch.tensor(self.y[idx])


# test code
# splitter = ImageFolderSplitter("for_test")
# transforms = Compose([Resize((51, 51)), ToTensor()])
# x_train, y_train = splitter.getTrainingDataset()
# training_dataset = DatasetFromFilename(x_train, y_train, transforms=transforms)
# training_dataloader = DataLoader(training_dataset, batch_size=2, shuffle=True)
# x_valid, y_valid = splitter.getValidationDataset()
# validation_dataset = DatasetFromFilename(x_valid, y_valid, transforms=transforms)
# validation_dataloader = DataLoader(validation_dataset, batch_size=2, shuffle=True)
# for x, y in training_dataloader:
#     print(x.shape, y.shape)

class AdamW(Optimizer):
    """Implements Adam algorithm.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            ICLR 2018 https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-4, amsgrad=False):
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        #super(AdamW, self).__init__(params, defaults)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                if group['weight_decay'] != 0:
                    decayed_weights = torch.mul(p.data, group['weight_decay'])
                    p.data.addcdiv_(-step_size, exp_avg, denom)
                    p.data.sub_(decayed_weights)
                else:
                    p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss
