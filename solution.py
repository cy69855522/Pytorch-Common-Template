from my_utility import ImageFolderSplitter, DatasetFromFilename, AdamW
import torch, torch.nn.functional as F
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader
import time
import pretrainedmodels
from tensorboardX import SummaryWriter
import adabound
from PIL import ImageFile
from multiprocessing import cpu_count

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set random number seed
SEED = 66
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
num_cpu = cpu_count()
# np.random.seed(SEED)


class Solution:
    def __init__(self, model, criterion, optimizer, image_path, scheduler=None, epochs=100, batch_size=64,
                 show_interval=20, valid_interval=100, record_loss=True, image_size=224, thread_size=8,
                 seed=66, ten_crops=False, test_ratio=0.25):
        data_transforms = {
            'train': transforms.Compose([
                # transforms.RandomResizedCrop(image_size),
                # transforms.Resize(224),
                transforms.RandomResizedCrop(int(image_size * 1.2)),
                # transforms.ToPILImage(),
                transforms.RandomAffine(15),
                # transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.RandomGrayscale(),
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            "valid": transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        if ten_crops:
            data_transforms['train'] = transforms.Compose([
                # transforms.RandomResizedCrop(image_size),
                # transforms.Resize(224),
                transforms.RandomResizedCrop(int(image_size * 1.2)),
                # transforms.ToPILImage(),
                transforms.RandomAffine(15),
                # transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.RandomGrayscale(),
                transforms.TenCrop(image_size),
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda(lambda crops: torch.stack(
                    [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop) for crop in crops])),

                # transforms.FiveCop(image_size),
                # Lambda(lambda crops: torch.stack([transfoms.ToTensor()(crop) for crop in crops])),
                # transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # data set
        splitter = ImageFolderSplitter(image_path, 1 - test_ratio, seed=seed)
        x_train, y_train = splitter.getTrainingDataset()
        train_dataset = DatasetFromFilename(x_train, y_train, transforms=data_transforms['train'])
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=thread_size)
        x_valid, y_valid = splitter.getValidationDataset()
        valid_dataset = DatasetFromFilename(x_valid, y_valid, transforms=data_transforms['valid'])
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=thread_size)

        if torch.cuda.is_available():
            self.input_type = torch.cuda.FloatTensor
            self.output_type = torch.cuda.LongTensor
            print('Using CUDA o(*≧▽≦)ツ')
        else:
            self.input_type = torch.FloatTensor
            self.output_type = torch.LongTensor

        self.dataloaders = {'train': train_dataloader, 'valid': valid_dataloader}
        self.dataset_size = {'train': len(train_dataset), 'valid': len(valid_dataset)}
        self.dataloader_size = {x: len(self.dataloaders[x]) for x in ['train', 'valid']}
        self.class_names = splitter.num2class
        self.writer = SummaryWriter()
        self.model = model.type(self.input_type)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.show_interval = show_interval
        self.valid_interval = valid_interval
        self.train_loss = 0.0
        self.train_corrects = 0
        self.valid_loss = 0.0
        self.valid_corrects = 0
        self.best_acc = 0.0
        self.train_count = 0
        self.valid_count = 0
        self.record_loss = record_loss
        self.batch_size = batch_size
        self.ncrops = 10
        self.ten_crops = ten_crops

        # def imshow(inp, title=None):
        #     """Imshow for Tensor."""
        #     inp = inp.numpy().transpose((1, 2, 0))
        #     mean = np.array([0.485, 0.456, 0.406])
        #     std = np.array([0.229, 0.224, 0.225])
        #     inp = std * inp + mean
        #     inp = np.clip(inp, 0, 1)
        #     plt.imshow(inp)
        #     if title is not None:
        #         plt.title(title)
        #     plt.pause(5)  # pause a bit so that plots are updated
        #
        #
        # # Get a batch of training data
        # inputs, classes = next(iter(training_dataloader))
        #
        # # Make a grid from batch
        # out = torchvision.utils.make_grid(inputs)
        #
        # imshow(out, title=[class_names[x.item()] for x in classes])

    def train(self):
        since = time.time()

        for epoch in range(self.epochs):
            print('\nEpoch {}/{} TotalSteps {} '.format(
                epoch, self.epochs - 1, self.dataloader_size['train']) + '-' * 40)
            self.model.train()
            self.train_count = 0
            self.train_corrects = 0
            self.train_loss = 0

            for inputs, labels in self.dataloaders['train']:
                if self.ten_crops:
                    bs, ncrops, c, h, w = inputs.size()
                    inputs = inputs.view(-1, c, h, w)
                inputs = inputs.type(self.input_type)
                labels = labels.type(self.output_type)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward
                self.forward(True, inputs, labels, epoch, since)

                # valid
                if self.train_count % self.valid_interval == self.valid_interval - 1 \
                        or self.train_count == self.dataloader_size['train'] - 1:
                    self.model.eval()
                    self.valid_count = 0
                    self.valid_corrects = 0
                    self.valid_loss = 0

                    for inputs, labels in self.dataloaders['valid']:
                        inputs = inputs.type(self.input_type)
                        labels = labels.type(self.output_type)
                        self.forward(False, inputs, labels, epoch, since)

                        self.valid_count += 1

                self.train_count += 1

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60) + ' Best val Acc: {:4f}'.format(self.best_acc))

    def forward(self, training, inputs, labels, epoch, since):
        with torch.set_grad_enabled(training):
            outputs = self.model(inputs)
            if training and self.ten_crops:outputs = outputs.view(self.batch_size, self.ncrops, -1).mean(1)
            _, preds = torch.max(outputs, 1)
            loss = self.criterion(outputs, labels)

            # backward + optimize only if in training mode
            if training:
                loss.backward()
                self.optimizer.step()

        running_loss = loss.item()
        running_corrects = torch.sum(preds == labels.data).double()

        if training:
            self.train_loss += running_loss
            self.train_corrects += running_corrects

            if self.train_count % self.show_interval == self.show_interval - 1:
                if self.record_loss:
                    self.writer.add_scalars('Train_val_loss_' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(since)),
                                            {'train_loss': running_loss},
                                            epoch * self.dataloader_size['train'] + self.train_count)
                print('      {}/{} Count: {} Loss: {:.4f} Acc: {:.4f}'.format(
                    epoch, self.epochs - 1, self.train_count, running_loss, running_corrects/len(preds)))

            if self.train_count == self.dataloader_size['train'] - 1:
                print('Training   TotalLoss: {:.4f} TotalAcc: {:.4f}'.format(
                    self.train_loss/self.dataloader_size['train'], self.train_corrects/self.dataset_size['train']))

        else:
            self.valid_loss += running_loss
            self.valid_corrects += running_corrects

            if self.valid_count == self.dataloader_size['valid'] - 1:
                valid_acc = self.valid_corrects/self.dataset_size['valid']
                if self.record_loss:
                    self.writer.add_scalars('Train_val_loss_' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(since)),
                                            {'valid_loss': self.valid_loss/self.dataloader_size['valid']},
                                            epoch * self.dataloader_size['train'] + self.train_count)

                # save best wights
                if valid_acc > self.best_acc:
                    self.best_acc = valid_acc
                    print('Saving best weights with validation accuracy {}'.format(self.best_acc))
                    torch.save(self.model.state_dict(), 'best.pth')

                print('Validation TotalLoss: {:.4f} TotalAcc: {:.4f}'.format(
                    self.valid_loss/self.dataloader_size['valid'], valid_acc))

            # update lr
            if self.scheduler is not None:
                self.scheduler.step(running_loss)

    def draw_model(self, comment):
        with self.writer:
            for inputs, labels in self.dataloaders['valid']:
                inputs = inputs.type(self.input_type)
                self.writer.add_graph(self.model, inputs)
                break
            print('Model structure drawing is completed')


# model = Net(weight='resnet152_101.pth')
model = pretrainedmodels.__dict__['senet154'](num_classes=1000, pretrained='imagenet')
model.last_linear = nn.Linear(model.last_linear.in_features, 13051)
# self.model.load_state_dict(torch.load('best.pth'))
criterion = nn.CrossEntropyLoss()
optimizer = adabound.AdaBound(model.parameters(), lr=1e-2, final_lr=0.1, weight_decay=1e-7, amsbound=True)
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
S = Solution(model,
             criterion,
             optimizer,
             './cars_data',
             scheduler=None,
             epochs=100,
             batch_size=8,
             thread_size=num_cpu,
             show_interval=20,
             valid_interval=2000,
             seed=SEED,
             )
S.train()
# S.draw_model()
