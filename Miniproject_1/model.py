import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

np.random.seed(5)
random.seed(5)
torch.manual_seed(5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class XYMixerDataset(Dataset):
    """
    Class for randomly mixing the inputs and targets of the dataset. 
    """
    def __init__(self, noisy_imgs_1, noisy_imgs_2):
        assert len(noisy_imgs_1) == len(noisy_imgs_2)
        self.noisy_imgs_1 = noisy_imgs_1
        self.noisy_imgs_2 = noisy_imgs_2

    def __len__(self):
        return len(self.noisy_imgs_1)

    def __getitem__(self, idx):
        random_order = random.randint(0, 1)
        curr_imgs = (self.noisy_imgs_1[idx], self.noisy_imgs_2[idx])
        return curr_imgs[random_order], curr_imgs[1 - random_order]


class Net(nn.Module):
    """
    Class for the network.
    """
    def __init__(self):
        """
        Initializes the network structure.
        """
        super(Net, self).__init__()

        # encoder
        self._block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.MaxPool2d(2))

        self._block2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.MaxPool2d(2))

        # decoder
        self._block3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1))

        self._block4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1))

        self._block5 = nn.Sequential(
            nn.Conv2d(192, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1))

        self._block6 = nn.Sequential(
            nn.Conv2d(131, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        """Through encoder, then decoder by adding U-skip connections. """

        # Encoder
        pool1 = self._block1(x)  # 16
        pool2 = self._block2(pool1)  # 8
        pool3 = self._block2(pool2)  # 4

        # Decoder
        upsample4 = self._block3(pool3)  # 8
        concat4 = torch.cat((upsample4, pool2), dim=1)
        upsample3 = self._block4(concat4)  # 16
        concat3 = torch.cat((upsample3, pool1), dim=1)
        upsample2 = self._block5(concat3)  # 32
        concat1 = torch.cat((upsample2, x), dim=1)

        # Final activation
        return self._block6(concat1)


class Model():
    """
    Class for initializing, training and testing a model.
    """
    def __init__(self) -> None:
        # defining the model
        self.model = Net().to(device)
        # defining the optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=0.001, betas=[0.9, 0.99], eps=1e-8)
        # defining the loss function
        self.criterion = nn.MSELoss().to(device)

    def load_pretrained_model(self) -> None:
        """
        Loads pretrained model from bestmodel.pth.
        """
        model_path = Path(__file__).parent / "bestmodel.pth"
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def train(self, train_input, train_target, num_epochs) -> None:
        """
        Trains model on the provided training input and prints its loss for each epoch.
        """
        train_dataset = XYMixerDataset(
            train_input.float(), train_target.float())
        train_loader = DataLoader(
            train_dataset, batch_size=32, shuffle=True)
            
        for epoch in range(num_epochs):
            self.model.train()
            train_losses = []
            for batch_idx, (x_train, y_train) in enumerate(train_loader):
                x_train = x_train.to(device)
                y_train = y_train.to(device)

                # Denoise image
                source_denoised = self.model(x_train)

                # Calculate loss
                loss_train = self.criterion(source_denoised, y_train)
                train_losses.append(loss_train.item())

                # Zero gradients, perform a backward pass, and update the weights
                self.optimizer.zero_grad()
                loss_train.backward()
                self.optimizer.step()
            print('Epoch : ',epoch+1, '\t', 'train-loss :', np.mean(train_losses))

    def predict(self, test_input) -> torch.Tensor:
        """
        Predicts output on the given test_input.
        """
        self.model.eval()
        input_type = test_input.type()
        denoised = self.model(test_input.float().to(device))
        return torch.clamp(denoised, min=0, max=255).type(input_type) # Make sure that the output is between 0-255
