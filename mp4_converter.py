
import skvideo
# skvideo.setFFmpegPath('/Users/daniel/opt/anaconda3/bin/')
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from torch.utils.data import Dataset, DataLoader
import skvideo.io
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from torch.nn.modules.activation import LeakyReLU
from tqdm import tqdm


def load_mp4(file_csv_path, batch_size):
    df = pd.read_csv(file_csv_path)
    labels = []
    inputs = []
    for index, row in df.iterrows():
        labels.append(row['Icon_name'])
        print(row['mp4_link'])
        videodata = skvideo.io.vread(row['mp4_link'], num_frames = 50)
        print(videodata.shape)
        inputs.append(videodata)

#normalize inputs
class VideoDataset(Dataset):
    def __init__(self, file_csv_path, transform = None):
        self.transform = transform
        self.file_names = pd.read_csv(file_csv_path)
        self.file_names = self.file_names.iloc[0:15]
    
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, index):
        video_data = skvideo.io.vread(self.file_names.iloc[index, 2], num_frames = 50)
        print(self.file_names.iloc[index, 0])
        video_data = video_data/255.0
        video_data = np.moveaxis(video_data, -1, 0)
        if (self.transform):
            video_data = self.transform(video_data)
        return video_data

    # convert back into mp4 file
    def save_as_mp4(self, data):
        pass

# redo layers of encoder for input of size (3 x 50 x 512 x 512)
class Encoder(nn.Module):
    def __init__(self, nz):
        super().__init__()
        self.nz = nz
        self.net = nn.Sequential(
            # 10 x 23 x 254 x 254
            nn.Conv3d(3, 10, (5,5,5), stride = (2,2, 2),  bias = True),
            nn.LeakyReLU(),
            # 20 x 10 x 125 x 126
            nn.Conv3d(10, 20, (5,5,3), stride = (2 , 2, 2),  bias = True),
            nn.LeakyReLU(),
            # 30 x 2 x 25 x 42
            nn.BatchNorm3d(20),
            nn.Conv3d(20, 30, (5,5,3), stride = (5 , 5, 3),  bias = True),
            nn.LeakyReLU(),
            nn.BatchNorm3d(30),
            # 40 x 2 x 5 x 8
            nn.Conv3d(30, 40, (1,5,5), stride = (1 , 5, 5),  bias = True),
            nn.LeakyReLU(),
            nn.BatchNorm3d(40),
            # 3200
            nn.Flatten(),
            nn.Linear(3200, self.nz)
        )
    
    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, nz):
        super().__init__()
        self.nz = nz
        self.map = nn.Linear(nz, 3200)
        self.net = nn.Sequential(
            nn.BatchNorm3d(40),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(40, 30, (1, 5, 14), stride = (1, 5, 4), bias = True),
            nn.BatchNorm3d(30),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(30, 20 , (5, 5, 3), stride = (5, 5, 3), bias = True),
            nn.BatchNorm3d(20),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(20, 10 , (5, 6, 4), stride = (2, 2, 2), bias = True),
            nn.ConvTranspose3d(10, 3 , (6, 6, 6 ), stride = (2, 2, 2), bias = True),
            nn.Sigmoid()
        )
    def forward(self, x):
        output = self.map(x).reshape(-1, 40 , 2, 5, 8)
        return(self.net(output))

class AutoEncoder(nn.Module):
    def __init__(self, nz):
        super().__init__()
        self.encoder = Encoder(nz)
        self.decoder = Decoder(nz)

    def forward(self, x):
        y = self.encoder(x)
        return self.decoder(y)

    def reconstruct(self, x):
        """Only used later for visualization."""
        return self.forward(x)


# def load_dataset(file_path):
#     video_dataset = VideoDataset(file_path)
#     return video_dataset


def train_autoencoder(epochs = 0, learning_rate = .5):
    video_dataset = VideoDataset("icons.csv")
    video_dataloader = DataLoader(video_dataset, batch_size = 1, shuffle = True, num_workers = 0, drop_last=True)
    auto_encoder = AutoEncoder(15)
    auto_encoder = auto_encoder.train()
    auto_encoder = auto_encoder.float()
    opt = torch.optim.Adam(auto_encoder.parameters(), lr= learning_rate)          # create optimizer instance
    criterion = nn.MSELoss()
    for ep in tqdm(range(epochs)):
        print("Epoch: " + str(ep))
        for input in video_dataloader:
            print(input.shape)
            opt.zero_grad()
            outputs = auto_encoder(input.float())
            print(outputs.shape)
            loss = criterion(outputs.float(), input.float())
            loss.backward()
            opt.step()
    videodata = skvideo.io.vread('mp4/Idea.mp4', num_frames = 50)
    videodata = videodata/255.0
    videodata = np.moveaxis(videodata, -1, 0)
    videodata = videodata[None, :]
    print(videodata.shape)
    auto_encoder.eval()
    skvideo.io.vwrite("test/test.mp4", auto_encoder(videodata))
    # torch.save(auto_encoder.state_dict(), "Icon_WebScraping/model")
    # input = next(iter(video_dataloader))
    # print(input)
    # video_dataset.save_as_mp4(auto_encoder(input.float()))


if __name__ == "__main__":
    video_dataset = VideoDataset("icons.csv")
    video_dataloader = DataLoader(video_dataset, batch_size = 2, shuffle = True, num_workers = 0)
    encoder = Encoder(30)
    decoder = Decoder(30)
    train_autoencoder(epochs = 50, learning_rate = .5)

        
    
    






    
    



