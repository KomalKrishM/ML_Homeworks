import argparse
import torch.nn as nn
import torch


def get_args():
    parser = argparse.ArgumentParser(description='Args for training networks')
    parser.add_argument('-seed', type=int, default=1, help='random seed')
    parser.add_argument('-num_epochs', type=int, default=20, help='num epochs')
    parser.add_argument('-batch', type=int, default=16, help='batch size')
    parser.add_argument('-lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('-drop', type=float, default=0.3, help='drop rate')
    args, _ = parser.parse_known_args()
    return args


class LeNet(nn.Module):
    def __init__(self, args):
        super(LeNet, self).__init__()
        ### YOUR CODE HERE

        self.layer1 = nn.Sequential(nn.Conv2d(3,6, 5, 1, 0),
                      nn.ReLU(),
                      nn.MaxPool2d(kernel_size=2, stride=2),
                      nn.BatchNorm2d(6),
                      nn.Conv2d(6, 16, 5, 1,0),
                      nn.ReLU(),
                      nn.MaxPool2d(2, 2),
                      nn.BatchNorm2d(16)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Dropout2d(p=0.3),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.Linear(84, 10)
        )

        ### END YOUR CODE

    def forward(self, x):
        '''
        Input x: a batch of images (batch size x 3 x 32 x 32)
        Return the predictions of each image (batch size x 10)
        '''
        ### YOUR CODE HERE

        result = self.layer1(x)
        result = torch.flatten(result, 1)
        logits = self.layer2(result)
        return logits


        ### END YOUR CODE
