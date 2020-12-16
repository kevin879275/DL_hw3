import numpy as np
import matplotlib.pyplot as plt
import os

BATCH_SIZE = 64
lr = 0.01
Train_root = "./Data/Data_train"
Test_root = "./Data/Data_test"


class DataLoader():
    def __init__(self, split=0.7, root='', Is_Train=True):
        if Is_Train:
            imgs = []
            label = []
            if(os.path.isdir(root) == False):
                print("Directory not found !!")
                return
            dirs = os.listdir(root)
            for dir in dirs:
                path = root + '/' + dir
                for file in dir:
                    f = open(file, 'rb')
                    imgs.append(numpy.array(f))
                    label.append(file)

        if Is_Train == False:
            pass


def main():
    DataLoader(split=0.7, root=Train_root, Is_Train=True)


if __name__ == "__main__":
    main()
