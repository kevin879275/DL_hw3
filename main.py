import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import random
import model

Epoch = 40
BATCH_SIZE = 32
lr = 0.01
Train_root = "./Data/Data_train"
Test_root = "./Data/Data_test"

classes = {"Carambula": 0, "Lychee": 1, "Pear": 2}


class DataLoader():
    def __init__(self, split=0.7, root='', Is_Train=True, shuffle=True):
        if Is_Train:
            imgs_label = []
            if(os.path.isdir(root) == False):
                print("Directory not found !!")
                return
            dirs = os.listdir(root)
            for dir in dirs:
                path = root + '/' + dir
                for file in os.listdir(path):
                    img_path = path + '/' + file
                    im = Image.open(img_path).convert("RGB")
                    #img = Image.fromarray(np.array(im))
                    # img.show()
                    imgs_label.append(
                        [np.transpose(np.array(im) / 255, (2, 0, 1)), classes[dir]])
            self.shuffle = shuffle
            if self.shuffle == True:
                random.shuffle(imgs_label)

            self.train_data = imgs_label[:int(split*len(imgs_label))]
            self.valid_data = imgs_label[int(split*len(imgs_label)):]

        if Is_Train == False:
            imgs_label = []
            if(os.path.isdir(root) == False):
                print("Directory not found !!")
                return
            dirs = os.listdir(root)
            for dir in dirs:
                path = root + '/' + dir
                for file in os.listdir(path):
                    img_path = path + '/' + file
                    im = Image.open(img_path).convert("RGB")
                    imgs_label.append(
                        [np.transpose(np.array(im) / 255, (2, 0, 1)), classes[dir]])
            self.test_data = imgs_label

    def get_train_batch(self, batch_size=BATCH_SIZE):
        if self.shuffle:
            random.shuffle(self.train_data)
        for start in range(0, len(self.train_data), batch_size):
            yield np.array([data[0] for data in self.train_data[start:start + batch_size]]), np.array([data[1] for data in self.train_data[start:start + batch_size]])

    def get_valid_batch(self, batch_size=BATCH_SIZE):
        for start in range(0, len(self.valid_data), batch_size):
            yield np.array([data[0] for data in self.valid_data[start:start + batch_size]]), np.array([data[1] for data in self.valid_data[start:start + batch_size]])

    def get_test_batch(self, batch_size=BATCH_SIZE):
        for start in range(0, len(self.test_data), batch_size):
            yield np.array([data[0] for data in self.test_data[start:start + batch_size]]), np.array([data[1] for data in self.test_data[start:start + batch_size]])


def Softmax(logits):
    exps = np.exp(logits)
    sum_of_exps = np.sum(exps, axis=1)
    softmax = [exps[i] / sum_of_exps[i] for i in range(sum_of_exps.shape[0])]
    return np.asarray(softmax)


def Cross_Entropy(y, y_predict):
    # -(sigma(target * log(predict))) / size
    # y_predict.shape = (batch_size , num_of_classes)
    # y.shape = (batch_size, )
    reference = np.zeros_like(y_predict)
    reference[np.arange(y_predict.shape[0]), y] = 1
    mul = np.multiply(reference, np.log(y_predict))
    Sum = np.sum(mul)
    loss = - (1 / y.shape[0]) * Sum
    return loss


def Grad_Cross_Entropy(y, y_predict):
    reference = np.zeros_like(y_predict)
    reference[np.arange(y.shape[0]), y] = 1
    softmax = np.exp(y_predict)/np.sum(np.exp(y_predict),
                                       axis=-1, keepdims=True)
    return (-reference + softmax) / y_predict.shape[0]


def main():
    Train_dataLoader = DataLoader(split=0.7, root=Train_root, Is_Train=True)
    Test_dataLoader = DataLoader(root=Test_root, Is_Train=False)
    Model = model.Model(lr=lr)
    print("training start\n")
    train_loss_list = []
    val_loss_list = []
    val_acc_list = []
    list_epochs = [i + 1 for i in range(Epoch)]
    for epoch in range(Epoch):
        print("epoch", epoch+1, "/", Epoch)
        train_batch = 0
        train_loss = 0
        valid_loss = 0
        valid_acc = 0
        for train_imgs, train_label in Train_dataLoader.get_train_batch(batch_size=BATCH_SIZE):
            train_batch = train_batch + 1
            activations = Model.forward(train_imgs)
            logits = activations[-1]
            softmax = Softmax(logits)
            train_loss += Cross_Entropy(train_label, softmax)
            loss_grad = Grad_Cross_Entropy(train_label, logits)
            Model.backward(loss_grad, activations)
        for valid_imgs, valid_label in Train_dataLoader.get_valid_batch(batch_size=len(Train_dataLoader.valid_data)):
            activations = Model.forward(valid_imgs)
            logits = activations[-1]
            softmax = Softmax(logits)
            valid_loss += Cross_Entropy(valid_label, softmax)
            valid_acc += np.mean(logits.argmax(axis=-1) == valid_label)
        train_loss_list.append(train_loss / train_batch)
        val_loss_list.append(valid_loss)
        val_acc_list.append(valid_acc)
        print("Train_loss = ",
              train_loss_list[-1], " , Valid loss = ", val_loss_list[-1], " , valid acc = ", round(val_acc_list[-1], 3)*100, '%')
    plt.title('train and validation')
    plt.subplot(131)
    plt.xlabel('epoch')
    plt.ylabel('train loss value')
    plt.plot(list_epochs, train_loss_list)
    plt.subplot(132)
    plt.xlabel('epoch')
    plt.ylabel('valid loss value')
    plt.plot(list_epochs, val_loss_list)
    plt.subplot(133)
    plt.xlabel('epoch')
    plt.ylabel('valid acc')
    plt.plot(list_epochs, val_acc_list)
    plt.show()
    print("testing start\n")
    test_loss = 0
    test_acc = 0
    batch = 0
    for test_imgs, test_label in Test_dataLoader.get_test_batch(batch_size=1470):
        batch = batch + 1
        activations = Model.forward(test_imgs)
        logits = activations[-1]
        softmax = Softmax(logits)
        test_loss += Cross_Entropy(test_label, softmax)
        test_acc += np.mean(logits.argmax(axis=-1) == test_label)

    print("test loss = ", test_loss,
          " ,  test acc = ", round(test_acc, 3)*100, '%')


if __name__ == "__main__":
    main()
