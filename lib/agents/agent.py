import time
import torch
import pickle
import json
import os


class Agent:
    def __init__(self, args):
        self.args = args
        self.timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.is_cuda = torch.cuda.is_available()
        self.tmp_epoch = 0  # saving writes tmp_epoch to state dict

        self.list_train_loss = []
        self.list_test_loss = []
        self.list_train_acc = []
        self.list_test_acc = []
        self.list_lr = []

    def save_args(self, path):
        with open(path, 'w') as f:
            json.dump(self.args.__dict__, f, indent=2)

    def save_resultlists_as_dict(self, path):
        dict = {
            'train_loss': self.list_train_loss,
            'train_acc': self.list_train_acc,
            'test_loss': self.list_test_loss,
            'test_acc': self.list_test_acc,
            'lr': self.list_lr,
        }
        file = open(path, "wb")
        pickle.dump(dict, file)

    def load_resultlists_from_pickle(self, path):
        file_to_read = open(path)
        loaded_dict = pickle.load(file_to_read)
        return loaded_dict

    def load_ckpt(self, file_name):
        raise NotImplementedError

    def save_ckpt(self, file_name="ckpt.pth.tar"):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def train_epoch(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def __create_folders__(self, name):
        print("Creating folders...")
        if not os.path.isdir(self.args.result_folder):
            os.makedirs(self.args.result_folder)

        self.run_path = self.args.result_folder + "run_" + name + "_{0}".format(self.timestamp) + "/"

        os.makedirs(self.run_path)

        self.train_path = self.run_path + "train_" + name + "_" + self.timestamp + "/"
        self.test_path = self.run_path + "test_" + name + "_" + self.timestamp + "/"
        self.train_ckpt = self.train_path + "ckpt/"
        self.train_plots = self.train_path + "plots/"

        self.test_plots = self.test_path + "plots/"

        os.makedirs(self.train_path)
        os.makedirs(self.train_ckpt)
        os.makedirs(self.train_plots)

        os.makedirs(self.test_path)
        os.makedirs(self.test_plots)



