from lib.agents.inc_agent import InceptionAgent
from lib.agents.densenet_agent import DenseNetAgent
from lib.agents.resnet_agent import ResNetAgent
from lib.agents.fmpn_agent import FmpnAgent
import lib.eval.eval_utils as eval


class Runner:
    def __init__(self, args):
        self.args = args
        self.model = self.init()

    def init(self):
        if self.args.model_to_train == "incv3":
            print("Initializing Inceptionv3 Network...")
            model = InceptionAgent(self.args)
        elif self.args.model_to_train == "densenet":
            print("Initializing DenseNet Network...")
            model = DenseNetAgent(self.args)
        elif self.args.model_to_train == "resnet18":
            print("Initializing ResNet18 Network...")
            model = ResNetAgent(self.args)
        elif self.args.model_to_train == "fmg":
            model = 0
        elif self.args.model_to_train == "fmpn":
            print("Initializing Facial Motion Prior Network")
            model = FmpnAgent(self.args)
        else:
            raise NotImplementedError
        return model

    def start(self):
        model = self.model

        if self.args.mode == "train":
            model.run()
            if self.args.dataset == "fer":
                eval.make_ds_distribution_plot(train_dl=model.train_dl, test_dl=model.test_dl, save_to=model.train_plots, valid_dl=model.valid_dl, n_classes=self.args.n_classes, classnames=["anger", "disgust", "fear", "happy", "sadness", "surprise"])
            elif self.args.dataset == "rafdb":
                if self.args.n_classes == 6:
                    print("rafdb 6 classes")
                    eval.make_ds_distribution_plot(train_dl=model.train_dl, test_dl=model.test_dl, save_to=model.train_plots, valid_dl=model.valid_dl, n_classes=self.args.n_classes, classnames=['surprise', 'fear', 'disgust', 'happiness', 'sadness', 'anger'])
                elif self.args.n_classes == 7:
                    print("rafdb 7 classes")
                    eval.make_ds_distribution_plot(train_dl=model.train_dl, test_dl=model.test_dl,
                                                   save_to=model.train_plots, valid_dl=model.valid_dl,
                                                   n_classes=self.args.n_classes,
                                                   classnames=['surprise', 'fear', 'disgust', 'happiness', 'sadness', 'anger', 'neutral'])
            else:
                eval.make_ds_distribution_plot(train_dl=model.train_dl, test_dl=model.test_dl,
                                               save_to=model.train_plots, valid_dl=model.valid_dl,
                                               n_classes=self.args.n_classes)
            eval.make_loss_plot(path_to_dict=model.train_logs_path, save_to=model.train_plots)
            eval.make_acc_plot(path_to_dict=model.train_logs_path, save_to=model.train_plots)
            if self.args.model_to_train == "fmpn":
                eval.make_lr_plot_fmpn(path_to_dict=model.train_logs_path, save_to=model.train_plots)
            else:
                # eval.make_loss_plot(path_to_dict=model.train_logs_path, save_to=model.train_plots)
                eval.make_lr_plot(path_to_dict=model.train_logs_path, save_to=model.train_plots)
            model.test()
            #
            # model.evaluate().....
            #
        if self.args.mode == "test":
            """simple test loop if you want to load a ckpt and test it.
            Otherwise it is tested directly after training"""
            model.__create_folders__()
            model.save_args(model.run_path + "args.txt")
            model.test()
