from lib.agents.agent import Agent


class ResNetAgent(Agent):
    def __init__(self, args):
        super(ResNetAgent, self).__init__(args)
        self.name = "resnet18"
        self.args = args

        # :TODO: make resnet for transfer learning
        self.model = resnet18(pretrained=self.args.pretrained)

    def __init_optimizer__(self):
        pass

    def __set_device__(self):
        pass

    def load_ckpt(self, file_name):
        pass

    def save_ckpt(self):
        pass

    def __create_folders__(self):
        pass

    def save_resultlists_as_dict(self, path):
        pass

    def train(self):
        pass

    def train_epoch(self):
        pass

    def eval_epoch(self):
        pass

    def test(self):
        pass

    def run(self):
        pass

    def __calc_accuracy__(self, predictions, labels):
        pass

    def __adjust_lr__(self):
        pass