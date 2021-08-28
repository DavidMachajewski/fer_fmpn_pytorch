from lib.models.models import SCNN0
from args2 import Setup
from torchsummary import summary
from lib.agents.scnn_agent import SCNNAgent
from lib.agents.runner import Runner


def train_rafdb_using_scnn():
    pass


if __name__ == "__main__":
    args = Setup().parse()
    args.mode = "train"
    args.model_to_train = "scnn"
    args.scnn_nr = 0
    args.scnn_config = 'C'
    args.epochs = 200
    args.gpu_id = 0
    args.dataset = "rafdb"
    args.augmentation = 1
    args.batch_size = 64
    args.lr_gen = 0.0005
    args.load_size = 120
    args.final_size = 100   # rafdb Net input size
    args.n_classes = 6  # if you use n_classes 6 then delete class nr 7!!!! otherwise 7
    args.remove_class = 7
    args.scnn_llfeatures = 2048  # 512 # 1024  # 2048 # 4096  # reduce it later if overfitting occures
    args.trainsplit = "train_ids_0.csv"
    args.testsplit = "test_ids_0.csv"
    args.validsplit = "valid_ids_0.csv"

    # scnn = SCNN0(args).cuda()
    scnn_agent = SCNNAgent(args)
    print(summary(scnn_agent.model, (3, 100, 100)))
    # print(scnn_agent.model.fc2.weight.grad)
    # summary(scnn, (3, 100, 100))

    runner = Runner(args)
    runner.start()
