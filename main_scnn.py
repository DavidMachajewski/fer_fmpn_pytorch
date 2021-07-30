from lib.models.models import SCNN0
from args2 import Setup
from torchsummary import summary
from lib.agents.scnn_agent import SCNNAgent
from lib.agents.runner import Runner

if __name__ == "__main__":
    args = Setup().parse()
    args.mode = "train"
    args.model_to_train = "scnn"
    args.scnn_nr = 0
    args.epochs = 20
    args.gpu_id = 0
    args.dataset = "rafdb"
    args.lr_gen = 0.001
    args.load_size = 100
    args.final_size = 100  # rafdb Net input size
    args.n_classes = 7  # if you use n_classes 6 then delete class nr 7
    # args.remove_class = 7
    args.trainsplit = "train_ids_0.csv"
    args.testsplit = "test_ids_0.csv"
    args.validsplit = "valid_ids_0.csv"

    # scnn = SCNN0(args).cuda()
    scnn_agent = SCNNAgent(args)
    summary(scnn_agent.model, (3, 100, 100))
    print(scnn_agent.model.fc2.weight.grad)
    # summary(scnn, (3, 100, 100))

    runner = Runner(args)
    runner.start()
