from args2 import Setup
from lib.agents.inc_agent import InceptionAgent
from lib.dataloader.datasets import AffectNet


def count_classes(trainset):
    cls = [0, 0, 0, 0, 0, 0, 0]
    for idx, sample in enumerate(trainset):
        cls[sample['label']] += 1
    return cls


if __name__ == "__main__":
    args = Setup().parse()
    args.trainsplit = "train_ids_0.csv"
    args.testsplit = "test_ids_0.csv"
    args.validsplit = "valid_ids_0.csv"

    args.load_size = 320
    args.final_size = 299
    args.n_classes = 7
    args.remove_class = 0
    args.ckp_label_type = 1

    args.affectnet_img_parentfolder_man = "D:/Downloads/Manually_Annotated_compressed/"
    args.affectnet_img_parentfolder_aut = "D:/Downloads/Automatically_Annotated_compressed/"


    # train = AffectNet(args=args, train=True, valid=False, ckp_label_type=True, remove_class=0)
    # test = AffectNet(args=args, train=False, valid=False, ckp_label_type=True, remove_class=0)
    val = AffectNet(args=args, train=False, valid=True, ckp_label_type=True, remove_class=0)

    count = count_classes(val)
    print(count)



