from args2 import Setup
from lib.agents.inc_agent import InceptionAgent

if __name__ == "__main__":
    args = Setup().parse()
    args.trainsplit = "train_ids_0.csv"
    args.testsplit = "test_ids_0.csv"
    args.validsplit = "valid_ids_0.csv"

