from lib.dataloader.datasets import get_ckp
from args2 import Setup
import cv2 as cv
import numpy as np

if __name__ == '__main__':
    args = Setup().parse()

    args.model_to_train = "fmpn"
    args.mode = "test"
    args.gpu_id = 0
    args.trainsplit = "train_ids_2.csv"
    args.testsplit = "test_ids_2.csv"
    args.validsplit = "valid_ids_2.csv"

    ckp_train_dl, ckp_test_dl = get_ckp(args, batch_size=8)

    for batch_id, batch in enumerate(ckp_test_dl):
        for idx in range(0, len(batch)):
            mask = batch["mask"][idx].detach().permute(1, 2, 0).numpy()
            mask_rgb = cv.cvtColor(mask, cv.COLOR_GRAY2RGB).astype(np.uint8)
            cv.imshow("", mask)
            cv.waitKey()