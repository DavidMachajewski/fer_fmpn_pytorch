for %i in (2,3,4) do (for %j in (0,1) do python main.py --mode train --gpu_id 0 --model_to_train incv3 --epochs 200 --pretrained %j --dataset ckp --save_ckpt_intv 50 --batch_size 8 --lr_gen 0.001 --trainsplit train_ids_%i.csv --testsplit test_ids_%i.csv )



do python main.py --mode train --gpu_id 0 --model_to_train incv3 --epochs 200 --pretrained %j --dataset ckp --batch_size 8 --lr_gen 0.001 --trainsplit train_ids_%i.csv --testsplit test_ids_%i.csv




for %i in (0,1) do (for %j in (0,1) do python main.py --mode train --gpu_id 0 --model_to_train densenet --epochs 200 --pretrained %j --dataset ckp --save_ckpt_intv 50 --batch_size 8 --lr_gen 0.001 --trainsplit train_ids_%i.csv --testsplit test_ids_%i.csv )
