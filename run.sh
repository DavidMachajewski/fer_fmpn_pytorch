for %i in (2,3,4) do (for %j in (0,1) do python main.py --mode train --gpu_id 0 --model_to_train incv3 --epochs 200 --pretrained %j --dataset ckp --save_ckpt_intv 50 --batch_size 8 --lr_gen 0.001 --trainsplit train_ids_%i.csv --testsplit test_ids_%i.csv )



do python main.py --mode train --gpu_id 0 --model_to_train incv3 --epochs 200 --pretrained %j --dataset ckp --batch_size 8 --lr_gen 0.001 --trainsplit train_ids_%i.csv --testsplit test_ids_%i.csv




for %i in (5,6,7) do (for %j in (0,1) do python main.py --mode train --gpu_id 0 --model_to_train incv3 --epochs 200 --pretrained %j --dataset ckp --save_ckpt_intv 50 --batch_size 8 --lr_gen 0.001 --trainsplit train_ids_%i.csv --testsplit test_ids_%i.csv )


# densenet
for %i in (8, 9) do (for %j in (0,1) do python main.py --mode train --gpu_id 0 --model_to_train densenet --epochs 200 --pretrained %j --dataset ckp --save_ckpt_intv 100 --batch_size 8 --lr_gen 0.001 --trainsplit train_ids_%i.csv --testsplit test_ids_%i.csv )
for %i in (0) do (for %j in (0) do python main.py --mode train --gpu_id 0 --model_to_train densenet --epochs 200 --pretrained %j --dataset ckp --save_ckpt_intv 100 --batch_size 8 --lr_gen 0.001 --trainsplit train_ids_%i.csv --testsplit test_ids_%i.csv )



# fmpn
# for %i in (0) do python main.py --mode train --gpu_id 0 --model_to_train fmpn --epochs 500 --save_ckpt_intv 50 --load_size 320 --final_size 299 --load_ckpt_fmg_only 1 --fmpn_cn
# inc_v3 --fmpn_cn_pretrained 1 --dataset ckp --batch_size 8 --scheduler_type linear_x --ckpt_fmg ./results/run_fmg_2021-04-14_13-57-45/train_fmg_2021-04-14_13-57-45\ckpt/fmg_ckpt.pth.tar --trainsplit train_ids_%i.csv --testsplit test_i
# ds_%i.csv


for %i in (1) do (for %j in (0) do python main.py --mode train --gpu_id 0 --model_to_train fmpn --epochs 500 --save_ckpt_intv 50 --load_size 320 --final_size 299 --load_ckpt_fmg_only 1 --fmpn_cn inc_v3 --fmpn_cn_pretrained %j --dataset ckp --batch_size 8 --scheduler_type linear_x --ckpt_fmg ./results/run_fmg_2021-04-14_13-57-45/train_fmg_2021-04-14_13-57-45\ckpt/fmg_ckpt.pth.tar --trainsplit train_ids_%i.csv --testsplit test_ids_%i.csv)

# to train fmpn on new trained fmg
for %i in (0) do (for %j in (0) do python main.py --mode train --gpu_id 0 --model_to_train fmpn --epochs 500 --save_ckpt_intv 50 --load_size 320 --final_size 299 --load_ckpt_fmg_only 1 --fmpn_cn inc_v3 --fmpn_cn_pretrained %j --dataset ckp --batch_size 8 --scheduler_type linear_x --ckpt_fmg ./results/run_fmg_2021-05-19_13-23-51/train_fmg_2021-05-19_13-23-51/ckpt/fmg_2021-05-19_13-23-51_epoch_299_ckpt.pth.tar --trainsplit train_ids_%i.csv --testsplit test_ids_%i.csv)
#
# :TODO:
# train FMG for training split 1 and 2
# then train FMPN accordingly
#
# Implement FERP and AFFECTNET and try FMPN on FERP first
#

testsplit 0:
/run_fmg_2021-05-19_13-23-51/train_fmg_2021-05-19_13-23-51/ckpt/fmg_2021-05-19_13-23-51_epoch_299_ckpt.pth.tar


testsplit 2:
/run_fmg_2021-05-20_16-19-04/train_fmg_2021-05-20_16-19-04/ckpt/fmg_2021-05-20_16-19-04_epoch_299_ckpt.pth.tar

for %i in (0) do (for %j in (0) do python main.py --mode train --gpu_id 0 --model_to_train fmpn --epochs 500 --save_ckpt_intv 100 --load_size 320 --final_size 299 --load_ckpt_fmg_only 1 --fmpn_cn inc_v3 --fmpn_cn_pretrained %j --dataset ckp --batch_size 8 --scheduler_type linear_x --ckpt_fmg ./results/run_fmg_2021-05-20_16-19-04/train_fmg_2021-05-20_16-19-04/ckpt/fmg_2021-05-20_16-19-04_epoch_299_ckpt.pth.tar --trainsplit train_ids_%i.csv --testsplit test_ids_%i.csv)


for %i in (0) do (for %j in (1) do python main.py --mode train --gpu_id 0 --model_to_train fmpn --epochs 500 --save_ckpt_intv 100 --load_size 320 --final_size 299 --load_ckpt_fmg_only 1 --fmpn_cn inc_v3 --fmpn_cn_pretrained %j --dataset ckp --batch_size 16 --scheduler_type linear_x --ckpt_fmg ./results/run_fmg_2021-05-27_10-37-01/train_fmg_2021-05-27_10-37-01/ckpt/fmg_2021-05-27_10-37-01_epoch_299_ckpt.pth.tar --trainsplit train_ids_%i.csv --testsplit test_ids_%i.csv)

./results/run_fmg_2021-05-27_10-37-01/train_fmg_2021-05-27_10-37-01/ckpt/fmg_2021-05-27_10-37-01_epoch_299_ckpt.pth.tar





fer

for %i in (0) do (for %j in (0) do python main.py --mode train --gpu_id 0 --model_to_train incv3 --epochs 100 --pretrained %j --dataset fer --save_ckpt_intv 50 --batch_size 16 --lr_gen 0.001 --trainsplit train_ids_%i.csv --testsplit test_ids_%i.csv )



































for %i in (0) do (for %j in (0) do python main.py --mode train --gpu_id 0 --model_to_train fmpn --epochs 304 --save_ckpt_intv 100 --load_size 320 --final_size 299 --load_ckpt_fmg_only 1 --fmpn_cn inc_v3 --fmpn_cn_pretrained %j --dataset ckp --batch_size 8 --scheduler_type linear_x --ckpt_fmg ./results/run_fmg_2021-05-27_10-37-01/train_fmg_2021-05-27_10-37-01/ckpt/fmg_2021-05-27_10-37-01_epoch_299_ckpt.pth.tar --trainsplit train_ids_%i.csv --testsplit test_ids_%i.csv)


python3 main.py --mode train --gpu_id 1 --model_to_train fmpn --epochs 5 --save_ckpt_intv 100 --load_size 320 --final_size 299 --load_ckpt_fmg_only 1 --fmpn_cn inc_v3 --fmpn_cn_pretrained 0 --dataset ckp --batch_size 16 --scheduler_type linear_x --ckpt_fmg ./results/run_fmg_2021-05-27_10-37-01/train_fmg_2021-05-27_10-37-01/ckpt/fmg_2021-05-27_10-37-01_epoch_299_ckpt.pth.tar --trainsplit train_ids_0.csv --testsplit test_ids_0.csv