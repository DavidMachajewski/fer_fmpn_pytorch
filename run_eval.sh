python eval.py --mode test --gpu_id 0 --model_to_train fmpn --load_ckpt 1 --dataset ckp --load_size 320 --final_size 299 --batch_size 16 --load_ckpt_fmg_only 0 --fmpn_cn inc_v3 --fmpn_cn_pretrained 1 --ckpt_fmg ./results/run_fmg_2021-06-22_17-16-28/train_fmg_2021-06-22_17-16-28/ckpt/fmg_2021-06-22_17-16-28_epoch_299_ckpt.pth.tar --ckpt_pfn ... --ckpt_cn ... --trainsplit train_ids_7.csv --testsplit test_ids_7.csv



python eval.py --mode test --gpu_id 0 --model_to_train fmpn --load_ckpt 1 --dataset ckp --load_size 320 --final_size 299 --batch_size 16 --load_ckpt_fmg_only 0 --fmpn_cn inc_v3 --fmpn_cn_pretrained 1 --ckpt_fmg  --ckpt_pfn "F:\trainings2\fmpn\run_fmpn_2021-06-27_15-39-08\train_fmpn_2021-06-27_15-39-08\ckpt\fmpn_pfn_2021-06-27_15-39-08_epoch_449_ckpt.pth.tar" --ckpt_cn "F:\trainings2\fmpn\run_fmpn_2021-06-27_15-39-08\train_fmpn_2021-06-27_15-39-08\ckpt\fmpn_cn_2021-06-27_15-39-08_epoch_449_ckpt.pth.tar" --trainsplit train_ids_7.csv --testsplit test_ids_7.csv