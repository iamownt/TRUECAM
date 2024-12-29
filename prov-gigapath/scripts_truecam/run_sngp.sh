#!/bin/bash

start_time=$(date +%s)
echo "Starting script at: $(date)"

exp_name="sngp"
folds=20
task_cfg_path="finetune/task_configs/nsclc.yaml"
dataset_csv="../dataset_csv/nsclc/nsclc_labels_one_run.csv"
root_path="{$HOME}/sngp/project/destination_20X/h5file"
save_dir="{$HOME}/wangtao/TRUECAM/prov-gigapath/outputs/tcga"
mask_tile_threshold=0.4
mask_pkl_path="{$HOME}/sngp/UniConch/models/ambpkl/newambk/conch_itest_ambiguity_dict_autogluon_0.2_tuning0.pkl"


python finetune/main.py --exp_name $exp_name --folds "$folds" --gaussian_process \
        --task_cfg_path "$task_cfg_path" --report_to wandb --model_select val \
        --dataset_csv "$dataset_csv" --root_path "$root_path" --save_dir "$save_dir"


python finetune/main.py --exp_name $exp_name --folds "$folds" --gaussian_process \
        --task_cfg_path  "$task_cfg_path" --report_to wandb --model_select val \
        --dataset_csv "$dataset_csv" --root_path "$root_path" --save_dir "$save_dir" \
        --evaluate_only --mask_tile --mask_tile_threshold "$mask_tile_threshold" \
        --mask_pkl_path "$mask_pkl_path"


end_time=$(date +%s)
duration=$((end_time - start_time))

echo "Script finished at: $(date)"
echo "Total duration: $((duration / 60)) minutes and $((duration % 60)) seconds"