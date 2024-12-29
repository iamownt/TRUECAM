#!/bin/bash

start_time=$(date +%s)
echo "Starting script at: $(date)"

exp_name="sngp"
folds=20
task_cfg_path="finetune/task_configs/nsclc.yaml"
mask_tile_threshold=0.4

# generate ood parquet compatible
ood_dataset_names=("ucs" "uvm" "blca" "acc")

for ood_dataset in "${ood_dataset_names[@]}"; do
    python finetune/generate_ood_parquet_compatible.py --exp_name "$exp_name" --folds "$folds" --gaussian_process \
        --task_cfg_path "$task_cfg_path" --evaluate_only --ood_dataset_name "${ood_dataset}" \
        --dataset_csv "../dataset_csv/ood_dataset/${ood_dataset}.csv" \
        --root_path "{$HOME}/sngp/UniConch/prov-gigapath_${ood_dataset}_h5file" \
        --save_dir "{$HOME}/wangtao/prov-gigapath/prov-gigapath/outputs/tcga"

    python finetune/generate_ood_parquet_compatible.py --exp_name "$exp_name" --folds "$folds" --gaussian_process \
        --task_cfg_path "$task_cfg_path" --evaluate_only --ood_dataset_name "${ood_dataset}" \
        --dataset_csv "../dataset_csv/ood_dataset/${ood_dataset}.csv" --mask_tile --mask_tile_threshold "$mask_tile_threshold" \
        --mask_pkl_path "{$HOME}/sngp/UniConch/models/ambpkl/newambk/conch_${ood_dataset}_ambiguity_dict_autogluon_0.2_tuning0.pkl" \
        --root_path "{$HOME}/sngp/UniConch/prov-gigapath_${ood_dataset}_h5file" \
        --save_dir "{$HOME}/wangtao/prov-gigapath/prov-gigapath/outputs/tcga"
done

to_destination="{$HOME}/wangtao/prov-gigapath/prov-gigapath/outputs/tcga/nsclc/${exp_name}/eval_pretrained_nsclc"


python evaluate_everything.py --eval_type ood --model_name prov-gigapath \
    --save_destination "$to_destination" --trial 4 --fold 5 \
    --ood_dataset_name unified --ood_detection_type probability

python evaluate_everything.py --eval_type ood --model_name prov-gigapath \
    --save_destination "$to_destination" --trial 4 --fold 5 \
    --ood_dataset_name unified --ood_detection_type uncertainty


python evaluate_everything.py --eval_type ood --model_name prov-gigapath \
    --save_destination "$to_destination" --trial 4 --fold 5 \
    --ood_dataset_name unified --ood_detection_type uncertainty --mask_tile --mask_tile_threshold "$mask_tile_threshold"



end_time=$(date +%s)
duration=$((end_time - start_time))

echo "Script finished at: $(date)"
echo "Total duration: $((duration / 60)) minutes and $((duration % 60)) seconds"