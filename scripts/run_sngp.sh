#!/bin/bash

start_time=$(date +%s)
echo "Starting script at: $(date)"

RESULTS_FILE_PATH="results_sngp_oct29_evaluate.csv"

# Loop through your configurations
for model_name in "uni" "conch"; do
    for spec_norm_bound in "None" "0.5" "1." "2." "3." "4." "5." "6."; do
        for dataset in "tcga" "cptac"; do
              additional_arg="--gaussian_process"
              if [ "$spec_norm_bound" == "None" ]; then
                additional_arg=""
              fi
            echo "Running model: $model_name, spec_norm_bound: $spec_norm_bound, dataset: $dataset"
            # Run the python command in the background
            python data_specific_abmil.py \
                --model_name $model_name \
                --spec_norm_bound $spec_norm_bound \
                --save_to_parquet \
                --training_dataset $dataset \
                --results_file_path $RESULTS_FILE_PATH "$additional_arg"
        done
    done
done


end_time=$(date +%s)
duration=$((end_time - start_time))

echo "Script finished at: $(date)"
echo "Total duration: $((duration / 60)) minutes and $((duration % 60)) seconds"
