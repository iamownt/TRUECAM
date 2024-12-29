#!/bin/bash


start_time=$(date +%s)
echo "Starting script at: $(date)"

invert_thres="none"
for dataset in "tcga" "cptac"; do
    for mask_tile_category in "rand" "in_slide"; do
      echo "Running with dataset=$dataset, invert_thres=$invert_thres, mask_tile_category=$mask_tile_category"
      RESULTS_FILE_PATH="results_sngp_eat_${invert_thres}_evaluate.csv"
      for threshold in 0.001 0.01 1.0 $(seq 0.1 0.1 0.9)
        do
            echo "Running with mask_tile_threshold=$threshold"
            python data_specific_abmil.py --model_name uni --spec_norm_bound 2. \
                  --save_to_parquet --gaussian_process --training_dataset "$dataset" \
                  --evaluate_only --mask_tile --mask_tile_threshold "$threshold" \
                  --mask_tile_category "$mask_tile_category" \
                  --results_file_path "$RESULTS_FILE_PATH"

            python data_specific_abmil.py --model_name conch --spec_norm_bound 2. \
                  --save_to_parquet --gaussian_process --training_dataset "$dataset" \
                  --evaluate_only --mask_tile --mask_tile_threshold "$threshold" \
                  --mask_tile_category "$mask_tile_category" \
                  --results_file_path "$RESULTS_FILE_PATH"
        done
      fi

  done
done


end_time=$(date +%s)
duration=$((end_time - start_time))

echo "Script finished at: $(date)"
echo "Total duration: $((duration / 60)) minutes and $((duration % 60)) seconds"