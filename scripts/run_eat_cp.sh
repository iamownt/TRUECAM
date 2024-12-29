#!/bin/bash

start_time=$(date +%s)
echo "Starting script at: $(date)"

RESULTS_FILE_PATH="results_sngp_eat_none_cp_evaluate.csv"


for mask_type in "in_slide" "rand"
  do
  for threshold in 0.001 0.01 1.0 $(seq 0.1 0.1 0.9)
  do
      echo "Running with mask_tile_threshold=$threshold"
      python evaluate_everything.py --model_name uni --spec_norm_bound 2.0 \
              --gaussian_process --eval_type cp \
              --ispecial_prefix "maskin_slide_thres${threshold}_invert0" \
              --especial_prefix "maskin_slide_thres${threshold}_invert0" \
              --results_file_path "$RESULTS_FILE_PATH"

      python evaluate_everything.py --model_name conch --spec_norm_bound 2.0 \
              --gaussian_process --eval_type cp \
              --ispecial_prefix "mask${mask_type}_thres${threshold}_invert0_nearby_tiles1" \
              --especial_prefix "mask${mask_type}_thres${threshold}_invert0_nearby_tiles1" \
              --results_file_path "$RESULTS_FILE_PATH"
  done
done


end_time=$(date +%s)
duration=$((end_time - start_time))

echo "Script finished at: $(date)"
echo "Total duration: $((duration / 60)) minutes and $((duration % 60)) seconds"