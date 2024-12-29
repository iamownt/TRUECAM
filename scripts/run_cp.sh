#!/bin/bash


start_time=$(date +%s)
echo "Starting script at: $(date)"

threshold_range=(None 0.5 1.0 2.0 3.0 4.0 5.0 6.0)
RESULTS_FILE_PATH="results_cp_evaluate.csv"


for threshold in "${threshold_range[@]}"; do
    if [ "$threshold" == "None" ]; then
      additional_arg=()
    else
      additional_arg=( "--gaussian_process" )
    fi
    echo "Threshold: $threshold", "Additional args:", "${additional_arg[@]}"
    python evaluate_everything.py --model_name uni --spec_norm_bound "$threshold" \
                                  --save_to_parquet --eval_type cp \
                                  --results_file_path $RESULTS_FILE_PATH "${additional_arg[@]}"

    python evaluate_everything.py --model_name conch --spec_norm_bound "$threshold" \
                                --save_to_parquet --eval_type cp \
                                --results_file_path $RESULTS_FILE_PATH "${additional_arg[@]}"

done

end_time=$(date +%s)
duration=$((end_time - start_time))

echo "Script finished at: $(date)"
echo "Total duration: $((duration / 60)) minutes and $((duration % 60)) seconds"