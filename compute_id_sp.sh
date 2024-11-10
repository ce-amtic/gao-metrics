# python eval/instantiation_distance_sp.py \
#     --gt_dir eval/testgt \
#     --data_dir eval/testin \
#     --output_dir eval/ID_output \
#     --N_states 10 \
#     --N_pcl 4096 \
#     --world_size 4

# python eval/instantiation_distance_sp.py \
#     --gt_dir StorageFurniture_128/gt/data_gt \
#     --data_dir StorageFurniture_128/2024-11-05_19-28-28/data \
#     --output_dir StorageFurniture_128/2024-11-05_19-28-28/eval_output \
#     --sample_file_path eval/selected_files.json \
#     --N_states 10 \
#     --N_pcl 4096 \
#     --world_size 4

python eval/instantiation_distance_sp.py \
    --gt_dir StorageFurniture_768/gt/data_gt \
    --data_dir StorageFurniture_768/2024-11-09_15-42-37/data \
    --output_dir StorageFurniture_768/2024-11-09_15-42-37/eval_output \
    --sample_file_path eval/selected_files.json \
    --N_states 10 \
    --N_pcl 4096 \
    --world_size 4