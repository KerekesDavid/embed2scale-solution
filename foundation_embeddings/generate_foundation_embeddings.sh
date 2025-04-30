#!/bin/bash

conda activate embed2scale_solution

data_dir=../data
downstream_data_dir=${data_dir}/SSL4EO-S12-downstream

python3 embed_copernicus.py ${downstream_data_dir}/data_dev/ ${data_dir}/copernicus-fm/copernicusfm_concat_temporal_raw_4x768_dev.npz
python3 embed_copernicus.py ${downstream_data_dir}/data_eval/ ${data_dir}/copernicus-fm/copernicusfm_concat_temporal_raw_4x768_eval.npz

python3 embed_dofa.py ${downstream_data_dir}/data_dev/ ${data_dir}/dofa/dofa_temporal_raw_4x1536_dev.npz
python3 embed_dofa.py ${downstream_data_dir}/data_eval/ ${data_dir}/dofa/dofa_temporal_raw_4x1536_eval.npz

python3 embed_croma.py ${downstream_data_dir}/data_dev/ ${data_dir}/croma-dev-embeddings
python3 embed_croma.py ${downstream_data_dir}/data_eval/ ${data_dir}/croma-eval-embeddings

python3 embed_prithvi.py ${downstream_data_dir}/data_dev/ ${data_dir}/prithvi2/prithvi2_1x1536_dev.npz
python3 embed_prithvi.py ${downstream_data_dir}/data_eval/ ${data_dir}/prithvi2/prithvi2_1x1536_eval.npz

python3 embed_scale_mae.py ${downstream_data_dir}/data_dev/ ${data_dir}/scalemae_temporal_data_dev_4x1536.npz
python3 embed_scale_mae.py ${downstream_data_dir}/data_eval/ ${data_dir}/scalemae_temporal_data_eval_4x1536.npz

conda deactivate
