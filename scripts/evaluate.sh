#!/usr/bin/env bash
# set -e  # exit on error

# Initialize conda for this shell
eval "$(conda shell.bash hook)" 

# ### prepare the diffseg mask
# conda activate diffseg ### replace the path with your diffseg environment namee
# python "./segment/apply_diffseg_regular.py" --img_root "/home/angus/uce_images" --result_directory "./segment/segment_results"

# train the model and prepare the concept inference
conda activate apai3799 ### replace the path with your conceptexpress environment namee
# bash ./scripts/prepare_eval.sh

### compute the similarity score
python ./evaluate/simc_clip.py
python ./evaluate/simc_dino.py
python ./evaluate/simi_clip.py
python ./evaluate/simi_dino.py

### compute the classification score
python ./evaluate/acc_clip.py
python ./evaluate/acc_dino.py