# Concept extraction framework
This code repo is based on the Concept Extraction Framework code from Fernando Julio Cendra (cendra@hku.hk), which is modified from: ConceptExpress: https://github.com/haoosz/ConceptExpress.
Segmentation code for "DiffSeg: A Segmentation Model for Skin Lesions Based on Diffusion Difference" is also applied for segmentation.

`@article{tian2023diffuse,
 title={Diffuse, Attend, and Segment: Unsupervised Zero-Shot Segmentation using Stable Diffusion},
 author={Tian, Junjiao and Aggarwal, Lavisha and Colaco, Andrea and Kira, Zsolt and Gonzalez-Franco, Mar},
 journal={arXiv preprint arXiv:2308.12469},
 year={2023}
}`

`@InProceedings{hao2024conceptexpress,
    title={Concept{E}xpress: Harnessing Diffusion Models for Single-image Unsupervised Concept Extraction}, 
    author={Shaozhe Hao and Kai Han and Zhengyao Lv and Shihao Zhao and Kwan-Yee~K. Wong},
    booktitle={ECCV},
    year={2024},
}`

## Training

1.  **Prepare your dataset:**

    *   Create a directory for each concept you want to train. Each directory (e.g., `datasets/0/`, `datasets/1/`, etc.) should contain:
        *   `img.jpg`: The main image.
        *   `mask_*.jpg`: Segmentation masks for each asset in the image, named as `mask_{order}.jpg`.

2.  **Configure training parameters:**

    *   Edit the `scripts/run.sh` script to set the desired training parameters. Key parameters include:
        *   `--instance_data_dir`: Path to the concept data directory (e.g., `datasets/0/`).
        *   `--class_data_dir`: Path to the prior preservation images directory (e.g., `outputs/preservation_images/`).
        *   `--phase1_train_steps`: Number of training steps for the first phase.
        *   `--phase2_train_steps`: Number of training steps for the second phase.
        *   `--output_dir`: Path to the output directory for saving the trained model.
        *   `--lambda_attention`: The weight of attention loss.
        *   `--lambda_cls`: The weight of classification loss.
        *   `--t_dist`: Temperature for the weighted timestep based on ReVersion.
        *   `--noise_offset`:  The offset for noise.
        *   `--asset_token`: The concept-specific placeholder token.

3.  **Run the training script:**

    Execute the `scripts/train.sh` script to start the training process.

    ```bash
    CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh
    ```

4.  **Run the training script:**

   Execute the `scripts/evaluate.sh` script to generate the necessary files for evaluation. This may take very long time

## Inference

1.  **Set the model path:**

    *   Edit the `scripts/infer.sh` script and set the `MODEL_PATH` variable to the path of your trained model.

2.  **Configure inference parameters:**

    *   Modify the `--prompt` argument in `scripts/infer.sh` to specify the desired prompt.  Make sure to include the asset tokens (e.g., `<asset0>`).
    *   Set the `--output_path` argument to the desired path for the output image.

3.  **Run the inference script:**

    Execute the `scripts/infer.sh` script to generate an image using the trained model.

    ```bash
    CUDA_VISIBLE_DEVICES=0 bash scripts/infer.sh
    ```

