import os
import argparse
import warnings


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-2-1-base", 
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--revision", type=str, default=None, required=False,
        help="Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be float32 precision.",
    )
    parser.add_argument("--tokenizer_name", type=str, default=None, help="Pretrained tokenizer name or path if not the same as model_name",)
    parser.add_argument("--instance_data_dir", type=str, default=None, required=True, help="A folder containing the training data of instance images.",)
    parser.add_argument("--class_data_dir", type=str, default=None, required=False, help="A folder containing the training data of class images.",)
    parser.add_argument("--class_prompt", type=str, default="a photo at the beach", help="The prompt to specify images in the same class as provided instance images.",)
    parser.add_argument("--no_prior_preservation", action="store_false", dest="with_prior_preservation", help="Flag to add prior preservation loss.")
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.",)
    parser.add_argument("--num_class_images", type=int, default=100,
        help="Minimal class images for prior preservation loss. If there are not enough images already present in class_data_dir, additional images will be sampled with class_prompt.",
    )
    parser.add_argument("--output_dir", type=str, default="outputs", help="The output directory where the model predictions and checkpoints will be written.",)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512, help="The resolution for input images, all the images in the train/validation dataset will be resized to this .resolution",)
    parser.add_argument("--center_crop", default=False, action="store_true",
        help="Whether to center crop the input images to the resolution. If not set, the images will be randomly cropped. The images will be resized to the resolution first before cropping.",
    )
    parser.add_argument("--no_train_text_encoder", action="store_false", dest="train_text_encoder", help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",)
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader.",)
    parser.add_argument("--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images.",)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--phase1_train_steps", type=int, default="400", help="Number of trainig steps for the first phase.",)
    parser.add_argument("--phase2_train_steps", type=int, default="400", help="Number of trainig steps for the second phase.",)
    parser.add_argument("--checkpointing_steps", type=int, default=5000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
        help="Whether training should be resumed from a previous checkpoint. Use a path saved by `--checkpointing_steps`, or 'latest' to automatically select the last available checkpoint.",
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",)
    parser.add_argument("--learning_rate", type=float, default=2e-6, help="Initial learning rate (after the potential warmup period) to use.",)
    parser.add_argument("--initial_learning_rate", type=float, default=5e-4, help="The LR for the Textual Inversion steps.",)
    parser.add_argument("--scale_lr", action="store_true", default=False, help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
        help='The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial","constant", "constant_with_warmup"]',
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler.",)
    parser.add_argument("--lr_num_cycles", type=int, default=1, help="Number of hard resets of the lr in cosine_with_restarts scheduler.",)
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.",)
    parser.add_argument("--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes.",)
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.",)
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.",)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer",)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.",)
    parser.add_argument("--hub_model_id", type=str, default=None, help="The name of the repository to keep in sync with the local `output_dir`.",)
    parser.add_argument("--logging_dir", type=str, default="logs",
        help="[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.",
    )
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--report_to", type=str, default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--prior_generation_precision", type=str, default=None, choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank",)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.",)
    parser.add_argument("--set_grads_to_none", action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument("--lambda_attention", type=float, default=1e-5)
    parser.add_argument("--img_log_steps", type=int, default=200)
    parser.add_argument("--placeholder_token", type=str, default="<asset>", help="A token to use as a placeholder for the concept.",)
    parser.add_argument("--do_not_apply_masked_loss", action="store_false", help="Use masked loss instead of standard epsilon prediciton loss", dest="apply_masked_loss")
    parser.add_argument("--log_checkpoints", action="store_true", help="Indicator to log intermediate model checkpoints",)

    # configs added
    parser.add_argument("--noise_offset", type=float, default=0.1, help="https://www.crosslabs.org//blog/diffusion-with-offset-noise")
    parser.add_argument("--t_dist", type=float, default=0.0, help="Temperature for the weighted timestep based on ReVersion",)
    parser.add_argument("--asset_token", type=str, default="<asset>", help="A token to use as a concept-specific placeholder for the concept.",)
    parser.add_argument("--text_encoder_use_attention_mask", action="store_true", required=False, help="Whether to use attention mask for the text encoder",)

    if input_args is not None: args = parser.parse_args(input_args)
    else: args = parser.parse_args()

    args.max_train_steps = args.phase1_train_steps + args.phase2_train_steps

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn(
                "You need not use --class_data_dir without --with_prior_preservation."
            )
        if args.class_prompt is not None:
            warnings.warn(
                "You need not use --class_prompt without --with_prior_preservation."
            )

    return args