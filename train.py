# ---------------------------------------------------------------------------------------------
# Main file of ConceptExpress + DiffSeg framework
# Stage Two: Structured Concept Learning
#
# Copyright 2025, by Fernando Julio Cendra (cendra@hku.hk)
# Modified from: ConceptExpress: https://github.com/haoosz/ConceptExpress
# ---------------------------------------------------------------------------------------------
import os
import hashlib
import itertools
import logging
import math
import random
from pathlib import Path
from tqdm.auto import tqdm
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import WeightedRandomSampler

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, ProjectConfiguration
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
    DDIMScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from transformers import AutoTokenizer

from utils.config import parse_args
from utils.ptp_utils import (
    P2PCrossAttnProcessor, 
    AttentionStore, 
    wasser_loss,
)
from utils.dataset import (
    DreamBoothDataset, 
    PromptDataset, 
    collate_fn, 
    prompt_template,
)
from utils.model_util import (
    import_model_class_from_model_name_or_path, 
    tokenize_prompt, 
    encode_prompt,
)

check_min_version("0.12.0")

logger = get_logger(__name__)


class ConceptExtraction:
    def __init__(self):
        self.args = parse_args()
        self.main()

    def main(self):
        logging_dir = Path(self.args.output_dir, self.args.logging_dir)
        accelerator_project_config = ProjectConfiguration(project_dir=self.args.output_dir, logging_dir=logging_dir)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=self.args.mixed_precision,
            log_with=self.args.report_to,
            project_config=accelerator_project_config,
        )

        if (
            self.args.train_text_encoder
            and self.args.gradient_accumulation_steps > 1
            and self.accelerator.num_processes > 1
        ):
            raise ValueError(
                "Gradient accumulation is not supported when training the text encoder in distributed training. "
                "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
            )

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(self.accelerator.state, main_process_only=False)

        # If passed along, set the training seed now.
        if self.args.seed is not None: set_seed(self.args.seed)

        # Generate class images if prior preservation is enabled.
        if self.args.with_prior_preservation:
            class_images_dir = Path(self.args.class_data_dir)
            if not class_images_dir.exists(): class_images_dir.mkdir(parents=True)
            cur_class_images = len(list(class_images_dir.iterdir()))

            if cur_class_images < self.args.num_class_images:
                torch_dtype = (
                    torch.float16
                    if self.accelerator.device.type == "cuda"
                    else torch.float32
                )
                if self.args.prior_generation_precision == "fp32": torch_dtype = torch.float32
                elif self.args.prior_generation_precision == "fp16": torch_dtype = torch.float16
                elif self.args.prior_generation_precision == "bf16": torch_dtype = torch.bfloat16
                pipeline = DiffusionPipeline.from_pretrained(
                    self.args.pretrained_model_name_or_path,
                    torch_dtype=torch_dtype,
                    safety_checker=None,
                    revision=self.args.revision,
                )
                pipeline.set_progress_bar_config(disable=True)
                num_new_images = self.args.num_class_images - cur_class_images
                logger.info(f"Number of class images to sample: {num_new_images}.")
                sample_dataset = PromptDataset(self.args.class_prompt, num_new_images)
                sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=self.args.sample_batch_size)
                sample_dataloader = self.accelerator.prepare(sample_dataloader)
                pipeline.to(self.accelerator.device)

                for example in tqdm(
                    sample_dataloader,
                    desc="Generating class images",
                    disable=not self.accelerator.is_local_main_process,
                ):
                    images = pipeline(example["prompt"]).images

                    for i, image in enumerate(images):
                        hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                        image_filename = (
                            class_images_dir
                            / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                        )
                        image.save(image_filename)

                del pipeline
                if torch.cuda.is_available(): torch.cuda.empty_cache()

        # Handle the directory creation
        if self.accelerator.is_main_process: os.makedirs(self.args.output_dir, exist_ok=True)

        # import correct text encoder class
        text_encoder_cls = import_model_class_from_model_name_or_path(self.args.pretrained_model_name_or_path, self.args.revision)

        # Load scheduler and models
        self.noise_scheduler = DDPMScheduler.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="scheduler")
        self.text_encoder = text_encoder_cls.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="text_encoder", revision=self.args.revision,)
        self.vae = AutoencoderKL.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="vae", revision=self.args.revision,)
        self.unet = UNet2DConditionModel.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="unet", revision=self.args.revision,)

        # Load the tokenizer
        if self.args.tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer_name, revision=self.args.revision, use_fast=False)
        elif self.args.pretrained_model_name_or_path:
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="tokenizer", revision=self.args.revision, use_fast=False,)

        # ***************** Process datasets *****************
        dataset_path = self.args.instance_data_dir
        image_path = os.path.join(dataset_path, "img.jpg")
        mask_paths = [f for f in os.listdir(dataset_path) if f.startswith("mask")]

        self.num_of_assets = len(mask_paths)

        # self.object_anchors = []
        # for mask_file in mask_paths:
        #     object_name = mask_file.split("_")[1]
        #     self.object_anchors.append(object_name)
        # ***************** Process datasets *****************
        self.asset_tokens = [self.args.asset_token.replace(">", f"{idx}>") for idx in range(self.num_of_assets)]

        # add asset tokens to tokenizer
        num_added_tokens = self.tokenizer.add_tokens(self.asset_tokens)

        #print num of added tokens
        print("Number of added tokens: ", num_added_tokens)

        # Convert assets tokens to ids
        self.placeholder_token_ids = {}
        self.placeholder_token_ids["asset"] = self.tokenizer.convert_tokens_to_ids(self.asset_tokens)

        # Resize the token embeddings
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        token_embeds = self.text_encoder.get_input_embeddings().weight.data
        # for tkn_idx, initializer_token in enumerate(self.object_anchors):
        #     # initialize the concept specific token embeddings
        #     curr_token_ids = self.tokenizer.encode(initializer_token, add_special_tokens=False)
        #     token_embeds[self.placeholder_token_ids["asset"][tkn_idx]] = token_embeds[curr_token_ids[0]]  

        # Prepare placeholder tokens
        self.placeholder_tokens= [f"{asset}" for asset in self.asset_tokens]

        # Set validation scheduler for logging
        self.validation_scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        self.validation_scheduler.set_timesteps(50)

        # We start by only optimizing the embeddings
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)

        # Freeze all parameters except for the token embeddings in text encoder
        self.text_encoder.text_model.encoder.requires_grad_(False)
        self.text_encoder.text_model.final_layer_norm.requires_grad_(False)
        self.text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

        if self.args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                self.unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError(
                    "xformers is not available. Make sure it is installed correctly"
                )

        if self.args.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
            if self.args.train_text_encoder:
                self.text_encoder.gradient_checkpointing_enable()

        if self.args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        if self.args.scale_lr:
            self.args.learning_rate = (
                self.args.learning_rate
                * self.args.gradient_accumulation_steps
                * self.args.train_batch_size
                * self.accelerator.num_processes
            )

        if self.args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )
            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        # We start by only optimizing the embeddings
        params_to_optimize = self.text_encoder.get_input_embeddings().parameters()
        self.optimizer = optimizer_class(
            params_to_optimize,
            lr=self.args.initial_learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay,
            eps=self.args.adam_epsilon,
        )

        # Dataset and DataLoaders creation:
        train_dataset = DreamBoothDataset(
            instance_data_root=self.args.instance_data_dir,
            image_path=image_path,
            mask_paths=mask_paths,
            class_data_root=self.args.class_data_dir
            if self.args.with_prior_preservation
            else None,
            class_prompt=self.args.class_prompt,
            tokenizer=self.tokenizer,
            size=self.args.resolution,
            center_crop=self.args.center_crop,
            num_of_assets=self.num_of_assets,
        )
        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            collate_fn=lambda examples: collate_fn(examples, self.args.with_prior_preservation),
            num_workers=self.args.dataloader_num_workers,
        )

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.args.gradient_accumulation_steps)
        if self.args.max_train_steps is None:
            self.args.max_train_steps = (self.args.num_train_epochs * num_update_steps_per_epoch)
            overrode_max_train_steps = True

        self.lr_scheduler = get_scheduler(
            self.args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.lr_warmup_steps
            * self.args.gradient_accumulation_steps,
            num_training_steps=self.args.max_train_steps
            * self.args.gradient_accumulation_steps,
            num_cycles=self.args.lr_num_cycles,
            power=self.args.lr_power,
        )

        (self.unet, self.text_encoder, self.optimizer, self.train_dataloader, self.lr_scheduler) = self.accelerator.prepare(
            self.unet, self.text_encoder, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16": self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16": self.weight_dtype = torch.bfloat16

        # Move vae and text_encoder to device and cast to weight_dtype
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)

        low_precision_error_string = (
            "Please make sure to always have all model weights in full float32 precision when starting training - even if"
            " doing mixed precision training. copy of the weights should still be float32."
        )

        if self.accelerator.unwrap_model(self.unet).dtype != torch.float32:
            raise ValueError(
                f"Unet loaded as datatype {self.accelerator.unwrap_model(self.unet).dtype}. {low_precision_error_string}"
            )

        if (self.args.train_text_encoder and self.accelerator.unwrap_model(self.text_encoder).dtype != torch.float32):
            raise ValueError(
                f"Text encoder loaded as datatype {self.accelerator.unwrap_model(self.text_encoder).dtype}."
                f" {low_precision_error_string}"
            )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.args.gradient_accumulation_steps)
        if overrode_max_train_steps: self.args.max_train_steps = (self.args.num_train_epochs * num_update_steps_per_epoch)

        # Afterwards we recalculate our number of training epochs
        self.args.num_train_epochs = math.ceil(self.args.max_train_steps / num_update_steps_per_epoch)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if self.accelerator.is_main_process: self.accelerator.init_trackers("ice", config=vars(self.args))

        # Train
        total_batch_size = (
            self.args.train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num batches each epoch = {len(self.train_dataloader)}")
        logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        logger.info(f"  Noise offset = {self.args.noise_offset}")
        logger.info(f"  T dist = {self.args.t_dist}")
        logger.info(f"  Instantaneous batch size per device = {self.args.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.args.max_train_steps}")

        # log tokens
        logger.info(f"  Asset tokens = {self.asset_tokens}")

        # Create attention controller
        self.controller = AttentionStore()
        self.register_attention_control(self.controller)

    def train(self):
        """
        Execute the training process for Concept Extraction framework.
        """
        # Start tracking from the first epoch after potential checkpoint resumption
        first_epoch = 0
        global_step = 0
        
        # Setup progress tracking
        if self.args.resume_from_checkpoint:
            first_epoch, global_step, resume_step = self._setup_checkpoint_resumption()
        else:
            resume_step = 0
        
        # Initialize progress bar for training tracking
        progress_bar = tqdm(
            range(global_step, self.args.max_train_steps),
            disable=not self.accelerator.is_local_main_process,
            desc="Training Steps"
        )
        
        # Store original embeddings as reference for regularization
        orig_embeds_params = self._get_original_embeddings()
        
        # Setup attention controller for spatial guidance
        self.controller = AttentionStore()
        self.register_attention_control(self.controller)
        
        # Configure timestep sampling distribution if using the t_dist parameter
        prob_t_weights = self._setup_timestep_sampling()

        # Main training loop
        for epoch in range(first_epoch, self.args.num_train_epochs):
            self.unet.train()
            if self.args.train_text_encoder:
                self.text_encoder.train()

            for step, batch in enumerate(self.train_dataloader):
                # Skip steps when resuming from checkpoint
                if self._should_skip_step(epoch, step, first_epoch, resume_step):
                    if step % self.args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue

                # Step transition handling - from Step 1 to Step 2
                if self.args.phase1_train_steps == global_step:
                    # Reconfigure model for step 2: enable UNet training and update optimizer
                    self.optimizer, self.lr_scheduler = self._transition_to_concept_refinement()

                logs = {}

                # Main training step with gradient accumulation
                with self.accelerator.accumulate(self.unet):
                    # Process batch and compute loss
                    loss, logs = self._training_step(batch, global_step, prob_t_weights, logs)
                    
                    # Backward pass and optimization
                    self.accelerator.backward(loss)
                    
                    # Clear attention store after use
                    # print("before")
                    # print(self.controller.attention_store)
                    self.controller.attention_store = {}
                    self.controller.cur_step = 0
                    # print("after")
                    # print(self.controller.attention_store)
                    
                    
                    # Gradient clipping and optimization step
                    if self.accelerator.sync_gradients:
                        self._clip_gradients()
                    
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=self.args.set_grads_to_none)
                    
                    # Preserve original embeddings during phase 1
                    if global_step < self.args.phase1_train_steps:
                        self._preserve_original_embeddings(global_step, orig_embeds_params)

                # Update global step and handle checkpoints
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    
                    # Save checkpoint at specified intervals
                    if global_step % self.args.checkpointing_steps == 0:
                        self._save_checkpoint(global_step)
                
                # Update logging information
                logs["loss"] = loss.detach().item()
                logs["lr"] = self.lr_scheduler.get_last_lr()[0]
                progress_bar.set_postfix(**logs)
                self.accelerator.log(logs, step=global_step)
                
                # Break if we've reached max steps
                if global_step >= self.args.max_train_steps:
                    break

            # Break epoch loop if we've reached max steps
            if global_step >= self.args.max_train_steps:
                break

        # Save final model and end training
        self.save_pipeline(self.args.output_dir)
        self.accelerator.end_training()

    def _setup_checkpoint_resumption(self):
        """Setup training resumption from checkpoint"""
        if self.args.resume_from_checkpoint != "latest":
            path = os.path.basename(self.args.resume_from_checkpoint)
        else:
            # Find the most recent checkpoint
            dirs = os.listdir(self.args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            self.accelerator.print(
                f"Checkpoint '{self.args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            self.args.resume_from_checkpoint = None
            return 0, 0, 0
        else:
            self.accelerator.print(f"Resuming from checkpoint {path}")
            self.accelerator.load_state(os.path.join(self.args.output_dir, path))
            global_step = int(path.split("-")[1])
            resume_global_step = global_step * self.args.gradient_accumulation_steps
            num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.args.gradient_accumulation_steps)
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * self.args.gradient_accumulation_steps)
            return first_epoch, global_step, resume_step

    def _get_original_embeddings(self):
        """Get a copy of the current text encoder embeddings"""
        return (
            self.accelerator.unwrap_model(self.text_encoder)
            .get_input_embeddings()
            .weight.data.clone()
        )

    def _setup_timestep_sampling(self):
        """Setup timestep sampling weights if t_dist is enabled"""
        if self.args.t_dist > 0.0:
            def weight_function(x):
                return (1 / self.noise_scheduler.config.num_train_timesteps) * (
                    1 - self.args.t_dist * np.cos(np.pi * x / self.noise_scheduler.config.num_train_timesteps)
                )
            return [weight_function(t_) for t_ in np.arange(self.noise_scheduler.config.num_train_timesteps)]
        return None

    def _should_skip_step(self, epoch, step, first_epoch, resume_step):
        """Determine if current step should be skipped when resuming"""
        return (
            self.args.resume_from_checkpoint and 
            epoch == first_epoch and 
            step < resume_step
        )

    def _transition_to_concept_refinement(self):
        """Update model configuration for transition to concept refinement"""
        # Enable gradients for UNet
        self.unet.requires_grad_(True)
        if self.args.train_text_encoder:
            self.text_encoder.requires_grad_(True)
        
        # Configure parameters to optimize
        unet_params = self.unet.parameters()
        if self.args.train_text_encoder:
            params_to_optimize = itertools.chain(unet_params, self.text_encoder.parameters())
        else:
            params_to_optimize = itertools.chain(
                unet_params, self.text_encoder.get_input_embeddings().parameters()
            )
        
        # Create new optimizer
        if self.args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )
            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW
            
        self.optimizer = optimizer_class(
            params_to_optimize,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay,
            eps=self.args.adam_epsilon,
        )
        
        # Create new scheduler
        self.lr_scheduler = get_scheduler(
            self.args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.lr_warmup_steps * self.args.gradient_accumulation_steps,
            num_training_steps=self.args.max_train_steps * self.args.gradient_accumulation_steps,
            num_cycles=self.args.lr_num_cycles,
            power=self.args.lr_power,
        )
        
        # Prepare with accelerator
        return self.accelerator.prepare(self.optimizer, self.lr_scheduler)

    def _training_step(self, batch, global_step, prob_t_weights, logs):
        """Execute a single training step with loss computation"""
        # Convert images to latents
        latents = self.vae.encode(
            batch["pixel_values"].to(dtype=self.weight_dtype)
        ).latent_dist.sample()
        latents = latents * 0.18215

        # Sample noise for diffusion process
        noise = torch.randn_like(latents)
        if self.args.noise_offset:
            # Add offset noise (https://www.crosslabs.org//blog/diffusion-with-offset-noise)
            noise += self.args.noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
            )

        # Sample timesteps - either with importance sampling or uniform
        bsz = latents.shape[0]
        if self.args.t_dist and prob_t_weights: 
            # ReVersion-style importance sampling
            timesteps = torch.tensor(
                list(WeightedRandomSampler(prob_t_weights, bsz, replacement=True)), 
                device=latents.device
            )
        else:
            # Uniform sampling
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
            )
        timesteps = timesteps.long()

        # Add noise to latents according to noise schedule
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Prepare prompt based on training phase
        token_ids_to_use = batch["token_ids"].item()
        tokens_to_use = self.placeholder_tokens[token_ids_to_use]
        prompt = random.choice(prompt_template).format(tokens_to_use)
        
        # Get text embeddings for conditioning
        text_inputs = tokenize_prompt(self.tokenizer, [prompt])
        encoder_hidden_states = encode_prompt(self.text_encoder, text_inputs.input_ids)
        
        # Handle class conditioning if using prior preservation
        if self.args.with_prior_preservation:
            encoder_class_hidden_states = encode_prompt(
                self.text_encoder, batch["class_ids"][0]
            )
            encoder_hidden_states = torch.cat(
                [encoder_class_hidden_states, encoder_hidden_states], dim=0
            )

        # Get noise prediction from diffusion model
        model_pred = self.unet(
            noisy_latents, timesteps, encoder_hidden_states
        ).sample

        # Get the target for loss computation
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        # Compute the main diffusion loss
        loss = self._compute_diffusion_loss(model_pred, target, batch, self.args.with_prior_preservation)

        # Compute attention-based spatial guidance loss if enabled
        if self.args.lambda_attention != 0:
            attn_loss = self._compute_attention_loss(batch, text_inputs, global_step)
            loss += attn_loss
            logs["attn_loss"] = attn_loss.detach().item()
            
        return loss, logs

    def _compute_diffusion_loss(self, model_pred, target, batch, with_prior_preservation):
        """Compute the main diffusion loss, optionally with prior preservation"""
        if with_prior_preservation:
            # Split prediction and target for instance and class samples
            model_pred_prior, model_pred = torch.chunk(model_pred, 2, dim=0)
            target_prior, target = torch.chunk(target, 2, dim=0)

            # Apply mask if using masked loss
            if self.args.apply_masked_loss:
                max_masks = torch.max(batch["instance_masks"], axis=1).values
                downsampled_mask = F.interpolate(input=max_masks, size=(64, 64))
                model_pred = model_pred * downsampled_mask
                target = target * downsampled_mask

            # Compute loss for instance and class samples
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
            return loss + self.args.prior_loss_weight * prior_loss
        else:
            # Apply mask if using masked loss
            if self.args.apply_masked_loss:
                max_masks = torch.max(batch["instance_masks"], axis=1).values
                downsampled_mask = F.interpolate(input=max_masks, size=(64, 64))
                model_pred = model_pred * downsampled_mask
                target = target * downsampled_mask
            return F.mse_loss(model_pred.float(), target.float(), reduction="mean")

    def _compute_attention_loss(self, batch, text_inputs, global_step):
        """
        Compute attention-based spatial guidance loss
        Encourages the model to attend to correct spatial regions for each token
        """
        attn_loss = 0.0
        for batch_idx in range(self.args.train_batch_size):
            # Get ground truth segmentation masks and resize to attention map size
            GT_masks = F.interpolate(input=batch["instance_masks"][batch_idx], size=(16, 16))
            
            # Aggregate cross-attention maps across layers
            agg_attn = self.aggregate_attention(
                res=16,
                from_where=("up", "down"),
                is_cross=True,
                select=batch_idx,
            )

            for mask_id in range(len(GT_masks)):
                curr_token_id = batch["token_ids"][batch_idx][mask_id]
                
                # Get attention map for concept-specific token
                curr_placeholder_token_id = self.placeholder_token_ids["asset"][curr_token_id]
                asset_idx = (text_inputs.input_ids[0] == curr_placeholder_token_id).nonzero().item()
                asset_attn_mask = agg_attn[..., asset_idx]          

                # Calculate Wasserstein loss between attention and ground truth mask
                attn_loss += wasser_loss(GT_masks[mask_id, 0].float(), asset_attn_mask.float())

        return self.args.lambda_attention * (attn_loss / self.args.train_batch_size)

    def _clip_gradients(self):
        """Clip gradients to prevent exploding gradients"""
        if self.args.train_text_encoder:
            params_to_clip = itertools.chain(self.unet.parameters(), self.text_encoder.parameters())
        else:
            params_to_clip = self.unet.parameters()
        self.accelerator.clip_grad_norm_(params_to_clip, self.args.max_grad_norm)

    def _preserve_original_embeddings(self, global_step, orig_embeds_params):
        """
        Preserve original embeddings for non-trainable tokens during phase 1
        This prevents catastrophic forgetting of the base model's knowledge
        """
        with torch.no_grad():
            # Preserve all except asset tokens
            num_special_tokens = self.num_of_assets
            self.accelerator.unwrap_model(self.text_encoder).get_input_embeddings().weight.data[
                :-num_special_tokens
            ] = orig_embeds_params[:-num_special_tokens]

    def _save_checkpoint(self, global_step):
        """Save a checkpoint at the current training state"""
        if self.accelerator.is_main_process:
            save_path = os.path.join(self.args.output_dir, f"checkpoint-{global_step}")
            self.accelerator.save_state(save_path)
            logger.info(f"Saved state to {save_path}")

    def register_attention_control(self, controller):
        attn_procs = {}
        cross_att_count = 0
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else self.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[
                    block_id
                ]
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
                place_in_unet = "down"
            else:
                continue
            cross_att_count += 1
            attn_procs[name] = P2PCrossAttnProcessor(
                controller=controller, place_in_unet=place_in_unet
            )

        self.unet.set_attn_processor(attn_procs)
        controller.num_att_layers = cross_att_count
    
    def get_average_attention(self):
        average_attention = {
            key: [
                item / self.controller.cur_step
                for item in self.controller.attention_store[key]
            ]
            for key in self.controller.attention_store
        }
        return average_attention

    def aggregate_attention(self, res: int, from_where: List[str], is_cross: bool, select: int):
        out = []
        attention_maps = self.get_average_attention()
        num_pixels = res**2
        for location in from_where:
            for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                if item.shape[1] == num_pixels:
                    cross_maps = item.reshape(
                        self.args.train_batch_size, -1, res, res, item.shape[-1]
                    )[select]
                    out.append(cross_maps)
        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        return out

    def save_pipeline(self, path):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            pipeline = DiffusionPipeline.from_pretrained(
                self.args.pretrained_model_name_or_path,
                unet=self.accelerator.unwrap_model(self.unet),
                text_encoder=self.accelerator.unwrap_model(self.text_encoder),
                tokenizer=self.tokenizer,
                revision=self.args.revision,
            )
            pipeline.save_pretrained(path)


def main():    
    '''Initialize concept extraction framework...'''
    ce = ConceptExtraction()
    
    '''Start training...'''
    try: ce.train()
    except Exception as e: logger.error(f"An error occurred: {e}")
    finally: torch.cuda.empty_cache()
            
if __name__ == "__main__":
    main()