import os
import argparse

from diffusers import DiffusionPipeline, DDIMScheduler
import torch


class inference_benchmark:
    def __init__(self):
        self._parse_args()
        self._load_pipeline()

    def _parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset_path", type=str, required=True)
        parser.add_argument("--model_path", type=str, required=True)
        parser.add_argument("--output_path", type=str, required=True)
        parser.add_argument("--device", type=str, default="cuda")
        parser.add_argument("--seed", type=int, default=20)
        self.args = parser.parse_args()

    def _load_pipeline(self):
        self.pipeline = DiffusionPipeline.from_pretrained(
            self.args.model_path,
            torch_dtype=torch.float16,
        )
        self.pipeline.scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        self.pipeline.to(self.args.device)

    def prepare_prompt(self):
        dataset_path = self.args.dataset_path
        mask_paths = [f for f in os.listdir(dataset_path) if f.startswith("mask")]

        number_of_masks = len(mask_paths)

        prompts = {
            'identity':[],
            'composition':[],
        }

        for i in range(number_of_masks):
            # Indentity
            prompts['identity'].append(f"a photo of <*asset{i}>")

        # composition --> combine all identity prompts into one prompt
        iden = [f"<asset{i}>" for i in range(number_of_masks)]
        prompts['composition'].append("a photo of " + " and ".join(iden))

        return prompts

    @torch.no_grad()
    def run_and_save(self):

        id_path = os.path.split(self.args.model_path)[1]
        # mkdir "assets"
        os.makedirs(self.args.output_path + f"/{id_path}", exist_ok=True)
        g = torch.Generator(device="cuda")
        g.manual_seed(self.args.seed)

        # make dirs for each prompt
        prompts = self.prepare_prompt()
        for key, prompt in prompts.items():
            os.makedirs(self.args.output_path + f"/{id_path}/{key}", exist_ok=True)
            for p in prompt:
                if key != 'composition':
                    # Find the last occurrence of '>'
                    last_angle_bracket = p.rfind('>')
                    asset_id = p[last_angle_bracket-1]
                    # check if the dir exist
                    if not os.path.exists(self.args.output_path + f"/{id_path}/{key}/{asset_id}"):
                        os.makedirs(self.args.output_path + f"/{id_path}/{key}/{asset_id}", exist_ok=True)

        # Generate and save images
        for key, prompt in prompts.items():
            for p in prompt:
                images = [self.pipeline([p], generator=g).images[0] for _ in range(8)]
                # images = [self.pipeline([p]).images[0] for _ in range(8)]

                # Save the images
                if key == 'composition':
                    for i, image in enumerate(images):
                        image.save(self.args.output_path + f"/{id_path}/{key}/{i}_{p}.png")
                else:
                    last_angle_bracket = p.rfind('>')
                    asset_id = p[last_angle_bracket-1]
                    for i, image in enumerate(images):
                        image.save(self.args.output_path + f"/{id_path}/{key}/{asset_id}/{i}_{p}.png")


if __name__ == "__main__":
    try:
        infer = inference_benchmark()
        infer.run_and_save()
    except Exception as e:
        print(e)
        raise e
    