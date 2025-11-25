import torch
from transformers import CLIPImageProcessor, CLIPVisionModel
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.utils import load_image, make_image_grid
import face_alignment
from PIL import Image
import numpy as np
import os

from src.diffusers.models.referencenet.referencenet_unet_2d_condition import ReferenceNetModel
from src.diffusers.models.referencenet.unet_2d_condition import UNet2DConditionModel
from src.diffusers.pipelines.referencenet.pipeline_referencenet import StableDiffusionReferenceNetPipeline

from utils.segmented_anonymization import anonymize_faces_segmented

def main():
    print("Loading models...")
    face_model_id = "hkung/face-anon-simple"
    clip_model_id = "openai/clip-vit-large-patch14"
    sd_model_id = "stabilityai/stable-diffusion-2-1"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Using device: {device}")

    try:
        unet = UNet2DConditionModel.from_pretrained(face_model_id, subfolder="unet", use_safetensors=True)
        referencenet = ReferenceNetModel.from_pretrained(face_model_id, subfolder="referencenet", use_safetensors=True)
        conditioning_referencenet = ReferenceNetModel.from_pretrained(face_model_id, subfolder="conditioning_referencenet", use_safetensors=True)
        vae = AutoencoderKL.from_pretrained(sd_model_id, subfolder="vae", use_safetensors=True)
        scheduler = DDPMScheduler.from_pretrained(sd_model_id, subfolder="scheduler", use_safetensors=True)
        feature_extractor = CLIPImageProcessor.from_pretrained(clip_model_id, use_safetensors=True)
        image_encoder = CLIPVisionModel.from_pretrained(clip_model_id, use_safetensors=True)

        pipe = StableDiffusionReferenceNetPipeline(
            unet=unet,
            referencenet=referencenet,
            conditioning_referencenet=conditioning_referencenet,
            vae=vae,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            scheduler=scheduler,
        )
        pipe = pipe.to(device, dtype=dtype)
    except Exception as e:
        print(f"Error loading diffusion models: {e}")
        print("Continuing with mock pipe for testing other operators if possible...")
        pipe = None

    print("Initializing Face Alignment...")
    # Force CPU for face alignment if CUDA fails or just to be safe for this test script if VRAM is tight
    # But user has CUDA likely.
    try:
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, face_detector="sfd", device=device)
    except Exception as e:
        print(f"Error initializing FaceAlignment: {e}")
        return

    # Define masks
    mask1 = ['eyes', 'mouth', 'nostrils']
    mask2 = mask1 + ['eyebrows']
    mask3 = mask2 + ['lips', 'teeth']

    masks = {
        "Mask_1": mask1,
        "Mask_2": mask2,
        "Mask_3": mask3
    }

    operators = ['blur', 'mosaic', 'diffusion']

    # Load image
    image_path = "my_dataset/test/14795.png"
    if not os.path.exists(image_path):
        print(f"Image not found at {image_path}")
        # Try finding any image in my_dataset/test
        test_dir = "my_dataset/test"
        if os.path.exists(test_dir):
            files = os.listdir(test_dir)
            if files:
                image_path = os.path.join(test_dir, files[0])
                print(f"Using alternative image: {image_path}")
            else:
                print("No images found in test dir.")
                return
        else:
            print("Test dir not found.")
            return

    original_image = load_image(image_path)
    
    results = []
    
    generator = torch.manual_seed(42)

    print("Starting ablation...")
    for mask_name, features in masks.items():
        for op in operators:
            print(f"Processing {mask_name} with {op}...")
            
            if op == 'diffusion' and pipe is None:
                print("Skipping diffusion due to pipe load error.")
                continue

            try:
                anon_image = anonymize_faces_segmented(
                    image=original_image,
                    face_alignment_model=fa,
                    mask_features=features,
                    operator_type=op,
                    pipe=pipe,
                    generator=generator,
                    kernel_size=(31, 31),
                    block_size=15,
                    num_inference_steps=20, # Low steps for quick test
                    guidance_scale=4.0,
                    anonymization_degree=1.25
                )
                results.append(anon_image)
                
                # Save individual result
                output_filename = f"ablation_{mask_name}_{op}.png"
                anon_image.save(output_filename)
                print(f"Saved {output_filename}")
                
            except Exception as e:
                print(f"Error processing {mask_name} with {op}: {e}")
                import traceback
                traceback.print_exc()

    print("Done.")

if __name__ == "__main__":
    main()
