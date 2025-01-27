import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import torch
import numpy as np
from PIL import Image
from collections import OrderedDict
from object_replacer.src import models
from object_replacer.src.methods import rasg, sd
from object_replacer.src.utils import IImage, poisson_blend
import warnings
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=SyntaxWarning)
from torch.cuda.amp import autocast  

def get_inpainting_function(
    model_id: str,
    method: str,
    negative_prompt: str = '',
    positive_prompt: str = '',
    num_steps: int = 50,
    eta: float = 0.25,
    guidance_scale: float = 7.5
):
    inp_model = models.load_inpainting_model(model_id, device='cuda:0', cache=True)
    
    if 'rasg' in method:
        runner = rasg
    else:
        runner = sd
    
    def run(image: Image, mask: Image, prompt: str, seed: int = 1) -> Image:
        # Inference Mode
        with torch.inference_mode():
            # Autocast for mixed precision
            with autocast():
                inpainted_image = runner.run(
                    ddim=inp_model,
                    method=method,
                    prompt=prompt,
                    image=IImage(image),
                    mask=IImage(mask),
                    seed=seed,
                    eta=eta,
                    negative_prompt=negative_prompt,
                    positive_prompt=positive_prompt,
                    num_steps=num_steps,
                    guidance_scale=guidance_scale
                ).pil()

        # Resize the inpainted image to match the original size
        w, h = image.size
        inpainted_image = Image.fromarray(np.array(inpainted_image)[:h, :w])
        return inpainted_image
    return run


def get_inpainting_sr_function(
    positive_prompt='high resolution professional photo',
    negative_prompt='',
    noise_level=20,
    use_sam_mask=False,
    blend_trick=True,
    blend_output=True
):
    sr_model = models.sd2_sr.load_model(device='cuda:0')
    sam_predictor = None
    if use_sam_mask:
        sam_predictor = models.sam.load_model(device='cuda:0')

    def run(inpainted_image: Image, image: Image, mask: Image, prompt: str, seed: int = 1) -> Image:
        # Inference Mode
        with torch.inference_mode():
            # Autocast for mixed precision
            with autocast():
                return sr.run(
                    sr_model,
                    sam_predictor,
                    inpainted_image,
                    image,
                    mask,
                    prompt=f'{prompt}, {positive_prompt}',
                    noise_level=noise_level,
                    blend_trick=blend_trick,
                    blend_output=blend_output,
                    negative_prompt=negative_prompt, 
                    seed=seed,
                    use_sam_mask=use_sam_mask
                )
    return run
