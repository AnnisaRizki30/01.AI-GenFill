import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import torch
import numpy as np
import PIL.Image
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

negative_prompt_str = "text, bad anatomy, bad proportions, blurry, cropped, deformed, disfigured, duplicate, error, extra limbs, gross proportions, jpeg artifacts, long neck, low quality, lowres, malformed, morbid, mutated, mutilated, out of frame, ugly, worst quality"
positive_prompt_str = "Full HD, 4K, high quality, high resolution"

models.pre_download_inpainting_models()
inpainting_models = OrderedDict([
    #("Dreamshaper Inpainting V8", 'ds8_inp'),
    ("Stable-Inpainting 2.0", 'sd2_inp'),
    #("Stable-Inpainting 1.5", 'sd15_inp')
])
#sr_model = models.sd2_sr.load_model(device='cuda:1')
sam_predictor = models.sam.load_model(device='cuda:0')

inp_model_name = list(inpainting_models.keys())[0]
inp_model = models.load_inpainting_model(
    inpainting_models[inp_model_name], device='cuda:0', cache=True)

def add_channel_and_batch_size(mask):
    if isinstance(mask, Image.Image):
        mask = np.array(mask)

    if mask.ndim == 2:
        # Menambahkan dimensi batch size dan channel
        mask = mask[None, None, :, :]  # Menjadi (1, 1, H, W)
    elif mask.ndim == 3:
        # Jika dimensi sudah (H, W, C), tambahkan batch size
        mask = mask[None, ...]  # Menjadi (1, H, W, C)
    elif mask.ndim == 4:
        # Dimensi sudah sesuai (B, C, H, W)
        pass
    else:
        raise ValueError(f"Unsupported number of dimensions: {mask.ndim}")
    return mask


def resize_image(image, size):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((size, size))
    return np.array(resized_image)


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



def inpainting_run(model_name, use_rasg, use_painta, prompt, imageMask,
                   hr_image, seed, eta, negative_prompt, positive_prompt, ddim_steps,
                   guidance_scale=7.5, batch_size=1):
    torch.cuda.empty_cache()
    # set_model_from_name(model_name)

    method = ['default']
    if use_painta: method.append('painta')
    if use_rasg: method.append('rasg')
    method = '-'.join(method)

    if use_rasg:
        inpainting_f = rasg_run
    else:
        inpainting_f = sd_run

    seed = int(seed)
    batch_size = max(1, min(int(batch_size), 4))

    image = IImage(hr_image).resize(512)
    mask = IImage(imageMask['mask']).rgb().resize(512)

    inpainted_images = []
    blended_images = []
    with torch.inference_mode():
        for i in range(batch_size):
            seed = seed + i * 1000
            with autocast():
                inpainted_image = inpainting_f(
                    ddim=inp_model,
                    method=method,
                    prompt=prompt,
                    image=image,
                    mask=mask,
                    seed=seed,
                    eta=eta,
                    negative_prompt=negative_prompt_str,
                    positive_prompt=positive_prompt_str,
                    num_steps=ddim_steps,
                    guidance_scale=guidance_scale
                ).crop(image.size)
            blended_image = poisson_blend(
                orig_img=image.data[0],
                fake_img=inpainted_image.data[0],
                mask=mask.data[0],
                dilation=12
            )
            blended_images.append(blended_image)
            inpainted_images.append(inpainted_image.pil())
    return blended_images
