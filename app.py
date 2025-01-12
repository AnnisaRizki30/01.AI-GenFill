import os
import sys
from pathlib import Path
from collections import OrderedDict

import gradio as gr
import shutil
import uuid
import torch
from PIL import Image

demo_path = Path(__file__).resolve().parent
root_path = demo_path
sys.path.append(str(root_path))
from src import models
from src.methods import rasg, sd, sr
from src.utils import IImage, poisson_blend, image_from_url_text


TMP_DIR = root_path / 'gradio_tmp'
if TMP_DIR.exists():
    shutil.rmtree(str(TMP_DIR))
TMP_DIR.mkdir(exist_ok=True, parents=True)

os.environ['GRADIO_TEMP_DIR'] = str(TMP_DIR)


negative_prompt_str = "text, bad anatomy, bad proportions, blurry, cropped, deformed, disfigured, duplicate, error, extra limbs, gross proportions, jpeg artifacts, long neck, low quality, lowres, malformed, morbid, mutated, mutilated, out of frame, ugly, worst quality"
positive_prompt_str = "Full HD, high quality, high resolution"

# Load models
models.pre_download_inpainting_models()
inpainting_models = OrderedDict([
    ("Dreamshaper Inpainting V8", 'ds8_inp'),
    ("Stable-Inpainting 2.0", 'sd2_inp'),
    ("Stable-Inpainting 1.5", 'sd15_inp')
])
sr_model = models.sd2_sr.load_model(device='cuda')
sam_predictor = models.sam.load_model(device='cuda')

inp_model = models.load_inpainting_model("Stable-Inpainting 2.0", device='cuda', cache=True)

use_painta = True  # Set default value for use_painta
use_rasg = True  # Set default value for use_rasg



def set_model_from_name(new_inp_model_name):
    global inp_model
    global inp_model_name
    if new_inp_model_name != inp_model_name:
        print (f"Activating Inpaintng Model: {new_inp_model_name}")
        inp_model = models.load_inpainting_model(
            inpainting_models[new_inp_model_name], device='cuda', cache=True)
        inp_model_name = new_inp_model_name


def save_user_session(hr_image, hr_mask, lr_results, prompt, session_id=None):
    if session_id == '':
        session_id = str(uuid.uuid4())
    
    session_dir = TMP_DIR / session_id
    session_dir.mkdir(exist_ok=True, parents=True)
    
    hr_image.save(session_dir / 'hr_image.png')
    hr_mask.save(session_dir / 'hr_mask.png')

    lr_results_dir = session_dir / 'lr_results'
    if lr_results_dir.exists():
        shutil.rmtree(lr_results_dir)
    lr_results_dir.mkdir(parents=True)
    for i, lr_result in enumerate(lr_results):
        lr_result.save(lr_results_dir / f'{i}.png')

    with open(session_dir / 'prompt.txt', 'w') as f:
        f.write(prompt)
    
    return session_id


def recover_user_session(session_id):
    if session_id == '':
        return None, None, [], ''
    
    session_dir = TMP_DIR / session_id
    lr_results_dir = session_dir / 'lr_results'

    hr_image = Image.open(session_dir / 'hr_image.png')
    hr_mask = Image.open(session_dir / 'hr_mask.png')
  
    lr_result_paths = list(lr_results_dir.glob('*.png'))
    gallery = []
    for lr_result_path in sorted(lr_result_paths):
        gallery.append(Image.open(lr_result_path))

    with open(session_dir / 'prompt.txt', "r") as f:
        prompt = f.read()

    return hr_image, hr_mask, gallery, prompt


def set_model_from_name(new_inp_model_name):
    global inp_model
    global inp_model_name
    if new_inp_model_name != inp_model_name:
        print(f"Activating Inpainting Model: {new_inp_model_name}")
        inp_model = models.load_inpainting_model(
            inpainting_models[new_inp_model_name], device='cuda:0', cache=True)
        inp_model_name = new_inp_model_name


def inpainting_run(model_name, use_rasg, use_painta, prompt, imageMask,
    hr_image, seed, eta, negative_prompt, positive_prompt, ddim_steps,
    guidance_scale=7.5, batch_size=1, session_id=''
):
    torch.cuda.empty_cache()
    set_model_from_name(model_name)

    method = ['default']
    if use_painta: method.append('painta')
    if use_rasg: method.append('rasg')
    method = '-'.join(method)

    if use_rasg:
        inpainting_f = rasg.run
    else:
        inpainting_f = sd.run

    seed = int(seed)
    batch_size = max(1, min(int(batch_size), 4))

    image = IImage(hr_image).resize(512)
    mask = IImage(imageMask['mask']).rgb().resize(512)

    inpainted_images = []
    blended_images = []
    for i in range(batch_size):
        seed = seed + i * 1000

        inpainted_image = inpainting_f(
            ddim=inp_model,
            method=method,
            prompt=prompt,
            image=image,
            mask=mask,
            seed=seed,
            eta=eta,
            negative_prompt=negative_prompt,
            positive_prompt=positive_prompt,
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

    session_id = save_user_session(
        hr_image, imageMask['mask'], inpainted_images, prompt, session_id=session_id)
    
    return blended_images, session_id


with gr.Blocks(css=demo_path / 'style.css') as demo:
    gr.HTML(
        """
        <div style="text-align: center; max-width: 1200px; margin: 20px auto;">
        <h1 style="font-weight: 900; font-size: 3rem; margin-bottom: 0.5rem">
            AI Generative Fill
        </h1>
        """
    )

    prompt = gr.Textbox(label = "Inpainting Prompt")
    
    with gr.Row():
        with gr.Column():
            imageMask = gr.ImageMask(label="Input Image", brush_color='#ff0000', elem_id="inputmask", type="pil")
            hr_image = gr.Image(visible=False, type="pil")
            hr_image.change(fn=None, _js="function() {setTimeout(imageMaskResize, 200);}", inputs=[], outputs=[])
            imageMask.upload(
                fn=None,
                _js="async function (a) {hr_img = await resize_b64_img(a['image'], 2048); dp_img = await resize_b64_img(hr_img, 1024); return [hr_img, {image: dp_img, mask: null}]}",
                inputs=[imageMask],
                outputs=[hr_image, imageMask],
            )
            inpaint_btn = gr.Button("Inpaint", scale = 0)

        with gr.Column():
            output_gallery = gr.Gallery(
                [],
                columns = 4,
                preview = True,
                allow_preview = True,
                object_fit='scale-down',
                elem_id='outputgallery'
            )
            
    session_id = gr.Textbox(value='', visible=False)

    inpaint_btn.click(
        fn=inpainting_run, 
        inputs=[
            "Stable-Inpainting 2.0",  
            use_rasg,                 
            use_painta,               
            prompt,
            imageMask,
            hr_image,
            # seed,
            # eta,
            # negative_prompt,
            # positive_prompt,
            # ddim_steps,
            # guidance_scale,
            # batch_size,
            session_id
        ], 
        outputs=[output_gallery, session_id], 
        api_name="inpaint"
    )

demo.queue(max_size=20)
demo.launch(share=True, allowed_paths=[str(TMP_DIR)])
