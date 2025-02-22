import gradio as gr
import torch
from PIL import Image
from collections import OrderedDict
from object_replacer.src import models
from object_replacer.src.methods import rasg, sd
from object_replacer.src.utils import IImage, poisson_blend
import warnings
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=SyntaxWarning)

negative_prompt_str = "text, bad anatomy, bad proportions, blurry, cropped, deformed, disfigured, duplicate, error, extra limbs, gross proportions, jpeg artifacts, long neck, low quality, lowres, malformed, morbid, mutated, mutilated, out of frame, ugly, worst quality"
positive_prompt_str = "Full HD, high quality, high resolution"


models.pre_download_inpainting_models()

inpainting_models = OrderedDict([
    ("Dreamshaper Inpainting V8", 'ds8_inp'),
    ("Stable-Inpainting 2.0", 'sd2_inp'),
    ("Stable-Inpainting 1.5", 'sd15_inp')
])
inp_model = models.load_inpainting_model('sd2_inp', device='cuda', cache=True)


def inpainting_run(use_rasg, use_painta, prompt, imageMask,
    hr_image, negative_prompt, positive_prompt, seed=49123, eta=0.1, ddim_steps=50,
    guidance_scale=7.5, batch_size=1
):
    torch.cuda.empty_cache()

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
    
    return blended_images



def process_inpainting(prompt, image_mask):
    try:
        input_image = IImage(image_mask["image"]).resize(512)
        input_mask = IImage(image_mask["mask"]).resize(512).rgb()

        output_images = inpainting_run(
            use_rasg=True,
            use_painta=True,
            prompt=prompt,
            imageMask={"mask": input_mask.numpy()},
            hr_image=input_image.numpy(),
            negative_prompt=negative_prompt_str,
            positive_prompt=positive_prompt_str,
        )

        return output_images[0]
    except Exception as e:
        return f"Error: {str(e)}"


with gr.Blocks(css="""
.output-image img {
    display: block;
    margin: auto; /* Pusatkan gambar secara horizontal */
    max-width: 100%;
    max-height: 100%;
    object-fit: contain; /* Jaga rasio aspek tanpa pemotongan */
}
.output-image {
    display: flex;
    align-items: center; /* Pusatkan secara vertikal */
    justify-content: center; /* Pusatkan secara horizontal */
    height: 100%; /* Pastikan kontainer memiliki tinggi penuh */
    width: 100%; /* Pastikan kontainer memiliki lebar penuh */
}
""") as demo:
    gr.HTML(
        """
        <div style="text-align: center; max-width: 1200px; margin: 20px auto;">
        <h1 style="font-weight: 900; font-size: 2rem; margin-bottom: 0.5rem">
            AI Generative Fill
        </h1>
        """
    )
    
    with gr.Row():
        with gr.Column():
            image_mask = gr.Image(label="Upload Image & Draw Mask", tool="sketch", type="pil")
            prompt = gr.Textbox(label="Prompt", placeholder="Describe the image you want to generate.")
            run_button = gr.Button("Generate")
        
        with gr.Column():
            output_image = gr.Image(label="Generated Image", type="pil", elem_classes=["output-image"])

    run_button.click(
        process_inpainting,
        inputs=[prompt, image_mask],
        outputs=output_image,
    )

demo.queue().launch(debug=True)
