# -*- coding: utf-8 -*-

import sys
import io
import requests
import json
import base64
from PIL import Image
import numpy as np
import gradio as gr
import cv2
import tempfile

def inference_mask1(prompt,
              img,
              img_):
    files = {
        "pimage" : resizeImg(prompt["image"]),
        "pmask" : resizeImg(prompt["mask"]),
        "img" : resizeImg(img),
        "img_" : resizeImg(img_)
    }
    r = requests.post("http://120.92.79.209/painter/run", json = files)
    a = json.loads(r.text)
    res = []
    for i in range(len(a)):
        #out = Image.open(io.BytesIO(base64.b64decode(a[i])))
        #out = out.resize((224, 224))
        #res.append(np.uint8(np.array(out)))
        res.append(np.uint8(np.array(Image.open(io.BytesIO(base64.b64decode(a[i]))))))
    return res

def resizeImg(img):
    res, hres = 448, 448
    img = Image.fromarray(img).convert("RGB")
    img = img.resize((res, hres))
    temp = io.BytesIO()
    img.save(temp, format="WEBP")
    return base64.b64encode(temp.getvalue()).decode('ascii')


def inference_mask_video(
              prompt,
              vid,
              request: gr.Request,
              ):
    res, hres = 448, 448

    # load the shared prompt image pair
    img2 = Image.fromarray(prompt['image']).convert("RGB")
    img2 = img2.resize((res, hres))
    img2 = np.array(img2) / 255.

    tgt2 = Image.fromarray(prompt['mask']).convert("RGB")
    prompt_label_vis = np.array(tgt2).astype(np.uint8)

    tgt2 = tgt2.resize((res, hres), Image.NEAREST)
    tgt2 = np.array(tgt2) / 255.
    tgt = tgt2  # tgt is not available
    tgt = np.concatenate((tgt2, tgt), axis=0)

    assert tgt.shape == (2*res, res, 3)
    # normalize by ImageNet mean and std
    tgt = tgt - imagenet_mean
    tgt = tgt / imagenet_std
    tgt = tgt[None, ...]


    vidcap = cv2.VideoCapture(vid)
    success,image = vidcap.read()
    count = 0
    imgs = [image,]
    while success:
        success,image = vidcap.read()
        imgs.append(image)
        #print('Read a new frame: ', success)
        count += 1
    imgs = imgs[:-1]
    # max 16 frames
    imgs = imgs[:16]
    imgs = [Image.fromarray(img[:,:,::-1]) for img in imgs]

    output_list = []
    for img in imgs:
        ori_img = np.array(img)
        img = img.convert("RGB")
        size = img.size
        img = img.resize((res, hres))
        img = np.array(img) / 255.

        img = np.concatenate((img2, img), axis=0)
        assert img.shape == (2*res, res, 3)
        img = img - imagenet_mean
        img = img / imagenet_std

        img = img[None, ...]

        torch.manual_seed(2)
        output_image = run_one_image(img, tgt, size, painter, device)
        output_vis = (ori_img * 0.5 + output_image * 0.5).astype(np.uint8)
        #output_vis_pred = output_image.astype(np.uint8)
        output_list.append(output_vis)
        #output_list.append(output_vis_pred)
    output_list = [Image.fromarray(out).convert("RGB") for out in output_list]

    #file_out = tempfile.NamedTemporaryFile(suffix='.gif')
    file_out = 'out.gif'
    output_list[0].save(file_out, save_all=True, append_images=output_list[1:])
    return file_out


# define app features and run

examples = [
            ['./images/hmbb_1.jpg', './images/hmbb_2.jpg', './images/hmbb_3.jpg'],
            ['./images/rainbow_1.jpg', './images/rainbow_2.jpg', './images/rainbow_3.jpg'],
            ['./images/earth_1.jpg', './images/earth_2.jpg', './images/earth_3.jpg'],
            ['./images/obj_1.jpg', './images/obj_2.jpg', './images/obj_3.jpg'],
            ['./images/ydt_2.jpg', './images/ydt_1.jpg', './images/ydt_3.jpg'],
           ]

examples_video = [
            ['./videos/jeep-moving_Porsche/img1.jpg', './videos/jeep-moving_Porsche/img1.gif'],
            ['./videos/jeep-moving_Porsche/img2.jpg', './videos/jeep-moving_Porsche/img2.gif'],
            ['./videos/child-riding_lego/img1.jpg', './videos/child-riding_lego/img1.gif'],
            ['./videos/child-riding_lego/img2.jpg', './videos/child-riding_lego/img2.gif'],
            ['./videos/man-running_stephen/img1.jpg', './videos/man-running_stephen/img1.gif'],
            ['./videos/man-running_stephen/img2.jpg', './videos/man-running_stephen/img2.gif'],
]



demo_mask = gr.Interface(fn=inference_mask1,
                   inputs=[gr.ImageMask(brush_radius=8, label="prompt (提示图)"), gr.Image(label="img1 (测试图1)"), gr.Image(label="img2 (测试图2)")],
                    #outputs=[gr.Image(shape=(448, 448), label="output1 (输出图1)"), gr.Image(shape=(448, 448), label="output2 (输出图2)")],
                    outputs=[gr.Image(label="output1 (输出图1)").style(height=256, width=256), gr.Image(label="output2 (输出图2)").style(height=256, width=256)],
                    #outputs=gr.Gallery(label="outputs (输出图)"),
                    examples=examples,
                    #title="SegGPT for Any Segmentation<br>(Painter Inside)",
                    description="<p> \
                    Choose an example below &#128293; &#128293;  &#128293; <br>\
                    Or, upload by yourself: <br>\
                    1. Upload images to be tested to 'img1' and/or 'img2'. <br>2. Upload a prompt image to 'prompt' and draw a mask.  <br>\
                            <br> \
                            💎 The more accurate you annotate, the more accurate the model predicts. <br>\
                            💎 Examples below were never trained and are randomly selected for testing in the wild. <br>\
                            💎 Current UI interface only unleashes a small part of the capabilities of SegGPT, i.e., 1-shot case. \
</p>",
                   cache_examples=False,
                   allow_flagging="never",
                   )

demo_mask_video = gr.Interface(fn=inference_mask_video,
                   inputs=[gr.ImageMask(label="prompt (提示图)"), gr.Video(label="video (测试视频)").style(height=448, width=448)],
                    outputs=gr.Video().style(height=448, width=448),
                    examples=examples_video,
                    description="<p> \
                    Choose an example below &#128293; &#128293;  &#128293; \
                    Or, upload by yourself: <br>\
                    1. Upload a video to be tested to 'video'. <br>2. Upload a prompt image to 'prompt' and draw a mask.  <br>\
💎 The more accurate you annotate, the more accurate the model predicts. <br>\
💎 Examples below were never trained and are randomly selected for testing in the wild. <br>\
💎 Current UI interface only unleashes a small part of the capabilities of SegGPT, i.e., 1-shot case. \
</p>",
                   )

title = "SegGPT: Segmenting Everything In Context<br> \
<div align='center'> \
<h2><a href='https://arxiv.org/abs/2304.03284' target='_blank' rel='noopener'>[paper]</a> \
<a href='https://github.com/baaivision/Painter' target='_blank' rel='noopener'>[code]</a></h2> \
<br> \
<image src='file/rainbow.gif' width='720px' /> \
<h2>SegGPT performs arbitrary segmentation tasks in images or videos via in-context inference, such as object instance, stuff, part, contour, and text, with only one single model.</h2> \
</div> \
"

demo = gr.TabbedInterface([demo_mask, ], ['General 1-shot', '🎬Anything in a Video'], title=title)

#demo.launch(share=True, auth=("baai", "vision"))
demo.launch(enable_queue=False)



