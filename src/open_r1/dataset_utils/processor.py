from typing import Union, Any
import torch
from PIL import Image
import numpy as np

# FIXME
MAX_SIDE = 1024

def prepare_images(inputs: dict[str, Union[torch.Tensor, Any]]) -> list:
    images = []    
    for x in inputs:
        print(x["data_source"], "data_source", x)
        img = Image_Prepare_Funcs[x["data_source"]](x)    
        images.append(img)
    return images

"""
x is a dictionary!
- image: PIL.PngImagePlugin.PngImageFile, mode=RGBA, size=640x480
- problem: str,  
  "Detect all objects belonging to the category 'umbrella' in the image, 
   and provide the bounding boxes (between 0 and 1000, integer) and confidence (between 0 and 1, with two decimal places).\n
   If no object belonging to the category 'umbrella' in the image, return 'No Objects'.\n
   Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.
   The output answer format should be as follows:\n<think> ... </think> <answer>[{'Position': [x1, y1, x2, y2], 'Confidence': number}, ...]</answer>\n
   Please strictly follow the format.
  "
- solution: str,
  ""<answer>[{'Position': [64, 0, 950, 411], 'Confidence': 1}]</answer>""
- prompt: list, each item:
 [
    {
        'content': [
            {'text': None, 'type': 'image'}, 
            {'text': "Detect all objects belonging to the category 'umbrella' in the image, and provide the bounding boxes (between 0 and 1000, integer) and confidence (between 0 and 1, with two decimal places).\nIf no object belonging to the category 'umbrella' in the image, return 'No Objects'.\nOutput the thinking process in <think> </think> and final answer in <answer> </answer> tags.The output answer format should be as follows:\n<think> ... </think> <answer>[{'Position': [x1, y1, x2, y2], 'Confidence': number}, ...]</answer>\nPlease strictly follow the format.", 'type': 'text'}
        ], 
        'role': 'user'
    }
 ]
- data_source: str, 'ViRFT_COCO'
"""    
def prepare_image(x):
    img = x["image"]
    try:
        w, h = img.size
            
        if w < 28 or h < 28:
        # Calculate new dimensions maintaining aspect ratio
            if w < h:
                new_w = 28
                new_h = int(h * (28/w))
            else:
                new_h = 28
                new_w = int(w * (28/h))
        
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    except:
        pass    
    return img
    img_array = np.array(x["image"])  # shape is (H, W) or (H, W, C)
    img_scaled = (img_array * 255).astype(np.uint8)
    # Convert back to a PIL Image
    img_255 = Image.fromarray(img_scaled)
    return img_255


def prepare_image_rlaif(x):
    im = x["image"]
    w, h = im.size
    longer = max(w, h)
    scale = 1.0
    new_w=w
    new_h=h
    if longer > MAX_SIDE:
        scale = MAX_SIDE / float(longer)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        # LANCZOS = high-quality downsampling
        im = im.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    return im
def prepare_image_path(x):
    return Image.open(x["image_path"])#.convert('RGBA')

def prepare_image_coco(x):
    img = Image.open(x["image_path"])#.convert('RGBA')
    try:
        w, h = img.size
            
        if w < 28 or h < 28:
        # Calculate new dimensions maintaining aspect ratio
            if w < h:
                new_w = 28
                new_h = int(h * (28/w))
            else:
                new_h = 28
                new_w = int(w * (28/h))
        
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    except:
        pass
    
    
    return img

# each dataset would have a image processor
Image_Prepare_Funcs = {
    "ViRFT_COCO": prepare_image, 
    "SAT": prepare_image,
    "GEOQA": prepare_image,
    "LISA": prepare_image,
    "LISA_BBOX": prepare_image,
    "SCIENCEQA": prepare_image,
    "GEOQAV": prepare_image,
    "RefCOCO": prepare_image_coco,
    "ViRL39K": prepare_image_coco,
    "RLAIF_V": prepare_image_rlaif,
    "LLaVA_OneVision_OCR": prepare_image,
    "LLaVA_OneVision_OCR_10k": prepare_image,
    "LLaVA_OneVision_OCR_10k_128": prepare_image,
    "ThinkLite70k":prepare_image,
    "ThinkLite11k":prepare_image,
    
    "LLaVA_OneVision_OCR_5k": prepare_image,
    
}
