import gc
import os.path as osp
import torch
import folder_paths
import comfy.model_management as mm

device = mm.get_torch_device()
offload_device = mm.unet_offload_device()
dtype = torch.bfloat16
aifsh_dir = osp.join(folder_paths.models_dir,"AIFSH")

import yaml
from .utils import block_swap_to,t5_stack_forward
from safetensors.torch import load_file
from transformers import T5EncoderModel
from diffusers import FluxFillPipeline
from peft import LoraConfig

class FluxLoraLoader:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "lora":(folder_paths.get_filename_list("loras"),),
                "weight":("FLOAT",{
                    "default":1.0
                })
            }
        }
    
    RETURN_TYPES = ("FLUXTLORA",)
    RETURN_NAMES = ("lora",)

    FUNCTION = "load_lora"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH/FluxText"

    def load_lora(self,lora,weight):
        res = dict(
            lora_path = folder_paths.get_full_path_or_raise("loras",lora),
            weight = weight
        )
        return (res,)

class FluxTextPipeLine:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "pipe_name":("STRING",{
                    "default":"/root/FluxText/FLUX.1-Fill-dev",
                }),
                "flux_text":(folder_paths.get_filename_list("loras"),),
                "blocks_to_swap": ("INT", {"default": 10, "min": 0, "max": 19, "step": 1, "tooltip": "Number of transformer blocks to swap"}),
                "single_blocks_to_swap": ("INT", {"default": 19, "min": 0, "max": 38, "step": 1, "tooltip": "Number of transformer blocks to swap"}),
                "text_encoder2_blocks_to_swap": ("INT", {"default": 12, "min": 0, "max": 24, "step": 1, "tooltip": "Number of text encoder 2 blocks to swap"}),
            },
            "optional":{
                "lora":("FLUXTLORA",),
            }
        }
    
    RETURN_TYPES = ("FLUXTEXTPIPE",)
    RETURN_NAMES = ("pipe",)

    FUNCTION = "load_pipe"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH/FluxText"

    def load_pipe(self,pipe_name,flux_text,blocks_to_swap,
                  single_blocks_to_swap,
                  text_encoder2_blocks_to_swap,
                  lora=None):
        config_path = osp.join(now_dir, 'config.yaml')
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        pipe = FluxFillPipeline.from_pretrained(pipe_name,torch_dtype=dtype)
        if lora is not None:
            pipe.load_lora_weights(lora['lora_path'])
            pipe.fuse_lora(lora_scale=lora['weight'])

        pipe.vae_scale_factor = 8
        pipe.text_encoder.requires_grad_(False).eval()
        pipe.text_encoder_2.requires_grad_(False).eval()
        pipe.vae.requires_grad_(False).eval()
        lora_config = config['train']['lora_config']
        pipe.transformer.add_adapter(LoraConfig(**lora_config))

        state_dict = load_file(folder_paths.get_full_path_or_raise("loras",flux_text))
        state_dict1 = {x.replace('lora_A', 'lora_A.default').replace('lora_B', 'lora_B.default').replace('transformer.', ''): v for x, v in state_dict.items()}
        pipe.transformer.load_state_dict(state_dict1, strict=False)
        del state_dict,state_dict1
        gc.collect()
        pipe.__class__.to = block_swap_to
        pipe.transformer.__class__.offload_device = offload_device
        pipe.transformer.__class__.main_device = device
        pipe.transformer.__class__.blocks_to_swap = blocks_to_swap
        pipe.transformer.__class__.single_blocks_to_swap = single_blocks_to_swap
        pipe.transformer.__class__.use_non_blocking = True

        pipe.text_encoder_2.encoder.__class__.text_encoder2_blocks_to_swap = text_encoder2_blocks_to_swap
        pipe.text_encoder_2.encoder.__class__.offload_device = offload_device
        pipe.text_encoder_2.encoder.__class__.main_device = device
        pipe.text_encoder_2.encoder.__class__.forward = t5_stack_forward
        pipe.to(device)
        return (pipe,)

import numpy as np
from PIL import Image
from .flux_text.generate_fill import generate_fill
from .flux_text.condition import Condition
from .utils import (PIXELS, get_closest_ratio,
                    get_aspect_ratios_dict,
                    ASPECT_RATIO_LD_LIST,
                    render_glyph_multi,
                    now_dir
                    )

class FluxTextDrawGlyph:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "image":("IMAGE",),
                "mask":("IMAGE",),
                "words":("STRING",{
                    "multiline": True,
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("glyph_img",)

    FUNCTION = "draw"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH/FluxText"

    def comfy2pil(self,image):
        i = 255. * image.cpu().numpy()[0]
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        return img
    def pil2comfy(self,pil):
        # image = pil.convert("RGB")
        image = np.array(pil).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        return image
    
    def draw(self,image,mask,words):
        original_image = self.comfy2pil(image)
        computed_mask = self.comfy2pil(mask)
        texts = [line.strip() for line in words.splitlines() if line.strip()]
        render_img = render_glyph_multi(original_image, computed_mask, texts)
        return (self.pil2comfy(render_img),)

class FluxTextSampler:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "pipe":("FLUXTEXTPIPE",),
                "img":("IMAGE",),
                "glyph_img":("IMAGE",),
                "mask_img":("IMAGE",),
                "prompt":("STRING",),
                "num_inference_steps":("INT",{
                    "default": 8
                }),
                "seed":("INT",{
                    "default":42
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "sample"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH/FluxText"

    def comfy2pil(self,image):
        i = 255. * image.cpu().numpy()[0]
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        return img
    
    def pil2comfy(self,pil):
        # image = pil.convert("RGB")
        image = np.array(pil).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        return image

    def sample(self,pipe,img,glyph_img,mask_img,prompt,
               num_inference_steps,seed):
        
        img = self.comfy2pil(img)
        glyph_img = self.comfy2pil(glyph_img)
        mask_img = self.comfy2pil(mask_img)
        
        ori_width, ori_height = img.size
        num_pixel = min(PIXELS, key=lambda x: abs(x - ori_width * ori_height))
        aspect_ratio_dict = get_aspect_ratios_dict(num_pixel)
        close_ratio = get_closest_ratio(ori_height, ori_width, ASPECT_RATIO_LD_LIST)
        tgt_height, tgt_width = aspect_ratio_dict[close_ratio]
        
        hint = mask_img.resize((tgt_width, tgt_height)).convert('RGB')
        img = img.resize((tgt_width, tgt_height))
        condition_img = glyph_img.resize((tgt_width, tgt_height)).convert('RGB')
        hint = np.array(hint) / 255
        condition_img = np.array(condition_img)
        condition_img = (255 - condition_img) / 255
        condition_img = [condition_img, hint, img]
        position_delta = [0, 0]
        condition = Condition(
                        condition_type='word_fill',
                        condition=condition_img,
                        position_delta=position_delta,
                    )
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed)
        
        res = generate_fill(
            pipe,
            prompt=prompt,
            conditions=[condition],
            height=tgt_height,
            width=tgt_width,
            generator=generator,
            config_path=osp.join(now_dir,"config.yaml"),
            num_inference_steps=num_inference_steps
        )
        image = res.images[0]
        image = self.pil2comfy(image)
        return (image,)
    
NODE_CLASS_MAPPINGS = {
    "FluxTextDrawGlyph": FluxTextDrawGlyph,
    "FluxTextPipeLine":FluxTextPipeLine,
    "FluxTextSampler":FluxTextSampler,
    "FluxLoraLoader":FluxLoraLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxTextDrawGlyph": "DrawGlyph@关注超级面爸微信公众号",
    "FluxTextPipeLine":"PipeLine@关注超级面爸微信公众号",
    "FluxTextSampler":"Sampler@关注超级面爸微信公众号",
    "FluxLoraLoader":"FluxLoraLoader@关注超级面爸微信公众号",
}