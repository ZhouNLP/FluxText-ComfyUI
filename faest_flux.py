import torch
from diffusers import TorchAoConfig,FluxTransformer2DModel

'''
qcofig = TorchAoConfig(quant_type="int8wo")
transformer = FluxTransformer2DModel.from_pretrained(
    "/root/FluxText/FLUX.1-Fill-dev",
    subfolder="transformer",
    quantization_config=qcofig,
    torch_dtype=torch.bfloat16,
)
transformer.save_pretrained("/root/ComfyUI/models/AIFSH/flux-fill-transformer-int8wo",safe_serialization=False)
transformer.to("cuda")
'''

from torchao.quantization import quantize_, int8_weight_only, float8_weight_only
# transformer.save_pretrained("/root/ComfyUI/models/AIFSH/flux-fill-transformer-float8_dafw",safe_serialization=False)
transformer = FluxTransformer2DModel.from_pretrained(
    "/root/FluxText/FLUX.1-Fill-dev",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
)
# transformer = FluxTransformer2DModel.from_pretrained("/root/ComfyUI/models/AIFSH/flux-fill-transformer-float8_dafw",torch_dtype=torch.bfloat16,use_safetensors=False)
from safetensors.torch import load_file

# state_dict = load_file('runs_word_fill2/20250215-230658/ckpt/105000/pytorch_lora_weights.safetensors')
state_dict = load_file("/root/FluxText/FLUX-Text/model_multisize/pytorch_lora_weights.safetensors")
state_dict1 = {x.replace('lora_A', 'lora_A.default').replace('lora_B', 'lora_B.default').replace('transformer.', ''): v for x, v in state_dict.items()}
transformer.load_state_dict(state_dict1, strict=False)
quantize_(transformer,int8_weight_only())
transformer.save_pretrained("/root/ComfyUI/models/AIFSH/flux-text-transformer-int8wo",safe_serialization=False)
transformer.to("cuda")
# transformer = FluxTransformer2DModel.from_pretrained("/root/ComfyUI/models/AIFSH/flux-fill-transformer-float8_dafw",torch_dtype=torch.bfloat16,use_safetensors=False)
# transformer.save_pretrained("/root/ComfyUI/models/AIFSH/flux-text-transformer-float8_dafw",safe_serialization=False)
# transformer.to("cuda")

'''
from torchao.quantization import quantize_, int8_weight_only
from transformers import T5EncoderModel

t5_encoder = T5EncoderModel.from_pretrained(
    "/root/FluxText/FLUX.1-Fill-dev",
    subfolder="text_encoder_2",
)
quantize_(t5_encoder,int8_weight_only())
t5_encoder.save_pretrained("/root/ComfyUI/models/AIFSH/flux-fill-text_encoder_2-int8wo",safe_serialization=False)

# t5_encoder = T5EncoderModel.from_pretrained("/root/ComfyUI/models/AIFSH/flux-fill-text_encoder_2-float8wo")
t5_encoder.to("cuda")
'''