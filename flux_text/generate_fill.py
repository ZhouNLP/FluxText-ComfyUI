import os
from typing import List, Union, Optional, Dict, Any, Callable

from diffusers.pipelines import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux import (
    FluxPipelineOutput,
    calculate_shift,
    retrieve_timesteps,
    np,
)
import torch
import torch.nn.functional as F
import yaml

from .condition import Condition
from .transformer import tranformer_forward
from .pipeline_tools import encode_images
# from ..text_encoder.byt5_encoder import GlyphByt5Encoder

def get_config(config_path: str = None):
    config_path = config_path or os.environ.get("XFL_CONFIG")
    if not config_path:
        return {}
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def prepare_params(
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = 512,
    width: Optional[int] = 512,
    num_inference_steps: int = 28,
    timesteps: List[int] = None,
    guidance_scale: float = 3.5,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
    **kwargs: dict,
):
    return (
        prompt,
        prompt_2,
        height,
        width,
        num_inference_steps,
        timesteps,
        guidance_scale,
        num_images_per_prompt,
        generator,
        latents,
        prompt_embeds,
        pooled_prompt_embeds,
        output_type,
        return_dict,
        joint_attention_kwargs,
        callback_on_step_end,
        callback_on_step_end_tensor_inputs,
        max_sequence_length,
    )


def seed_everything(seed: int = 42):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    np.random.seed(seed)

from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines import FluxFillPipeline

def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)

        latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)

        return latents, latent_image_ids


@torch.no_grad()
def generate_fill(
    pipeline: FluxPipeline,
    conditions: List[Condition] = None,
    config_path: str = None,
    model_config: Optional[Dict[str, Any]] = {},
    condition_scale: float = 1.0,
    **params: dict,
):
    model_config = model_config or get_config(config_path).get("model", {})
    if condition_scale != 1:
        for name, module in pipeline.transformer.named_modules():
            if not name.endswith(".attn"):
                continue
            module.c_factor = torch.ones(1, 1) * condition_scale

    self = pipeline

    (
        prompt,
        prompt_2,
        height,
        width,
        num_inference_steps,
        timesteps,
        guidance_scale,
        num_images_per_prompt,
        generator,
        latents,
        prompt_embeds,
        pooled_prompt_embeds,
        output_type,
        return_dict,
        joint_attention_kwargs,
        callback_on_step_end,
        callback_on_step_end_tensor_inputs,
        max_sequence_length,
    ) = prepare_params(**params)

    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor
    
    # 1. Check inputs. Raise error if not correct
    '''
    self.check_inputs(
        prompt=prompt,
        prompt_2=prompt_2,
        height=height,
        width=width,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )
    '''
    self._guidance_scale = guidance_scale
    self._joint_attention_kwargs = joint_attention_kwargs
    self._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    lora_scale = (
        self.joint_attention_kwargs.get("scale", None)
        if self.joint_attention_kwargs is not None
        else None
    )
    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )

    # 4. Prepare latent variables
    # num_channels_latents = self.transformer.config.in_channels // 4
    
    num_channels_latents = self.vae.config.latent_channels
    self.__class__.prepare_latents = prepare_latents
    latents, latent_image_ids = self.prepare_latents(
        batch_size=batch_size * num_images_per_prompt,
        num_channels_latents=num_channels_latents,
        height = height,
        width = width,
        dtype = prompt_embeds.dtype,
        device = device,
        generator = generator,
        latents = latents,
    )
    
    # 4.1. Prepare conditions
    condition_img, hint, imgs = conditions[0].condition
    type_id = conditions[0].type_id
    _height = 2 * (int(height) // (self.vae_scale_factor * 2))
    _width = 2 * (int(width) // (self.vae_scale_factor * 2))
    condition_height, condition_width, _ = condition_img.shape
    condition_height = 2 * (int(condition_height) // (self.vae_scale_factor * 2))
    condition_width = 2 * (int(condition_width) // (self.vae_scale_factor * 2))
    
    condition_latents, condition_ids = encode_images(self, condition_img)
    condition_type_ids = (torch.ones_like(condition_ids[:, 0]) * type_id).unsqueeze(1)

    condition_img = condition_img[:, :, 0]
    mask_image = hint[:, :, 0]

    imgs = self.image_processor.preprocess(imgs, height=height, width=width) 
    mask_image = self.mask_processor.preprocess(mask_image, height=height, width=width)
    masked_image = imgs * (1 - mask_image)
    def _encode_mask(images):
        images = images.to(self.device).to(self.dtype)
        images = self.vae.encode(images).latent_dist.sample()
        images = (
            images - self.vae.config.shift_factor
        ) * self.vae.config.scaling_factor
        condition_images = F.interpolate(images, size=(condition_height, condition_width), mode='nearest')

        images_tokens = self._pack_latents(images, *images.shape)
        images_ids = self._prepare_latent_image_ids(
            images.shape[0],
            images.shape[2],
            images.shape[3],
            self.device,
            self.dtype,
        )
        conditon_images_tokens = self._pack_latents(condition_images, *condition_images.shape)
        return images_tokens, images_ids, conditon_images_tokens
        # return images_tokens, images_ids
    masked_image_latents, _, conditon_masked_images_latents = _encode_mask(masked_image)
    # masked_image_latents, _ = _encode_mask(masked_image)

    # mask = mask_image * (1 - condition_img)
    mask = mask_image
    mask = torch.tensor(mask)
    mask = mask.view(batch_size, _height, self.vae_scale_factor, _width, self.vae_scale_factor)
    mask = mask.permute(0, 2, 4, 1, 3)  # batch_size, 8, 8, height, width
    mask = mask.reshape(
        batch_size, self.vae_scale_factor * self.vae_scale_factor, _height, _width
    )  # batch_size, 8*8, height, width
    mask = self._pack_latents(mask, batch_size, self.vae_scale_factor * self.vae_scale_factor, _height, _width)
    mask = mask.to(masked_image_latents.device, dtype=masked_image_latents.dtype)

    masked_image_latents = torch.cat((masked_image_latents, mask), dim=-1)

    condition_mask = mask_image
    condition_mask = F.interpolate(condition_mask, size=(condition_img.shape[0], condition_img.shape[1]), mode='nearest')
    condition_mask = condition_mask.view(batch_size, condition_height, self.vae_scale_factor, condition_width, self.vae_scale_factor)
    condition_mask = condition_mask.permute(0, 2, 4, 1, 3)  # batch_size, 8, 8, height, width
    condition_mask = condition_mask.reshape(
        batch_size, self.vae_scale_factor * self.vae_scale_factor, condition_height, condition_width
    )  # batch_size, 8*8, height, width
    condition_mask = self._pack_latents(condition_mask, batch_size, self.vae_scale_factor * self.vae_scale_factor, condition_height, condition_width)
    condition_mask = condition_mask.to(conditon_masked_images_latents.device, dtype=conditon_masked_images_latents.dtype)

    conditon_masked_images_latents = torch.cat((conditon_masked_images_latents, condition_mask), dim=-1).to(self.dtype)

    # 5. Prepare timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    image_seq_len = latents.shape[1]
    # image_seq_len = (int(height) // self.vae_scale_factor // 2) * (int(width) // self.vae_scale_factor // 2)
    mu = calculate_shift(
        image_seq_len,
        self.scheduler.config.base_image_seq_len,
        self.scheduler.config.max_image_seq_len,
        self.scheduler.config.base_shift,
        self.scheduler.config.max_shift,
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        timesteps,
        sigmas,
        mu=mu,
    )
    num_warmup_steps = max(
        len(timesteps) - num_inference_steps * self.scheduler.order, 0
    )
    self._num_timesteps = len(timesteps)
    '''
    latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
    # 5. Prepare latent variables
    num_channels_latents = self.vae.config.latent_channels
    latents, latent_image_ids = self.prepare_latents(
        imgs,
        latent_timestep,
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )
    '''
    # 6. Denoising loop
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            # handle guidance
            if self.transformer.config.guidance_embeds:
                guidance = torch.tensor([guidance_scale], device=device)
                guidance = guidance.expand(latents.shape[0])
            else:
                guidance = None
            noise_pred = tranformer_forward(
                self.transformer,
                model_config=model_config,
                # Inputs of the condition (new feature)
                condition_latents=torch.cat((condition_latents, conditon_masked_images_latents), dim=2),
                condition_ids=condition_ids,
                condition_type_ids=condition_type_ids,
                # Inputs to the original transformer
                hidden_states=torch.cat((latents, masked_image_latents), dim=2),
                # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
                timestep=timestep / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
            )[0]

            # compute the previous noisy sample x_t -> x_t-1
            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    latents = latents.to(latents_dtype)

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

            # call the callback, if provided
            if i == len(timesteps) - 1 or (
                (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
            ):
                progress_bar.update()

    if output_type == "latent":
        image = latents

    else:
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = (
            latents / self.vae.config.scaling_factor
        ) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    self.maybe_free_model_hooks()

    if condition_scale != 1:
        for name, module in pipeline.transformer.named_modules():
            if not name.endswith(".attn"):
                continue
            del module.c_factor

    if not return_dict:
        return (image,)

    return FluxPipelineOutput(images=image)