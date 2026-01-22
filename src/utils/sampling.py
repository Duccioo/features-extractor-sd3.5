"""
Sampling Utilities
"""

import torch


def get_sigmas(sampling, steps, device):
    """Calculates the noise schedule (Sigmas) for sampling.

    Args:
        sampling: ModelSamplingDiscreteFlow instance
        steps: Number of diffusion steps
        device: Device for output tensor

    Returns:
        Tensor of sigma values
    """
    start = sampling.timestep(sampling.sigma_max)
    end = sampling.timestep(sampling.sigma_min)
    timesteps = torch.linspace(start, end, steps)
    sigs = []
    for x in range(len(timesteps)):
        ts = timesteps[x]
        sigs.append(sampling.sigma(ts))
    sigs += [0.0]  # Add the final zero
    return torch.FloatTensor(sigs).to(device)


def get_timesteps(sampling, steps, device):
    """Gets the diffusion timesteps.

    Args:
        sampling: ModelSamplingDiscreteFlow instance
        steps: Number of steps
        device: Device for output tensor

    Returns:
        Tensor of timestep values
    """
    start = sampling.timestep(sampling.sigma_max)
    end = sampling.timestep(sampling.sigma_min)
    return torch.linspace(start, end, steps).to(device)


def denoise(
    model, x, prompt_embeds, timesteps, pooled_prompt_embeds=None, guidance_scale=3.0
):
    """Simplified wrapper for the MM-DiT forward pass with CFG."""
    x_in = torch.cat([x] * 2)
    timesteps_in = torch.cat([timesteps] * 2)
    denoised_latents = model.forward(
        x_in,
        timesteps_in,
        context=prompt_embeds,
        y=pooled_prompt_embeds,
    )
    return perform_guidance(denoised_latents, guidance_scale)


def perform_guidance(denoised_latents, guidance_scale):
    """Applies classifier-free guidance."""
    pos, neg = denoised_latents.chunk(2)
    return neg + (pos - neg) * guidance_scale


def rescale_noise(denoised, timesteps, noise_pred):
    """Rescales the noise prediction."""
    return (denoised - noise_pred) / timesteps
