"""
Grad-CAM (Gradient-Weighted Class Activation Mapping) implemented from scratch.

Uses PyTorch forward and full backward hooks on the target convolutional layer
to compute activation saliency maps without any external Grad-CAM libraries.
"""

from __future__ import annotations

import base64
import io
from typing import Optional, Tuple

import matplotlib.cm as cm
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM engine implemented with PyTorch forward & backward hooks.

    Math:
      1. Gradient weights:
         alpha_k^c = (1 / (H * W)) * sum_{i=1}^H sum_{j=1}^W (d y^c / d A_{i, j}^k)
      2. Heatmap:
         L_{Grad-CAM}^c = ReLU( sum_{k=1}^C alpha_k^c * A^k )
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self.fwd_hook = self.target_layer.register_forward_hook(self._forward_hook)
        self.bwd_hook = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module: nn.Module, input_t: Tuple, output_t: torch.Tensor) -> None:
        self.activations = output_t.detach()

    def _backward_hook(
        self, module: nn.Module, grad_input: Tuple, grad_output: Tuple[torch.Tensor, ...]
    ) -> None:
        # grad_output[0] is the gradient of the scalar score w.r.t target layer activations
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generates normalized Grad-CAM heatmap for the input tensor.

        Args:
            input_tensor: Tensor of shape (1, 3, H, W)
            target_class: Target class index. If None, uses model's predicted class.

        Returns:
            cam: 2D numpy array of shape (H, W) in range [0, 1]
            pred_class: Predicted class index
            confidence: Softmax probability for the predicted class
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)
        
        # Forward pass
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)

        pred_class = int(logits.argmax(dim=1).item())
        confidence = float(probs[0, pred_class].item())

        c = pred_class if target_class is None else target_class

        # Zero existing gradients and backward pass on target class logit
        self.model.zero_grad()
        target_score = logits[0, c]
        target_score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Hooks failed to capture activations or gradients.")

        # Compute importance weights alpha_k via Global Average Pooling over spatial dims (H, W)
        # activations shape: (1, C, H_feat, W_feat)
        # gradients shape:   (1, C, H_feat, W_feat)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of feature maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, H_feat, W_feat)
        cam = F.relu(cam)  # Discard negative influences

        # Bilinear interpolation to match input resolution (H, W)
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()

        # Min-max normalization to [0, 1]
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, pred_class, confidence

    def remove_hooks(self) -> None:
        """Removes registered PyTorch hooks."""
        self.fwd_hook.remove()
        self.bwd_hook.remove()


def overlay_gradcam_on_image(
    image: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "turbo",
) -> Image.Image:
    """
    Blends Grad-CAM saliency heatmap on top of the original candlestick image.
    """
    img_rgb = np.array(image.convert("RGB")).astype(np.float32) / 255.0

    # Apply colormap to CAM heatmap
    try:
        import matplotlib
        cmap = matplotlib.colormaps[colormap]
    except Exception:
        cmap = cm.get_cmap(colormap)
    heatmap = cmap(cam)[:, :, :3]  # Strip alpha channel

    # Blend original image with heatmap
    overlay = (alpha * heatmap) + ((1.0 - alpha) * img_rgb)
    overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def image_to_base64(img: Image.Image, format: str = "PNG") -> str:
    """Encodes PIL Image to base64 data URI string."""
    buf = io.BytesIO()
    img.save(buf, format=format)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{encoded}"


def base64_to_image(b64_str: str) -> Image.Image:
    """Decodes base64 data URI or raw string to PIL Image."""
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    data = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(data)).convert("RGB")
