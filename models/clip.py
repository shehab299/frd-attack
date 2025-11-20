import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image


class CLIPImageEmbedder:
    def __init__(self, model_name='openai/clip-vit-large-patch14', device='cpu'):
        """
        CLIP embedder using ViTEmbedder structure, with CLIP processor.
        """
        self.device = torch.device(device)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)
        self.model.eval()

    def get_embeddings(self, images):
        """
        Extract embeddings for batch of images (for adversarial attacks).
        Handles both PIL images and torch tensors.
        """
        # Convert tensor to PIL for proper CLIP preprocessing
        if isinstance(images, torch.Tensor):
            # Convert [B, C, H, W] float [0,1] to PIL images
            pil_images = []
            for img_tensor in images:
                # Convert to [H, W, C] and scale to [0, 255]
                img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
                pil_images.append(Image.fromarray(img_np))
            images = pil_images
        
        # Preprocess with CLIP processor (handles resize, normalize, etc.)
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        
        # Get image features
        embeddings = self.model.get_image_features(pixel_values=pixel_values)
        
        # Normalize
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        return embeddings