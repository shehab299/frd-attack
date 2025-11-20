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
        # Handle Tensor inputs (differentiable path)
        if isinstance(images, torch.Tensor):
            images = images.to(self.device)
            # Input expected to be [B, C, H, W] in range [0, 1]
            # CLIP expects 224x224
            if images.shape[-2:] != (224, 224):
                images = torch.nn.functional.interpolate(images, size=(224, 224), mode='bicubic', align_corners=False)
            
            # Normalize using CLIP's mean and std from processor
            mean = torch.tensor(self.processor.image_processor.image_mean).to(self.device).view(1, 3, 1, 1)
            std = torch.tensor(self.processor.image_processor.image_std).to(self.device).view(1, 3, 1, 1)
            
            pixel_values = (images - mean) / std
            
        # Handle PIL inputs (non-differentiable path, standard usage)
        else:
            inputs = self.processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
        
        # Get image features
        embeddings = self.model.get_image_features(pixel_values=pixel_values)
        
        # Normalize
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        return embeddings