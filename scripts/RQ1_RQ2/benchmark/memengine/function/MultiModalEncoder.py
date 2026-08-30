from abc import ABC, abstractmethod
import torch
import os
import warnings

from benchmark.paths import IMAGE_ROOT
from benchmark.security import redact_runtime_text

class BaseMultiModalEncoder(ABC):
    """
    Encoder for multimodal data (text + image).
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def reset(self):
        pass
    
    @abstractmethod
    def encode_text(self, text, return_type='numpy'):
        pass
    
    @abstractmethod
    def encode_image(self, image_path_or_url, return_type='numpy'):
        pass
    
    @abstractmethod
    def encode_multimodal(self, text=None, image=None, return_type='numpy'):
        pass


class CLIPEncoder(BaseMultiModalEncoder):
    """
    CLIP-based multimodal encoder for text and images.
    """
    def __init__(self, config):
        super().__init__(config)
        
        try:
            from transformers import CLIPProcessor, CLIPModel
        except ImportError as exc:
            raise RuntimeError('install the optional `transformers` dependency to use CLIPEncoder') from exc

        model_name = getattr(config, 'path', None) or os.environ.get('CUE_MEM_CLIP_MODEL_PATH', '')
        if not model_name:
            raise ValueError('CLIPEncoder requires CUE_MEM_CLIP_MODEL_PATH or an explicit model path')
        print(f"Loading CLIP model: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode
        print(f"CLIP model loaded successfully on {self.device}")
    
    def _load_image(self, image_path_or_url):
        """Load image from local path or URL."""
        from io import BytesIO

        from PIL import Image
        import requests

        if os.path.isabs(image_path_or_url) or image_path_or_url.startswith('http://') or image_path_or_url.startswith('https://'):
            image_path_or_url = image_path_or_url
        else:
            image_path_or_url = str((IMAGE_ROOT / image_path_or_url).resolve())
        try:
            if image_path_or_url.startswith('http://') or image_path_or_url.startswith('https://'):
                # Load from URL
                response = requests.get(image_path_or_url, timeout=10)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                # Load from local path
                if not os.path.exists(image_path_or_url):
                    raise FileNotFoundError(f"Image file not found: {image_path_or_url}")
                image = Image.open(image_path_or_url).convert('RGB')
            return image
        except Exception as e:
            print(f"Error loading image {redact_runtime_text(image_path_or_url)}: {redact_runtime_text(e)}")
            return Image.new('RGB', (224, 224), color='white')
    
    def encode_text(self, text, return_type='numpy'):
        """Encode text into embeddings."""
        if not text or text.strip() == '':
            text = " "
        
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        if return_type == 'numpy':
            return text_features.cpu().numpy()
        elif return_type == 'tensor':
            return text_features
        else:
            raise ValueError(f"Unrecognized return type: {return_type}")
    
    def encode_image(self, image_path_or_url, return_type='numpy'):
        """Encode image into embeddings."""
        image = self._load_image(image_path_or_url)
        
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        if return_type == 'numpy':
            return image_features.cpu().numpy()
        elif return_type == 'tensor':
            return image_features
        else:
            raise ValueError(f"Unrecognized return type: {return_type}")
    
    def encode_multimodal(self, text=None, image=None, return_type='numpy'):
        """
        Encode multimodal data (text and/or image).
        If both are provided, average the embeddings.
        """
        embeddings = []
        
        if text is not None and text.strip() != '':
            text_emb = self.encode_text(text, return_type='tensor')
            embeddings.append(text_emb)
        
        if image is not None:
            image_emb = self.encode_image(image.get('path'), return_type='tensor')
            embeddings.append(image_emb)
        
        if not embeddings:
            # If both are empty, encode empty text
            return self.encode_text(" ", return_type=return_type)
        
        # Average the embeddings if both modalities are present
        if len(embeddings) > 1:
            combined = torch.mean(torch.stack(embeddings), dim=0)
        else:
            combined = embeddings[0]
        
        # Normalize
        combined = combined / combined.norm(dim=-1, keepdim=True)
        
        if return_type == 'numpy':
            return combined.cpu().numpy()
        elif return_type == 'tensor':
            return combined
        else:
            raise ValueError(f"Unrecognized return type: {return_type}")
    
    def __call__(self, obj, return_type='numpy'):
        """
        Main entry point. obj can be:
        - str: treated as text
        - dict with 'text' and/or 'image' keys
        """
        if isinstance(obj, str):
            return self.encode_text(obj, return_type)
        elif isinstance(obj, dict):
            text = obj.get('text', '')
            image = obj.get('image', None)
            return self.encode_multimodal(text, image, return_type)
        else:
            raise ValueError(f"Unsupported input type: {type(obj)}")


class GMEEncoder(BaseMultiModalEncoder):
    """
    GME (General Multimodal Embedding) Qwen2-VL-based encoder for text and images.
    Supports unified multimodal representations for Any2Any Search.
    """
    def __init__(self, config):
        super().__init__(config)

        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError('install the optional `transformers` dependency to use GMEEncoder') from exc
        
        model_name = getattr(config, 'path', None) or os.environ.get('CUE_MEM_GME_MODEL_PATH', '')
        if not model_name:
            raise ValueError('GMEEncoder requires an explicit model path or CUE_MEM_GME_MODEL_PATH')
        print(f"Loading GME model: {model_name}")
        
        # Load GME model with trust_remote_code=True
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
            device_map='auto',
            trust_remote_code=True
        )
        self.model.eval()  # Set to evaluation mode
        print(f"GME model loaded successfully on {self.device}")
    
    def _load_image(self, image_path_or_url):
        """Load image from local path or URL."""
        from io import BytesIO

        from PIL import Image
        import requests

        # If already absolute path or URL, use directly
        if os.path.isabs(image_path_or_url) or image_path_or_url.startswith('http://') or image_path_or_url.startswith('https://'):
            final_path = image_path_or_url
        else:
            final_path = str((IMAGE_ROOT / image_path_or_url).resolve())
        
        try:
            if final_path.startswith('http://') or final_path.startswith('https://'):
                # Load from URL
                response = requests.get(final_path, timeout=10)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                # Load from local path
                if not os.path.exists(final_path):
                    raise FileNotFoundError(f"Image file not found: {final_path}")
                image = Image.open(final_path).convert('RGB')
            return image
        except Exception as e:
            print(
                f"Error loading image {redact_runtime_text(image_path_or_url)} "
                f"(tried {redact_runtime_text(final_path)}): {redact_runtime_text(e)}"
            )
            # Return a blank image as fallback
            return Image.new('RGB', (224, 224), color='white')
    
    def encode_text(self, text, return_type='numpy'):
        """Encode text into embeddings."""
        if not text or text.strip() == '':
            text = " "
        
        with torch.no_grad():
            text_emb = self.model.get_text_embeddings(texts=[text])
            # Normalize
            text_emb = text_emb / torch.norm(text_emb, dim=-1, keepdim=True)
        
        if return_type == 'numpy':
            return text_emb.cpu().numpy()
        elif return_type == 'tensor':
            return text_emb
        else:
            raise ValueError(f"Unrecognized return type: {return_type}")
    
    def encode_image(self, image_path_or_url, return_type='numpy'):
        """Encode image into embeddings."""
        image = self._load_image(image_path_or_url)
        
        with torch.no_grad():
            # GME API: get_image_embeddings expects images as list of PIL Image
            image_emb = self.model.get_image_embeddings(images=[image])
            # Normalize
            image_emb = image_emb / torch.norm(image_emb, dim=-1, keepdim=True)
        
        if return_type == 'numpy':
            return image_emb.cpu().numpy()
        elif return_type == 'tensor':
            return image_emb
        else:
            raise ValueError(f"Unrecognized return type: {return_type}")
    
    def encode_multimodal(self, text=None, image=None, return_type='numpy'):
        """
        Encode multimodal data (text and/or image).
        GME supports single-modal (text/image) and fused-modal embeddings.
        If both are provided, we use fused-modal embedding for better representation.
        """
        # Determine which encoding method to use
        has_text = text is not None and text.strip() != ''
        has_image = image is not None
        
        if has_text and has_image:
            # Fused-modal embedding: best for multimodal representation
            image_path = image.get('path')
            loaded_image = self._load_image(image_path)
            with torch.no_grad():
                fused_emb = self.model.get_fused_embeddings(texts=[text], images=[loaded_image])
                fused_emb = fused_emb / torch.norm(fused_emb, dim=-1, keepdim=True)
            
            if return_type == 'numpy':
                return fused_emb.cpu().numpy()
            elif return_type == 'tensor':
                return fused_emb
            else:
                raise ValueError(f"Unrecognized return type: {return_type}")
        
        elif has_text:
            # Single-modal text embedding
            return self.encode_text(text, return_type)
        
        elif has_image:
            # Single-modal image embedding
            return self.encode_image(image.get('path'), return_type)
        
        else:
            # If both are empty, encode empty text
            return self.encode_text(" ", return_type)
    
    def __call__(self, obj, return_type='numpy'):
        """
        Main entry point. obj can be:
        - str: treated as text
        - dict with 'text' and/or 'image' keys
        """
        if isinstance(obj, str):
            return self.encode_text(obj, return_type)
        elif isinstance(obj, dict):
            text = obj.get('text', '')
            image = obj.get('image', None)
            return self.encode_multimodal(text, image, return_type)
        else:
            raise ValueError(f"Unsupported input type: {type(obj)}")
