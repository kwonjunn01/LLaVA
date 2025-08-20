#!/usr/bin/env python3
"""
CC3M Dataset loader for LLaVA calibration
Uses the CC3M pretrain dataset with 595K image-text pairs
"""

import json
import os
import random
import zipfile
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image
import io
import torch
from torch.utils.data import Dataset, DataLoader
from llava.mm_utils import tokenizer_image_token, process_images
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

class CC3MDataset(Dataset):
    """CC3M dataset for LLaVA calibration"""
    
    def __init__(self, 
                 data_path: str = "/home/diml/data/hj/cc3m",
                 tokenizer=None,
                 image_processor=None,
                 model_config=None,
                 max_samples: Optional[int] = None,
                 use_cache: bool = True):
        """
        Args:
            data_path: Path to CC3M dataset directory
            tokenizer: LLaVA tokenizer
            image_processor: LLaVA image processor
            model_config: Model configuration
            max_samples: Maximum number of samples to load
            use_cache: Whether to cache extracted images
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.use_cache = use_cache
        
        # Load chat data
        chat_file = self.data_path / "chat.json"
        with open(chat_file, 'r') as f:
            self.conversations = json.load(f)
        
        # Limit samples if specified
        if max_samples and max_samples < len(self.conversations):
            random.seed(42)  # For reproducibility
            self.conversations = random.sample(self.conversations, max_samples)
        
        # Setup image access
        self.images_zip = self.data_path / "images.zip"
        self.zipfile = None
        self.image_cache = {}
        
        # Extract images if using cache
        if self.use_cache:
            self.cache_dir = self.data_path / "extracted_images"
            if not self.cache_dir.exists():
                print(f"Extracting images to {self.cache_dir}...")
                self.cache_dir.mkdir(exist_ok=True)
                self._extract_subset()
    
    def _extract_subset(self):
        """Extract only the images we need"""
        needed_images = {conv['image'] for conv in self.conversations}
        
        with zipfile.ZipFile(self.images_zip, 'r') as zf:
            for img_name in needed_images:
                if img_name in zf.namelist():
                    try:
                        zf.extract(img_name, self.cache_dir)
                    except:
                        pass
    
    def _get_image(self, image_name: str):
        """Get image by name, from cache or zip"""
        # Try cache first
        if self.use_cache:
            img_path = self.cache_dir / image_name
            if img_path.exists():
                return Image.open(img_path).convert('RGB')
        
        # Try memory cache
        if image_name in self.image_cache:
            return self.image_cache[image_name]
        
        # Load from zip
        if self.zipfile is None:
            self.zipfile = zipfile.ZipFile(self.images_zip, 'r')
        
        try:
            with self.zipfile.open(image_name) as img_file:
                img_data = img_file.read()
                img = Image.open(io.BytesIO(img_data)).convert('RGB')
                
                # Cache in memory (limit cache size)
                if len(self.image_cache) < 100:
                    self.image_cache[image_name] = img
                
                return img
        except:
            # Return a blank image if not found
            return Image.new('RGB', (336, 336), color='white')
    
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        """Get a single sample"""
        conv = self.conversations[idx]
        
        # Get image
        image = self._get_image(conv['image'])
        
        # Process image
        if self.image_processor:
            if self.model_config and hasattr(self.model_config, 'image_aspect_ratio'):
                image_tensor = process_images([image], self.image_processor, self.model_config)[0]
            else:
                image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        else:
            image_tensor = torch.zeros(3, 336, 336)  # Dummy tensor
        
        # Process conversation
        conversations = conv['conversations']
        
        # Build input text (human question)
        human_text = conversations[0]['value'] if conversations else "Describe this image."
        
        # Add image token if not present
        if DEFAULT_IMAGE_TOKEN not in human_text:
            human_text = DEFAULT_IMAGE_TOKEN + '\n' + human_text
        
        # Build target text (assistant response)
        gpt_text = conversations[1]['value'] if len(conversations) > 1 else "A description of the image."
        
        # Tokenize if tokenizer is available
        if self.tokenizer:
            input_ids = tokenizer_image_token(
                human_text, 
                self.tokenizer, 
                IMAGE_TOKEN_INDEX,
                return_tensors='pt'
            )
        else:
            input_ids = torch.zeros(100, dtype=torch.long)  # Dummy
        
        return {
            'input_ids': input_ids,
            'image': image_tensor,
            'labels': gpt_text,  # For reference
            'image_name': conv['image'],
            'id': conv.get('id', f'sample_{idx}')
        }
    
    def __del__(self):
        """Clean up zip file"""
        if self.zipfile:
            self.zipfile.close()


def create_cc3m_dataloader(tokenizer, image_processor, model_config,
                           batch_size=1, num_samples=100, num_workers=0):
    """
    Create a dataloader for CC3M dataset
    
    Args:
        tokenizer: LLaVA tokenizer
        image_processor: LLaVA image processor
        model_config: Model configuration
        batch_size: Batch size for dataloader
        num_samples: Number of samples to use
        num_workers: Number of worker processes
    
    Returns:
        DataLoader for CC3M dataset
    """
    dataset = CC3MDataset(
        tokenizer=tokenizer,
        image_processor=image_processor,
        model_config=model_config,
        max_samples=num_samples,
        use_cache=True
    )
    
    # Custom collate function
    def collate_fn(batch):
        # Handle variable length inputs
        input_ids = [item['input_ids'] for item in batch]
        images = torch.stack([item['image'] for item in batch])
        
        # Pad input_ids
        max_len = max(ids.shape[0] for ids in input_ids)
        padded_input_ids = []
        attention_masks = []
        
        for ids in input_ids:
            pad_len = max_len - ids.shape[0]
            if pad_len > 0:
                padded_ids = torch.cat([ids, torch.zeros(pad_len, dtype=ids.dtype)])
                mask = torch.cat([torch.ones_like(ids), torch.zeros(pad_len, dtype=ids.dtype)])
            else:
                padded_ids = ids
                mask = torch.ones_like(ids)
            
            padded_input_ids.append(padded_ids)
            attention_masks.append(mask)
        
        return {
            'input_ids': torch.stack(padded_input_ids),
            'attention_mask': torch.stack(attention_masks),
            'images': images,
            'labels': [item['labels'] for item in batch],
            'ids': [item['id'] for item in batch]
        }
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return dataloader


def test_cc3m_loader():
    """Test the CC3M dataloader"""
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from llava.mm_utils import get_model_name_from_path
    
    model_path = "liuhaotian/llava-v1.5-7b"
    
    print("Loading model for testing...")
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name
    )
    
    print("\nCreating CC3M dataloader...")
    dataloader = create_cc3m_dataloader(
        tokenizer=tokenizer,
        image_processor=image_processor,
        model_config=model.config,
        batch_size=2,
        num_samples=10
    )
    
    print(f"\nDataloader created with {len(dataloader)} batches")
    
    # Test iteration
    for i, batch in enumerate(dataloader):
        print(f"\nBatch {i}:")
        print(f"  Input IDs shape: {batch['input_ids'].shape}")
        print(f"  Images shape: {batch['images'].shape}")
        print(f"  Labels: {batch['labels']}")
        
        if i >= 2:  # Just test a few batches
            break
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_cc3m_loader()