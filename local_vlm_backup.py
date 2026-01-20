# utils/local_vlm.py - Enhanced ONNX Runtime GenAI version for Phi-3.5-vision-instruct-onnx
# Optimized for Nepalese flora and fauna identification with tourism-focused descriptions

import onnxruntime_genai as og
from PIL import Image
import io
import os
import traceback
import logging
import time
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# IMPORTANT: This must point exactly to the folder containing genai_config.json
# From your tree: ./phi35-onnx-gpu/gpu/gpu-int4-rtn-block-32
MODEL_DIR = "./gpu-int4-rtn-block-32"

# Nepalese flora and fauna knowledge base
NEPAL_KNOWLEDGE_BASE = {
    "flora": {
        "rhododendron": {
            "scientific": "Rhododendron arboreum",
            "local_name": "Lali Gurans",
            "region": "Himalayan (3000-4000m)",
            "significance": "National flower of Nepal",
            "tourism_info": "Blooms in spring (March-May), found in Langtang, Annapurna, Everest regions"
        },
        "orchid": {
            "scientific": "Orchidaceae family",
            "local_name": "Rani Sunakhari",
            "region": "Terai to Hilly (500-2000m)",
            "significance": "Over 386 species in Nepal",
            "tourism_info": "Best viewed in Godavari Botanical Garden, Shivapuri National Park"
        },
        "pines": {
            "scientific": "Pinus wallichiana",
            "local_name": "Salla",
            "region": "Hilly to Himalayan (1500-3500m)",
            "significance": "Important for local communities",
            "tourism_info": "Common in Kathmandu valley hills, Nagarjun forest"
        }
    },
    "fauna": {
        "bengal_tiger": {
            "scientific": "Panthera tigris tigris",
            "local_name": "Bagh",
            "region": "Terai (Chitwan, Bardia, Shuklaphanta)",
            "conservation": "Endangered, ~200 in Nepal",
            "tourism_info": "Best seen in Chitwan National Park, Bardia National Park"
        },
        "one_horned_rhino": {
            "scientific": "Rhinoceros unicornis",
            "local_name": "Gainda",
            "region": "Terai (Chitwan, Bardia)",
            "conservation": "Vulnerable, ~752 in Nepal",
            "tourism_info": "Elephant safari in Chitwan offers best viewing experience"
        },
        "snow_leopard": {
            "scientific": "Panthera uncia",
            "local_name": "Him Chituwa",
            "region": "Himalayan (3000-5500m)",
            "conservation": "Vulnerable, ~300-500 in Nepal",
            "tourism_info": "Rare sightings in Upper Dolpo, Annapurna, Everest regions"
        },
        "red_panda": {
            "scientific": "Ailurus fulgens",
            "local_name": "Haley Rato",
            "region": "Hilly (2200-4800m)",
            "conservation": "Endangered",
            "tourism_info": "Found in Langtang, Rara, Makalu-Barun national parks"
        },
        "himalayan_monal": {
            "scientific": "Lophophorus impejanus",
            "local_name": "Danphe",
            "region": "Himalayan (2400-4500m)",
            "significance": "National bird of Nepal",
            "tourism_info": "Commonly seen in Annapurna, Langtang trekking routes"
        }
    }
}

# Quick existence check
if not os.path.exists(os.path.join(MODEL_DIR, "genai_config.json")):
    raise FileNotFoundError(
        f"ONNX GenAI model files not found in {MODEL_DIR}\n"
        "Make sure you have: genai_config.json, *.onnx, tokenizer.json, etc."
    )

logger.info(f"Loading Phi-3.5-vision ONNX model from: {MODEL_DIR}")
model = og.Model(MODEL_DIR)

# Correct way to create the multimodal (vision + text) processor
processor = model.create_multimodal_processor()

tokenizer = og.Tokenizer(model)

logger.info("ONNX Phi-3.5-vision loaded successfully!")

def get_nepal_context(object_name: str) -> Optional[Dict[str, Any]]:
    """Get Nepalese context for flora and fauna"""
    object_name_lower = object_name.lower().replace(" ", "_")
    
    # Search in flora
    for key, value in NEPAL_KNOWLEDGE_BASE["flora"].items():
        if object_name_lower in key or key in object_name_lower:
            return value
    
    # Search in fauna
    for key, value in NEPAL_KNOWLEDGE_BASE["fauna"].items():
        if object_name_lower in key or key in object_name_lower:
            return value
    
    return None

def describe_image_local(image_bytes: bytes, max_new_tokens: int = 220) -> str:
    """
    Describe an image using Phi-3.5-vision in engaging style + factual part.
    Optimized for general use with performance improvements.
    """
    start_time = time.time()
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        prompt = '''Describe what's in this image in an engaging, interesting style. 
        Be informative and slightly dramatic, but keep the explanation clean and safe. 
        After the engaging description, also give a factual, accurate description of what the objects in the image are, 
        including their purpose, materials, colors, and what the person can learn from it.'''

        # Chat-style messages (Phi-3.5 vision uses <|image_1|> token)
        messages = [{"role": "user", "content": f"<|image_1|>\n{prompt}"}]

        # Process text + image → creates the proper inputs
        inputs = processor(messages, images=img)

        # Generator setup with optimized parameters
        params = og.GeneratorParams(model)
        params.set_search_options(
            max_length=inputs.input_ids.shape[1] + max_new_tokens,
            temperature=0.1,          # Slightly creative but consistent
            top_k=50,
            top_p=0.9,
        )
        params.input_ids = inputs.input_ids
        params.pixel_values = inputs.pixel_values  # Vision embeddings go here

        generator = og.Generator(model, params)

        # Generate tokens efficiently
        output_tokens = []
        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()
            output_tokens.append(generator.get_next_tokens()[0])

        # Decode **only the newly generated part**
        response = tokenizer.decode(output_tokens)
        
        generation_time = time.time() - start_time
        logger.info(f"Generated description in {generation_time:.2f}s")

        return response.strip() or "No description could be generated."

    except Exception as e:
        logger.error(f"Description generation error: {str(e)}")
        traceback.print_exc()
        return f"Error during description: {str(e)}"

def describe_image_nepal_flora_fauna(image_bytes: bytes, max_new_tokens: int = 300) -> str:
    """
    Enhanced description specifically for Nepalese flora and fauna identification.
    Includes tourism information and local context.
    """
    start_time = time.time()
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        prompt = '''You are an expert guide specializing in Nepalese flora and fauna. 
        Analyze this image and provide a comprehensive description that includes:
        
        1. **Engaging Introduction**: Start with an interesting hook that would appeal to tourists
        2. **Identification**: Clearly identify the species if recognizable
        3. **Local Context**: Include Nepali/local names if known
        4. **Geographic Distribution**: Mention specific regions in Nepal (Himalayan, Hilly, Terai)
        5. **Tourism Information**: Where and when tourists can best see this species
        6. **Conservation Status**: Any conservation concerns or protected status
        7. **Cultural Significance**: Any importance to local communities or Nepali culture
        8. **Practical Tips**: How tourists can respectfully observe or photograph
        
        Format your response in two parts:
        - First part: Engaging, tourist-friendly description
        - Second part: Detailed factual information with practical tourism guidance
        
        Be accurate, informative, and engaging for international tourists visiting Nepal.'''

        # Chat-style messages
        messages = [{"role": "user", "content": f"<|image_1|>\n{prompt}"}]

        # Process text + image
        inputs = processor(messages, images=[img])

        # Generator setup with more creative parameters for tourism content
        params = og.GeneratorParams(model)
        params.set_search_options(
            max_length=inputs.input_ids.shape[1] + max_new_tokens,
            temperature=0.3,          # More creative for engaging tourism content
            top_k=50,
            top_p=0.9,
        )
        params.input_ids = inputs.input_ids
        params.pixel_values = inputs.pixel_values

        generator = og.Generator(model, params)

        # Generate tokens
        output_tokens = []
        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()
            output_tokens.append(generator.get_next_tokens()[0])

        # Decode response
        response = tokenizer.decode(output_tokens)
        
        generation_time = time.time() - start_time
        logger.info(f"Generated Nepal-focused description in {generation_time:.2f}s")

        return response.strip() or "No description could be generated."

    except Exception as e:
        logger.error(f"Nepal description generation error: {str(e)}")
        traceback.print_exc()
        return f"Error during description: {str(e)}"

def enhance_with_nepal_context(description: str, object_name: str) -> str:
    """Enhance description with Nepalese context if applicable"""
    context = get_nepal_context(object_name)
    if not context:
        return description
    
    # Add Nepalese context to the description
    enhanced = description + "\n\n"
    
    if "scientific" in context:
        enhanced += f"**Scientific Name**: {context['scientific']}\n"
    
    if "local_name" in context:
        enhanced += f"**Local Name**: {context['local_name']}\n"
    
    if "region" in context:
        enhanced += f"**Region**: {context['region']}\n"
    
    if "significance" in context:
        enhanced += f"**Significance**: {context['significance']}\n"
    
    if "conservation" in context:
        enhanced += f"**Conservation**: {context['conservation']}\n"
    
    if "tourism_info" in context:
        enhanced += f"**Tourism Info**: {context['tourism_info']}\n"
    
    return enhanced

# Performance monitoring
class PerformanceMonitor:
    def __init__(self):
        self.request_count = 0
        self.total_time = 0
    
    def log_request(self, duration: float):
        self.request_count += 1
        self.total_time += duration
        avg_time = self.total_time / self.request_count
        logger.info(f"Request #{self.request_count}: {duration:.2f}s (avg: {avg_time:.2f}s)")

perf_monitor = PerformanceMonitor()

# Quick standalone test
if __name__ == "__main__":
    test_img = Image.new("RGB", (512, 512), color=(100, 150, 200))
    buf = io.BytesIO()
    test_img.save(buf, format="PNG")
    
    print("=== General Description Test ===")
    print(describe_image_local(buf.getvalue()))
    
    print("\n=== Nepal-focused Description Test ===")
    print(describe_image_nepal_flora_fauna(buf.getvalue()))