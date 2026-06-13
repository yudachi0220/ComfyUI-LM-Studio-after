import requests
import json
import re
import os
import tempfile
import time
import numpy as np
from urllib.parse import urlparse
from PIL import Image
from typing import Optional, Dict, Tuple, Any

# Try to import LM Studio SDK
try:
    import lmstudio as lms
    HAS_SDK = True
    print("LM Studio SDK found and loaded")
except ImportError:
    lms = None
    HAS_SDK = False
    print("LM Studio SDK not found. Using API fallback. Install SDK with: pip install lmstudio")


class LMStudioNode:
    """
    Unified LM Studio Chat Interface with Vision Support
    Supports both text-only and text+image inputs
    """

    def __init__(self):
        self.default_stats = "Tokens per Second: 0.00\nInput Tokens: 0\nOutput Tokens: 0"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are a helpful assistant."
                }),
                "user_message": ("STRING", {
                    "multiline": True,
                    "default": "Explain quantum computing in simple terms"
                }),
                "model_id": ("STRING", {
                    "multiline": False,
                    "default": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
                }),
                "server_address": ("STRING", {
                    "multiline": False,
                    "default": "http://127.0.0.1:1234"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "max_tokens": ("INT", {
                    "default": 1000,
                    "min": 1,
                    "max": 256000,
                    "step": 1
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "display": "number"
                }),
                "thinking_tokens": ("BOOLEAN", {
                    "default": True,
                    "label": "Include thinking tokens"
                }),
                "use_sdk": ("BOOLEAN", {
                    "default": True,
                    "label": "Use SDK (if available)"
                }),
                "unload_after_use": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Unload the model after processing to free VRAM"
                }),
            },
            "optional": {
                "image": ("IMAGE",),
                "debug": ("BOOLEAN", {
                    "default": False
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response", "stats")
    FUNCTION = "get_response"
    CATEGORY = "LM Studio"

    def _clean_thinking_tokens(self, text: str) -> str:
        """Remove thinking tokens from text if needed"""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def _prepare_image(self, image: np.ndarray, debug: bool = False) -> Optional[str]:
        """Convert numpy array to temporary file and return path"""
        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(np.uint8(image[0] * 255))
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_path = temp_file.name
                pil_image.save(temp_path, format="JPEG")
            
            if debug:
                print(f"Debug: Saved image to temporary file: {temp_path}")
            
            return temp_path
        except Exception as e:
            if debug:
                print(f"Debug: Error preparing image: {str(e)}")
            return None

    def _format_stats(self, tokens_per_sec: float, input_tokens: int, output_tokens: int) -> str:
        """Format statistics output"""
        return (
            f"Tokens per Second: {tokens_per_sec:.2f}\n"
            f"Input Tokens: {input_tokens}\n"
            f"Output Tokens: {output_tokens}"
        )

    def get_response(self, system_prompt: str, user_message: str, model_id: str, 
                    server_address: str, temperature: float, max_tokens: int,
                    seed: int, thinking_tokens: bool, use_sdk: bool = True,
                    unload_after_use: bool = False,
                    image: Optional[np.ndarray] = None, debug: bool = False) -> Tuple[str, str]:
        """Main entry point for getting LM Studio response"""
        
        # Clean message content if thinking tokens disabled
        if not thinking_tokens:
            user_message = self._clean_thinking_tokens(user_message)

        # Debug information
        if debug:
            print(f"Debug: Starting get_response method")
            print(f"Debug: Model: {model_id}")
            print(f"Debug: Use SDK: {use_sdk}")
            print(f"Debug: Has image: {image is not None}")
            if image is not None:
                print(f"Debug: Image shape: {image.shape}")

        # Route to appropriate method
        if HAS_SDK and use_sdk:
            result = self._get_response_sdk(
                system_prompt, user_message, model_id, temperature, 
                max_tokens, seed, thinking_tokens, image, debug
            )
        else:
            if image is not None and debug:
                print("Warning: Image input is not supported with API mode. Install LM Studio SDK for image support.")
            result = self._get_response_api(
                system_prompt, user_message, model_id, server_address, 
                temperature, seed, thinking_tokens, debug
            )
        
        # Unload model after use if requested (Pirogs-Nodes style)
        if unload_after_use and model_id:
            if debug:
                print(f"Debug: Unloading model after use: {model_id}")
            self._unload_model(server_address, model_id)
        
        return result

    def _get_response_sdk(self, system_prompt: str, user_message: str, model_id: str,
                         temperature: float, max_tokens: int, seed: int, thinking_tokens: bool,
                         image: Optional[np.ndarray] = None, debug: bool = False) -> Tuple[str, str]:
        """Use the LM Studio SDK to get a response"""
        temp_path = None
        
        try:
            # Load the model
            start_time = time.time() if debug else 0
            model = lms.llm(model_id)
            
            if debug:
                print(f"Debug: Model loaded in {time.time() - start_time:.2f}s")
            
            # Create a new chat
            chat = lms.Chat(system_prompt)
            
            # Handle image if provided
            if image is not None:
                temp_path = self._prepare_image(image, debug)
                if temp_path:
                    image_handle = lms.prepare_image(temp_path)
                    chat.add_user_message(user_message, images=[image_handle])
                    if debug:
                        print(f"Debug: Added image to chat message")
                else:
                    # Fallback to text-only if image preparation failed
                    chat.add_user_message(user_message)
            else:
                chat.add_user_message(user_message)
            
            # Configure generation parameters
            config = {
                "temperature": temperature,
                "maxTokens": max_tokens,
                "seed": seed,
            }
            
            if debug:
                print(f"Debug: Sending request with config: {config}")
            
            # Generate response
            result = model.respond(chat, config=config)
            
            if debug:
                print(f"Debug: Response received: {result.content[:100]}...")
                print(f"Debug: Generation time: {result.stats.generation_time_sec}s")
            
            # Extract statistics
            tokens_per_sec = getattr(result.stats, 'tokens_per_second', 0.0)
            input_tokens = getattr(result.stats, 'prompt_tokens_count', 0)
            output_tokens = getattr(result.stats, 'predicted_tokens_count', 0)
            
            # Format output
            output = result.content
            if not thinking_tokens:
                output = self._clean_thinking_tokens(output)
            
            stats_str = self._format_stats(tokens_per_sec, input_tokens, output_tokens)
            
            return (output, stats_str)
            
        except Exception as e:
            error_message = f"Error processing with LM Studio SDK: {str(e)}"
            print(error_message)
            return (error_message, self.default_stats)
        finally:
            # Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
                if debug:
                    print(f"Debug: Removed temporary file: {temp_path}")

    def _get_response_api(self, system_prompt: str, user_message: str, model_id: str,
                         server_address: str, temperature: float, seed: int, thinking_tokens: bool,
                         debug: bool = False) -> Tuple[str, str]:
        """Use the LM Studio API to get a response (text-only)"""
        headers = {"Content-Type": "application/json"}

        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "temperature": temperature,
            "seed": seed,
            "stream": False
        }

        try:
            response = requests.post(
                f"{server_address}/api/v0/chat/completions",
                headers=headers,
                data=json.dumps(payload),
                timeout=120
            )
            response.raise_for_status()

            result = response.json()

            # Extract main response
            output = result['choices'][0]['message']['content']
            if not thinking_tokens:
                output = self._clean_thinking_tokens(output)

            # Extract statistics
            usage = result.get('usage', {})
            stats = result.get('stats', {})
            
            tokens_per_sec = stats.get("tokens_per_second", 0.0)
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

            stats_str = self._format_stats(tokens_per_sec, input_tokens, output_tokens)

            return (output, stats_str)

        except requests.ConnectionError:
            return (f"Connection error - is LM Studio running at {server_address}?", self.default_stats)
        except requests.Timeout:
            return ("Request timed out - try increasing timeout duration", self.default_stats)
        except Exception as e:
            return (f"Error: {str(e)}", self.default_stats)

    def _unload_model(self, server_address: str, model_name: str) -> bool:
        """Unload model using LM Studio SDK (Pirogs-Nodes style)."""
        if not HAS_SDK:
            print(f"Cannot unload model {model_name}: LM Studio SDK not available. Install with: pip install lmstudio")
            return False
        try:
            parsed_url = urlparse(server_address)
            api_host = f"{parsed_url.hostname}:{parsed_url.port}"
            client = lms.Client(api_host=api_host)
            client.llm.unload(model_name)
            print(f"Successfully unloaded model via SDK: {model_name}")
            return True
        except Exception as e:
            print(f"SDK unload failed for {model_name}: {e}")
            return False


class LMStudioUnloadNode:
    """Node to unload a model from LM Studio, freeing VRAM/resources."""
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "server_address": ("STRING", {
                    "multiline": False,
                    "default": "http://127.0.0.1:1234"
                }),
                "model_id": ("STRING", {
                    "multiline": False,
                    "default": ""
                }),
                "use_sdk": ("BOOLEAN", {
                    "default": True,
                    "label": "Use SDK (if available)"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "unload_model"
    CATEGORY = "LM Studio"
    
    def unload_model(self, server_address: str, model_id: str, 
                     use_sdk: bool = True) -> Tuple[str]:
        """Unload a model from LM Studio memory."""
        
        if HAS_SDK and use_sdk:
            return self._unload_sdk(server_address, model_id)
        else:
            return self._unload_api(server_address, model_id)
    
    def _unload_sdk(self, server_address: str, model_id: str) -> Tuple[str]:
        """Unload via LM Studio SDK (Pirogs-Nodes style Client pattern)."""
        if not HAS_SDK:
            return ("LM Studio SDK not available. Install with: pip install lmstudio",)
        try:
            parsed_url = urlparse(server_address)
            api_host = f"{parsed_url.hostname}:{parsed_url.port}"
            client = lms.Client(api_host=api_host)
            if model_id:
                client.llm.unload(model_id)
            else:
                client.llm.unload()
            return (f"Model unloaded successfully: {model_id or 'current model'}",)
        except Exception as e:
            return (f"SDK unload error: {str(e)}",)
    
    def _unload_api(self, server_address: str, model_id: str) -> Tuple[str]:
        """Unload via LM Studio REST API."""
        try:
            payload = {}
            if model_id:
                payload["instance_id"] = model_id
            
            response = requests.post(
                f"{server_address}/api/v1/models/unload",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                instance = result.get("instance_id", model_id or "all models")
                return (f"Model unloaded successfully: {instance}",)
            else:
                return (f"Unload failed: HTTP {response.status_code} - {response.text}",)
                
        except requests.ConnectionError:
            return (f"Connection error - is LM Studio running at {server_address}?",)
        except requests.Timeout:
            return ("Request timed out during unload",)
        except Exception as e:
            return (f"Unload error: {str(e)}",)


# Node registration
NODE_CLASS_MAPPINGS = {
    "LMStudioNode": LMStudioNode,
    "LMStudioUnloadNode": LMStudioUnloadNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LMStudioNode": "LM Studio Chat Interface",
    "LMStudioUnloadNode": "LM Studio Unload Model"
}
