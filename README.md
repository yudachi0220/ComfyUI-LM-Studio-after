# ComfyUI LM Studio Node

forked from gabe-init/ComfyUI-LM-Studio


## Features

- **Unified Interface**: Single node for both text-only and text+image inputs
- **Dual Mode Support**: Works with both LM Studio SDK (recommended) and REST API
- **Vision Model Support**: Process images alongside text prompts with compatible models
- **Real-time Statistics**: Monitor tokens per second, input/output token counts
- **Thinking Tokens**: Optional support for models that use thinking tokens
- **MAX_TOKEN**: Update Max_Token to 256K
- **Flexible Configuration**: Adjust temperature, max tokens, seed, and server settings
- **Random Seed Support**: Control reproducibility with seed parameter (0 = random)
- **Model Unloading**: Dedicated node to free VRAM by unloading models from memory
- **Debug Mode**: Built-in debugging for troubleshooting

