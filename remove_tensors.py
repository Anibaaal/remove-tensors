import safetensors.torch
import torch
import os

def filter_tensors(input_file, output_file, whitelist_substrings=None, blacklist_substrings=None):
    # Load the safetensors file
    tensor_dict = safetensors.torch.load_file(input_file)
    
    # Filter tensors:
    filtered_tensors = {}

    for name, tensor in tensor_dict.items():
        # Apply whitelist if provided
        whitelisted = any(substr in name for substr in whitelist_substrings) if whitelist_substrings else False
        blacklisted = any(substr in name for substr in blacklist_substrings) if blacklist_substrings else False
        
        # If a name is whitelisted but also blacklisted, it should be removed
        if whitelisted and not blacklisted:
            filtered_tensors[name] = tensor
        # If no whitelist is provided, apply blacklist only
        elif not whitelist_substrings and not blacklisted:
            filtered_tensors[name] = tensor
    
    print(f"{len(filtered_tensors)} tensors kept out of {len(tensor_dict)}.")
    
    # Save the filtered tensors to a new safetensors file
    safetensors.torch.save_file(filtered_tensors, output_file)
    print(f"Filtered tensors saved to {output_file}.")

# Example usage
if __name__ == "__main__":
    input_file = "C:/Path/to/input_lora.safetensors"  # Replace with the path to your input LoRA safetensors file
    output_file = "C:/Path/to/output_filtered_lora.safetensors"  # Replace with the path for the output file

    # List of substrings for whitelisting and blacklisting
    whitelist_substrings = ["transformer.transformer_blocks.0","transformer.transformer_blocks.7","transformer.transformer_blocks.12","transformer.single_transformer_blocks.2","transformer.single_transformer_blocks.7","transformer.single_transformer_blocks.8","transformer.single_transformer_blocks.9","transformer.single_transformer_blocks.10","transformer.single_transformer_blocks.11","transformer.single_transformer_blocks.13","transformer.single_transformer_blocks.14","transformer.single_transformer_blocks.20","transformer.single_transformer_blocks.35","transformer.single_transformer_blocks.37"]  # If provided, only tensors with these names will be kept
    blacklist_substrings = []  # Blacklist will remove these even if they're in the whitelist

    filter_tensors(input_file, output_file, whitelist_substrings, blacklist_substrings)
