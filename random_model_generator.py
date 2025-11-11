import subprocess
import os
import sys
import math
import logging
import random

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
# ===================================================================
# --- Main Configuration ---
# ===================================================================
GEN_MODEL_SCRIPT = "gen_model.py"
OUTPUT_DIR = "random_cnn_models"
LOG_FILE = "random_build_log.txt"

# Number of random models to attempt to build
NUM_MODELS_TO_GENERATE = 25 

# ===================================================================
# --- Random Generation Specs ---
# ===================================================================

# Define the "universe" of parameters our generator can pick from.
# Feel free to add/remove options here to test different ranges.
MODEL_SPECS = {
    "min_layers": 2,
    "max_layers": 100,
    "min_3d_layers_before_flatten": 0.75,
    "1d_input_dims": [32,64,128, 256, 512],
    "3d_input_h": [64,128,256,512,1024], 
    "3d_input_w": [64,128,256,512,1024],
    "3d_input_c": [1, 3],
    "final_classifier_units": [2,10,50] # Num classes
}

LAYER_CHOICES = {
    # Ops available when current_shape is 3D
    "ops_3d": ["conv2d","depthwise_conv2d", "max_pool2d", "avg_pool2d", "flatten"],
    # Ops available when current_shape is 1D
    "ops_1d": ["fully_connected"],

    "conv2d": {
        "filters": [8, 16, 32, 64, 128],
        "kernel": [1, 3, 5],
        "stride": [1, 2],
        "padding": ["SAME", "VALID"],
        "activation": ["relu", ""],
        "bias": [0, 1]
    },
    "depthwise_conv2d": {
        "multiplier": [1, 2],
        "kernel": [3, 5],
        "stride": [1, 2],
        "padding": ["SAME", "VALID"],
        "activation": ["relu", ""],
        "bias": [0, 1]
    },
    "pool2d": { # Covers avg_pool2d and max_pool2d
        "pool_size": [2, 3],
        "stride": [1, 2],
        "padding": ["SAME", "VALID"]
    },
    "fully_connected": {
        "units": [16, 64, 128, 256, 512],
        "activation": ["relu", ""],
        "bias": [0, 1]
    }
}


# ===================================================================
# --- Logging Setup ---
# ===================================================================
try:
    log_file = open(LOG_FILE, "w")
    sys.stdout = log_file
    sys.stderr = log_file
except Exception as e:
    print(f"Failed to open log file: {e}")
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

# ===================================================================
# --- Helper Function (Unchanged from your script) ---
# ===================================================================
def calculate_conv_output_dim(input_dim, kernel, padding, stride):
    """Calculates the output dimension for a Conv or Pool layer."""
    if padding.upper() == 'SAME':
        return math.ceil(input_dim / stride)
    elif padding.upper() == 'VALID':
        return math.ceil((input_dim - kernel + 1) / stride)
    else:
        raise ValueError(f"Unsupported padding type: {padding}")

# ===================================================================
# --- NEW: Random Config Generator ---
# ===================================================================
def generate_random_config(model_index: int) -> dict:
    """
    Generates a single, VALID model configuration dictionary.
    This function tracks the output shape as it adds layers to ensure
    the resulting model is buildable.
    """
    config = {}
    layers = []
    
    model_name = f"random_model_{model_index:04d}"
    config["name"] = model_name
    
    print(f"\n--- Generating Config: {model_name} ---")

    # 1. --- Set up Input Shape ---
    shape_type = "3d" # random.choice(["1d", "3d"])
    config["shape_type"] = shape_type
    
    if shape_type == "3d":
        h = random.choice(MODEL_SPECS["3d_input_h"])
        w = random.choice(MODEL_SPECS["3d_input_w"])
        c = random.choice(MODEL_SPECS["3d_input_c"])
        config["input_shape"] = (h, w, c)
        current_shape = (h, w, c) # (h, w, c)
        is_1d_model = False
    else: # "1d"
        dim = random.choice(MODEL_SPECS["1d_input_dims"])
        config["input_shape"] = (dim,)
        current_shape = (dim,) # (dim,)
        is_1d_model = True
        
    print(f"  Input Shape ({shape_type}): {current_shape}")
    num_3d_layers_added = 0
    min_3d_layers = MODEL_SPECS.get("min_3d_layers_before_flatten", 0.5)
    # 2. --- Generate Body Layers ---
    num_layers = random.randint(MODEL_SPECS["min_layers"], MODEL_SPECS["max_layers"])
    min_3d_layers = int(min_3d_layers*num_layers)
    print("No Layers: ",num_layers, min_3d_layers)
    for i in range(num_layers):
        layer_config = {}
        
        # --- Pick a valid operation based on current state (1D or 3D) ---
        if is_1d_model:
            if not LAYER_CHOICES["ops_1d"]: # No more 1D ops defined
                 break
            layer_op = random.choice(LAYER_CHOICES["ops_1d"])
        else:
            available_ops = list(LAYER_CHOICES["ops_3d"]) 
            
            # Enforce minimum 3D layers before allowing flatten
            if num_3d_layers_added < min_3d_layers and "flatten" in available_ops:
                available_ops.remove("flatten")
                print(f"     ... (Note) 'flatten' removed (need {min_3d_layers} 3D layers, have {num_3d_layers_added})")

            if not available_ops:
                print("     ... (Dead End) No valid 3D ops left. Stopping.")
                break
                
            layer_op = random.choice(available_ops)
        
        layer_config["op"] = layer_op
        print(f"  Layer {i}: ({layer_op})...")
        
        try:
            # --- Generate random parameters for the chosen op ---
            if layer_op == "conv2d":
                h, w, c = current_shape
                choices = LAYER_CHOICES["conv2d"]
                
                padding = random.choice(choices["padding"])
                stride = random.choice(choices["stride"])
                
                # *** CRITICAL VALIDATION ***
                # For 'VALID' padding, kernel cannot be larger than input dim
                max_kernel_h = h if padding == "VALID" else max(choices["kernel"])
                max_kernel_w = w if padding == "VALID" else max(choices["kernel"])
                
                valid_kernels = [k for k in choices["kernel"] if k <= max_kernel_h and k <= max_kernel_w]
                if not valid_kernels:
                    print("     ... (Dead End) No valid kernel size. Stopping.")
                    break # Stop adding layers
                
                kernel = random.choice(valid_kernels)
                filters = random.choice(choices["filters"])
                
                layer_config["filters"] = filters
                layer_config["kernel"] = kernel
                layer_config["stride"] = stride
                layer_config["padding"] = padding
                layer_config["activation"] = random.choice(choices["activation"])
                layer_config["bias"] = random.choice(choices["bias"])
                
                # Update current shape
                h_out = calculate_conv_output_dim(h, kernel, padding, stride)
                w_out = calculate_conv_output_dim(w, kernel, padding, stride)
                current_shape = (h_out, w_out, filters)

            elif layer_op == "depthwise_conv2d":
                h, w, c = current_shape
                choices = LAYER_CHOICES["depthwise_conv2d"]
                
                padding = random.choice(choices["padding"])
                stride = random.choice(choices["stride"])

                # *** CRITICAL VALIDATION ***
                max_kernel_h = h if padding == "VALID" else max(choices["kernel"])
                max_kernel_w = w if padding == "VALID" else max(choices["kernel"])
                
                valid_kernels = [k for k in choices["kernel"] if k <= max_kernel_h and k <= max_kernel_w]
                if not valid_kernels:
                    print("     ... (Dead End) No valid kernel size. Stopping.")
                    break
                
                kernel = random.choice(valid_kernels)
                multiplier = random.choice(choices["multiplier"])
                
                layer_config["multiplier"] = multiplier
                layer_config["kernel"] = kernel
                layer_config["stride"] = stride
                layer_config["padding"] = padding
                layer_config["activation"] = random.choice(choices["activation"])
                layer_config["bias"] = random.choice(choices["bias"])
                
                # Update current shape
                h_out = calculate_conv_output_dim(h, kernel, padding, stride)
                w_out = calculate_conv_output_dim(w, kernel, padding, stride)
                current_shape = (h_out, w_out, c * multiplier)

            elif layer_op in ["avg_pool2d", "max_pool2d"]:
                h, w, c = current_shape
                choices = LAYER_CHOICES["pool2d"]
                
                padding = random.choice(choices["padding"])
                stride = random.choice(choices["stride"])

                # *** CRITICAL VALIDATION ***
                max_pool_h = h if padding == "VALID" else max(choices["pool_size"])
                max_pool_w = w if padding == "VALID" else max(choices["pool_size"])
                
                valid_pools = [p for p in choices["pool_size"] if p <= max_pool_h and p <= max_pool_w]
                if not valid_pools:
                    print("     ... (Dead End) No valid pool size. Stopping.")
                    break
                
                pool_size = random.choice(valid_pools)
                
                layer_config["pool_size"] = pool_size
                layer_config["stride"] = stride
                layer_config["padding"] = padding
                
                # Update current shape
                h_out = calculate_conv_output_dim(h, pool_size, padding, stride)
                w_out = calculate_conv_output_dim(w, pool_size, padding, stride)
                current_shape = (h_out, w_out, c)

            elif layer_op == "flatten":
                h, w, c = current_shape
                current_shape = (h * w * c,)
                is_1d_model = True

            elif layer_op == "fully_connected":
                choices = LAYER_CHOICES["fully_connected"]
                units = random.choice(choices["units"])
                
                layer_config["units"] = units
                layer_config["activation"] = random.choice(choices["activation"])
                layer_config["bias"] = random.choice(choices["bias"])
                
                # Update current shape
                current_shape = (units,)

            # Check for invalid output shape (e.g., 0-dimension)
            if any(d <= 0 for d in current_shape):
                 print(f"     ... (Dead End) Output shape {current_shape} is invalid. Stopping.")
                 break
            if layer_op in ["conv2d", "depthwise_conv2d", "max_pool2d", "avg_pool2d"]: # <--- NEW
                num_3d_layers_added += 1
            layers.append(layer_config)
            print(f"     ... OK. New shape: {current_shape}")

        except Exception as e:
            print(f"     ... (Error) Failed to generate layer: {e}. Stopping.")
            break # Stop adding layers on any error

    # 3. --- Add Final Head ---
    # Ensure the model ends with a classifier
    if not is_1d_model:
        print("  Adding final Flatten layer...")
        layers.append({"op": "flatten"})
        h, w, c = current_shape
        current_shape = (h * w * c,)

    print("  Adding final Fully_Connected (classifier) layer...")
    layers.append({
        "op": "fully_connected",
        "units": random.choice(MODEL_SPECS["final_classifier_units"]),
        "activation": "", # Logits
        "bias": 1
    })

    config["layers"] = layers
    return config

# ===================================================================
# --- Model Build Function (Unchanged from your script) ---
# ===================================================================
def build_model_from_config(config: dict, model_index: int) -> bool:
    """
    Attempts to build a single model from a fixed configuration.
    Returns True on success, False on failure.
    (This function is identical to your provided script)
    """
    model_name = config.get("name", f"model_{model_index}")
    model_filename = os.path.join(OUTPUT_DIR, f"{model_name}.tflite")
    
    shape_type = config["shape_type"]
    input_shape = config["input_shape"]
    layers = config["layers"]
    num_layers = len(layers)

    print("\n" + "="*70)
    print(f"ATTEMPTING BUILD: {model_name} ({num_layers} layers)")
    print(f"   File: {model_filename}")
    print("="*70)

    # ==============================================================
    #  Input Shape
    # ==============================================================
    if shape_type == "3d":
        h, w, c = input_shape
        in_shape_str = f"{h},{w},{c}"
        current_shape = (h, w, c)
        is_1d_model = False
    elif shape_type == "1d":
        (dim,) = input_shape
        in_shape_str = str(dim)
        current_shape = (dim,)
        is_1d_model = True
    else:
        print(f" FAILED: Invalid shape_type '{shape_type}' in config.")
        return False

    # ==============================================================
    # Layer generation (multi-layer model)
    # ==============================================================
    cmd_args = ["python3", GEN_MODEL_SCRIPT, "tflite", model_filename, in_shape_str, str(num_layers)]

    for i, layer_config in enumerate(layers):
        layer_op = layer_config["op"]
        print(f"  -> Layer {i} ({layer_op}): Input shape {current_shape}")

        try:
            # -----------------------------------------
            # Build layer attributes
            # -----------------------------------------
            if layer_op == "conv2d":
                filters = layer_config["filters"]
                kernel = layer_config["kernel"]
                stride = layer_config["stride"]
                padding = layer_config["padding"].upper()
                activation = layer_config["activation"]
                use_bias = layer_config["bias"]
                
                h, w, _ = current_shape
                if padding == "VALID":
                    if h < kernel or w < kernel:
                        print(f"   Validation Error: 'VALID' padding with kernel {kernel} is too large for input {h}x{w}.")
                        raise ValueError("Invalid layer parameters")
                
                attrs = f"{filters},{kernel},{kernel},{stride},{stride},1,1,{padding},channels_last,{activation},{use_bias},1"
                cmd_args.extend(["conv2d", attrs])
                
                current_shape = (
                    calculate_conv_output_dim(h, kernel, padding, stride),
                    calculate_conv_output_dim(w, kernel, padding, stride),
                    filters
                )

            elif layer_op == "depthwise_conv2d":
                multiplier = layer_config["multiplier"]
                kernel = layer_config["kernel"]
                stride = layer_config["stride"]
                padding = layer_config["padding"].upper()
                activation = layer_config["activation"]
                use_bias = layer_config["bias"]

                h, w, c = current_shape
                if padding == "VALID":
                    if h < kernel or w < kernel:
                        print(f"   Validation Error: 'VALID' padding with kernel {kernel} is too large for input {h}x{w}.")
                        raise ValueError("Invalid layer parameters")

                attrs = f"{multiplier},{kernel},{kernel},{stride},{stride},1,1,{padding},channels_last,{activation},{use_bias}"
                cmd_args.extend(["depthwise_conv2d", attrs])
                
                current_shape = (
                    calculate_conv_output_dim(h, kernel, padding, stride),
                    calculate_conv_output_dim(w, kernel, padding, stride),
                    c * multiplier
                )

            elif layer_op in ["avg_pool2d", "max_pool2d"]:
                pool_size = layer_config["pool_size"]
                stride = layer_config["stride"]
                padding = layer_config["padding"].upper()

                h, w, c = current_shape
                if padding == "VALID":
                    if h < pool_size or w < pool_size:
                        print(f"   Validation Error: 'VALID' padding with pool_size {pool_size} is too large for input {h}x{w}.")
                        raise ValueError("Invalid layer parameters")
                        
                attrs = f"{pool_size},{pool_size},{stride},{stride},{padding},channels_last"
                cmd_args.extend([layer_op, attrs])
                
                current_shape = (
                    calculate_conv_output_dim(h, pool_size, padding, stride),
                    calculate_conv_output_dim(w, pool_size, padding, stride),
                    c
                )

            elif layer_op == "flatten":
                cmd_args.extend(["flatten", "_"])
                if len(current_shape) == 3: # Handle 3D
                    h, w, c = current_shape
                    current_shape = (h * w * c,)
                # if already 1D, shape is unchanged
                is_1d_model = True

            elif layer_op == "fully_connected":
                units = layer_config["units"]
                activation = layer_config["activation"]
                use_bias = layer_config["bias"]
                
                attrs = f",{units},{activation},{use_bias}"
                cmd_args.extend(["fully_connected", attrs])
                current_shape = (units,)

            else:
                print(f"   Validation Error: Unsupported layer type: {layer_op}")
                raise ValueError("Unsupported layer")
                
            print(f"     ... Layer {i} OK. Output shape {current_shape}")

        except KeyError as e:
            print(f" FAILED (Config Error): Missing parameter {e} for layer {i} ({layer_op}).")
            return False
        except ValueError as e:
            print(f" FAILED (Validation Error): {e} for layer {i} ({layer_op}).")
            return False

    # ==============================================================
    #  Execute
    # ==============================================================
    print("\n" + "-"*60)
    print(f"Executing model build for: {model_name}")
    # Print the command as a single string for easy copying/debugging
    print(" ".join(map(str, cmd_args)))
    print("-"*60)
    
    try:
        # Execute the model generation script
        subprocess.run(cmd_args, check=True, capture_output=True, text=True)
        
        print(f" SUCCESS: Model generated successfully: {model_filename}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f" FAILED (Subprocess Error): Model generation FAILED for {model_name}.")
        print(f"   Return Code: {e.returncode}")
        print(f"   Stdout: {e.stdout.strip()}")
        print(f"   Stderr: {e.stderr.strip()}")
        return False
    except Exception as e:
        print(f" FAILED (Unknown Error): An unexpected error occurred: {e}")
        return False

# ===================================================================
# --- NEW: Main Function (Randomized) ---
# ===================================================================
def main():
    if not os.path.exists(GEN_MODEL_SCRIPT):
        print(f"Error: The script '{GEN_MODEL_SCRIPT}' was not found.")
        print("Please make sure 'gen_model.py' is in the same directory.")
        sys.exit(1)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Starting RANDOM model build process...")
    print(f"Attempting to generate and build {NUM_MODELS_TO_GENERATE} models.")
    
    success_count = 0
    fail_count = 0
    
    # --- Loop and generate random configs ---
    for i in range(NUM_MODELS_TO_GENERATE):
        # 1. Generate a valid config
        config = generate_random_config(i)
        
        # 2. Try to build it
        success = build_model_from_config(config, i)
        
        if success:
            success_count += 1
        else:
            fail_count += 1
            
    # --- Report final results ---
    print("\n" + "="*70)
    
    print("="*70)
    print(f"  Total Configurations Attempted: {NUM_MODELS_TO_GENERATE}")
    print(f"   Successful Builds: {success_count}")
    print(f"   Failed Builds:     {fail_count}")
    print(f"\nLog file saved to: {os.path.abspath(log_file.name)}")


if __name__ == "__main__":
    main()
    # Explicitly close the log file
    if sys.stdout != sys.__stdout__:
        print(f"Build complete. See {LOG_FILE} for details.")
        sys.stdout.close()