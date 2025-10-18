import optuna
import subprocess
import os
import sys
import math
import logging
from functools import partial

# ===================================================================
# --- Main Configuration ---
# ===================================================================
GEN_MODEL_SCRIPT = "gen_model.py"
NUM_TRIALS_PER_STUDY = 10  # Number of trials for EACH combination
OUTPUT_DIR = "generated_models"

# ✅ DEFINE THE PARAMETERS TO LOOP THROUGH
SHAPE_TYPES_TO_TEST = ["3d", "1d"]
MAX_LAYERS_TO_TEST = list(range(16,65,16))

# --- Setup Logging ---
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- Helper Function (Unchanged) ---
def calculate_conv_output_dim(input_dim, kernel, padding, stride):
    if padding.upper() == 'SAME':
        return math.ceil(input_dim / stride)
    elif padding.upper() == 'VALID':
        return math.ceil((input_dim - kernel + 1) / stride)
    else:
        raise ValueError(f"Unsupported padding type: {padding}")

# ✅ MODIFIED objective function to accept config arguments
def objective(trial: optuna.trial.Trial, max_layers: int, shape_type: str):
    """
    Optuna objective function with a static search space, configured by arguments.
    """
    model_filename = os.path.join(OUTPUT_DIR, f"model_trial_{trial.study.study_name}_{trial.number}.tflite")

    # ===================================================================
    # 1. DEFINE THE COMPLETE, STATIC SEARCH SPACE
    # ===================================================================
    
    # --- Input Shape Definition (uses the `shape_type` argument) ---
    if shape_type == "3d":
        h = trial.suggest_categorical("in_height", [16, 32, 64, 96, 128, 256])
        w = trial.suggest_categorical("in_width", [16, 32, 64, 96, 128, 256])
        c = trial.suggest_categorical("in_channels", [1, 3, 16])
        initial_shape = (h, w, c)
        in_shape_str = f"{h},{w},{c}"
    else: # 1d
        dim = trial.suggest_categorical("in_dim", [256, 512, 1024, 2048])
        initial_shape = (dim,)
        in_shape_str = str(dim)

    # --- Number of layers to *actually use* in this trial ---
    # This allows exploring models of different depths within the same study.
    num_layers = max_layers # trial.suggest_int("num_layers", 1, max_layers)
    
    # --- Define hyperparameters for ALL potential layers up to the study's max_layers ---
    all_3d_ops = ["conv2d", "depthwise_conv2d", "avg_pool2d", "max_pool2d", "flatten"]
    for i in range(max_layers):
        trial.suggest_categorical(f"l{i}_op", all_3d_ops + ["fully_connected"])
        trial.suggest_int(f"l{i}_conv_filters", 8, 64)
        trial.suggest_categorical(f"l{i}_conv_kernel", [1, 3, 5])
        trial.suggest_categorical(f"l{i}_conv_stride", [1, 2])
        trial.suggest_categorical(f"l{i}_dw_multiplier", [1, 2, 4])
        trial.suggest_categorical(f"l{i}_dw_kernel", [3, 5])
        trial.suggest_categorical(f"l{i}_dw_stride", [1, 2])
        trial.suggest_categorical(f"l{i}_padding", ["SAME", "VALID"])
        trial.suggest_categorical(f"l{i}_activation", ["relu", ""])
        trial.suggest_categorical(f"l{i}_bias", [0, 1])
        trial.suggest_categorical(f"l{i}_pool_size", [2, 3])
        trial.suggest_categorical(f"l{i}_pool_stride", [1, 2])
        trial.suggest_int(f"l{i}_fc_units", 10, 512)

    # ===================================================================
    # 2. CONSTRUCT THE MODEL USING THE CHOSEN HYPERPARAMETERS
    # ===================================================================
    cmd_args = ["python3", GEN_MODEL_SCRIPT, "tflite", model_filename, in_shape_str, str(num_layers)]
    current_shape = initial_shape
    is_1d_model = (len(current_shape) == 1)

    # Loop only up to the `num_layers` chosen for this trial
    for i in range(num_layers):
        layer_op = trial.params[f"l{i}_op"]

        # --- Validate the chosen layer op ---
        if is_1d_model and layer_op != "fully_connected":
            raise optuna.exceptions.TrialPruned(f"Layer {i} must be fully_connected after flatten/1D input.")
        if not is_1d_model and layer_op == "fully_connected":
             raise optuna.exceptions.TrialPruned(f"Cannot use fully_connected on 3D data without a flatten layer.")
        if i == num_layers - 1 and layer_op == "flatten":
            raise optuna.exceptions.TrialPruned("Cannot use 'flatten' as the final layer.")

        # --- Build attributes for the chosen layer op (logic is unchanged) ---
        if layer_op == "conv2d":
            # (Attribute gathering logic...)
            filters, kernel, stride = trial.params[f"l{i}_conv_filters"], trial.params[f"l{i}_conv_kernel"], trial.params[f"l{i}_conv_stride"]
            padding, activation, use_bias = trial.params[f"l{i}_padding"], trial.params[f"l{i}_activation"], trial.params[f"l{i}_bias"]
            attrs = f"{filters},{kernel},{kernel},{stride},{stride},1,1,{padding},channels_last,{activation},{use_bias},1"
            cmd_args.extend([layer_op, attrs])
            h, w, _ = current_shape
            current_shape = (calculate_conv_output_dim(h, kernel, padding, stride),
                             calculate_conv_output_dim(w, kernel, padding, stride),
                             filters)
        elif layer_op == "depthwise_conv2d":
            # (Attribute gathering logic...)
            multiplier, kernel, stride = trial.params[f"l{i}_dw_multiplier"], trial.params[f"l{i}_dw_kernel"], trial.params[f"l{i}_dw_stride"]
            padding, activation, use_bias = trial.params[f"l{i}_padding"], trial.params[f"l{i}_activation"], trial.params[f"l{i}_bias"]
            attrs = f"{multiplier},{kernel},{kernel},{stride},{stride},1,1,{padding},channels_last,{activation},{use_bias}"
            cmd_args.extend([layer_op, attrs])
            h, w, c = current_shape
            current_shape = (calculate_conv_output_dim(h, kernel, padding, stride),
                             calculate_conv_output_dim(w, kernel, padding, stride),
                             c * multiplier)
        elif layer_op in ["avg_pool2d", "max_pool2d"]:
            # (Attribute gathering logic...)
            pool_size, stride, padding = trial.params[f"l{i}_pool_size"], trial.params[f"l{i}_pool_stride"], trial.params[f"l{i}_padding"]
            attrs = f"{pool_size},{pool_size},{stride},{stride},{padding},channels_last"
            cmd_args.extend([layer_op, attrs])
            h, w, c = current_shape
            current_shape = (calculate_conv_output_dim(h, pool_size, padding, stride),
                             calculate_conv_output_dim(w, pool_size, padding, stride),
                             c)
        elif layer_op == "flatten":
            cmd_args.extend(["flatten", "_"])
            h, w, c = current_shape
            current_shape = (h * w * c,)
            is_1d_model = True
        elif layer_op == "fully_connected":
            # (Attribute gathering logic...)
            units, activation, use_bias = trial.params[f"l{i}_fc_units"], trial.params[f"l{i}_activation"], trial.params[f"l{i}_bias"]
            attrs = f",{units},{activation},{use_bias}"
            cmd_args.extend([layer_op, attrs])
            current_shape = (units,)

    # ===================================================================
    # 3. Execute the command (Unchanged)
    # ===================================================================
    print("\n" + "-"*40)
    print(f"Trial {trial.number}: Executing command for a model with {num_layers} layers...")
    print(" ".join(map(str, cmd_args)))
    print("-"*40)
    try:
        subprocess.run(cmd_args, check=True, capture_output=True, text=True)
        print(f"✅ Trial {trial.number}: Model generated successfully.")
        return 0.0
    except subprocess.CalledProcessError as e:
        print(f"❌ Trial {trial.number}: Model generation FAILED.")
        print(f"   Stderr: {e.stderr.strip()}")
        raise optuna.exceptions.TrialPruned()

# ✅ MODIFIED main function to loop through configurations
def main():
    if not os.path.exists(GEN_MODEL_SCRIPT):
        print(f"Error: The script '{GEN_MODEL_SCRIPT}' was not found.")
        sys.exit(1)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- Outer loop to iterate through all desired configurations ---
    for shape_type in SHAPE_TYPES_TO_TEST:
        for max_layers_value in MAX_LAYERS_TO_TEST:
            # Create a unique study name for this configuration
            study_name = f"tflite-gen-{shape_type}-{max_layers_value}layers"
            
            print("\n" + "="*70)
            print(f"🚀 STARTING NEW STUDY: {study_name}")
            print(f"   Shape Type: {shape_type}, Max Layers: {max_layers_value}, Trials: {NUM_TRIALS_PER_STUDY}")
            print("="*70)

            study = optuna.create_study(
                direction="minimize",
                study_name=study_name,
                storage="sqlite:///tflite_generator.db", # All studies save to the same DB file
                load_if_exists=True,
                pruner=optuna.pruners.MedianPruner()
            )
            
            # Use a lambda to pass the current config to the objective function
            objective_with_config = lambda trial: objective(trial, max_layers=max_layers_value, shape_type=shape_type)

            try:
                study.optimize(objective_with_config, n_trials=NUM_TRIALS_PER_STUDY)
            except KeyboardInterrupt:
                print("Study interrupted by user.")
                sys.exit(0)
            
            # --- Report results for the completed study ---
            print("\n" + "-"*50)
            print(f"📊 STUDY FINISHED: {study_name}")
            pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
            complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
            print(f"  Pruned (failed/invalid) trials: {len(pruned_trials)}")
            print(f"  Complete (successful) trials: {len(complete_trials)}")
            print("-"*50)

if __name__ == "__main__":
    main()