"""Testing script for POPE Dataset with output display"""
# hyperparameters
# llava-onevision-llava-onevision-qwen2-0.5b has 23 layers in the language model
# set quantize_layer_start and quantize_layer_end to the range of layers you want to quantize to 2 bit
# 20 to 23 gives an accuracy boost on POPE dataset from 85 to 87
quantize_layer_start= 17 # 16 works well as the first layer. We have a problem exactly at layer 15.
quantize_layer_end = 23
quantize_layer_list = list(range(quantize_layer_start, quantize_layer_end + 1))

quantized_layer_limit = 10000
error_threshold = 0.99 #layers with relative error greater than this threshold will not be quantized

import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from datasets import load_dataset
import re
import traceback
import sys
import os
sys.path.append('rank-constrained-regression-main')
from src.caldera.utils.dataclasses import CalderaParams
from src.caldera.utils.quantization import QuantizerFactory
from src.caldera.decomposition.alg import caldera
from scipy.linalg import hadamard
import numpy as np
import matplotlib.pyplot as plt
import datetime
#use gpu 2
#pci order
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#run the command 'cd /home/pilanci/Dropbox/Python_ubuntu/llava-onevision'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
test_samples = 100
quantize = True #set to True to quantize using CALDERA_v0 (this is a simplified version of CALDERA)
output_activations = False #this is used to get the Hessian of the model if the Hessians file does not exist
Hadamard = False # set to True to use Hadamard transform compression (default is False)
diagnose_singular_values = False
quantized_param_count = 0
unquantized_language_param_count = 0
vision_param_count = 0
# Dictionary to store activations
layer_activations = {}
Hessians = {}
# load Hessians file
Hessians_path = "diag_Hessians.pt"# diagonals of the Hessian. Full Hessians are in "all_Hessians_full.pt"
Hall = torch.load(Hessians_path)
# Open a log file
date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = open("logs/output_log_" + date_time + ".txt", "w")
# Redirect `sys.stdout` to write to both the console and the file
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            if not stream.closed:  # Avoid writing to closed streams
                stream.write(message)
                stream.flush()

    def flush(self):
        for stream in self.streams:
            if not stream.closed:  # Avoid flushing closed streams
                stream.flush()

# Redirect `sys.stdout` and optionally `sys.stderr`
sys.stdout = Tee(sys.stdout, log_file)
# Define the hook function
def hook_fn(module, inputs, outputs, layer_name):
    # Store the inputs to the layer
    layer_activations[layer_name] = inputs[0].detach().cpu()
def next_power_of_two(n):
    """Return the next power of 2 greater than or equal to n."""
    return 1 << (n - 1).bit_length()

def get_normalized_hadamard(n):
    """Get a normalized Hadamard matrix of size n x n.
    n must be a power of 2."""
    H = hadamard(n)
    return H / np.sqrt(n)  # Normalize to preserve energy

def pad_matrix(W, target_rows, target_cols):
    """Pad matrix W with zeros to reach target dimensions."""
    padded = np.zeros((target_rows, target_cols))
    orig_rows, orig_cols = W.shape
    padded[:orig_rows, :orig_cols] = W
    return padded

def hadamard_transform(W, inverse=False, original_shape=None):
    """Apply Hadamard transform H1 W H2 to matrix W or its inverse.
    
    Parameters:
    -----------
    W : numpy.ndarray
        Input matrix
    inverse : bool
        If True, performs inverse transform. If False, performs forward transform.
    original_shape : tuple
        Required for inverse transform to recover original matrix dimensions.
        Format: (rows, cols)
    
    Returns:
    --------
    numpy.ndarray
        Transformed or inverse-transformed matrix
    tuple
        Original shape (only returned for forward transform)
    """
    rows, cols = W.shape if not inverse else original_shape
    
    # Calculate required dimensions (next power of 2)
    pad_rows = next_power_of_two(rows)
    pad_cols = next_power_of_two(cols)
    
    # Generate normalized Hadamard matrices
    H1 = get_normalized_hadamard(pad_rows)
    H2 = get_normalized_hadamard(pad_cols)
    
    if not inverse:
        # Forward transform
        W_padded = pad_matrix(W, pad_rows, pad_cols)
        result = H1 @ W_padded @ H2
        return result, (rows, cols)
    else:
        # Inverse transform
        # Since H is symmetric and orthogonal, H^(-1) = H
        result = H1 @ W @ H2
        # Truncate back to original dimensions
        result = result[:rows, :cols]
        return result

def apply_CALDERA_quantization(model):
    global Hall
    global Hessians_path
    global quantized_param_count
    global unquantized_language_param_count
    global vision_param_count
    global error_threshold
    if os.path.exists(Hessians_path):
        print(f"'{Hessians_path}' does exist.") 
    else:
        print(f"'{Hessians_path}' does not exist.")
    quantized_layer_counter = 1
    for name, module in model.named_modules():
        if ( #only consider language layers and layers that have weights
        hasattr(module, 'weight')
        and
        'language' in name 
        ):
            with torch.no_grad():
                if ( #restrict quantization to certain layers
                    any(key in name for key in ['mlp.up_proj', 'mlp.down_proj','mlp.gate_proj', 'q_proj','k_proj', 'v_proj', 'o_proj'])#, 'q_proj', 'k_proj', 'v_proj', 'o_proj']) #'mlp.gate_proj',  #only language mlp layers for now fc1 fc2 in vision tower is skipped
                    and
                    module.weight.size(0) > 500 and 
                    module.weight.size(1) > 500 and 
                    #('mlp' in name) and
                    any(f'layers.{i}' in name for i in quantize_layer_list) and
                    (quantized_layer_counter <= quantized_layer_limit)
                ):
                    h = Hall[name].to(torch.float32).to(device) #diagonals of the Hessian
                    #diagonals of the Hessian
                    H = torch.diag_embed(h) #form theg diagonal matrix
                    quantized_layer_counter += 1
                #caldera parameters
                    quant_factory_Q = QuantizerFactory(method="uniform", block_size=64)
                    quant_factory__LR = QuantizerFactory(method="uniform", block_size=64)
                    quant_params = CalderaParams(
                        compute_quantized_component=True,  
                        compute_low_rank_factors=True,      
                        Q_bits=2,                           
                        L_bits=16,                          
                        R_bits=16,
                        rank=200,
                        iters=5,
                        lplr_iters=5, #was 5
                        activation_aware_LR=True,
                        update_order=["Q", "LR"],
                        quant_factory_Q=quant_factory_Q,
                        quant_factory_LR=quant_factory__LR,
                        rand_svd=False,
                        sigma_reg=1e-8                             
                    )
                    W = module.weight.data
                    #save W to a file
                    if Hadamard == False:
                        caldera_decom = caldera(
                            quant_params=quant_params,
                            W = W,
                            H = H,
                            device="cuda",
                            use_tqdm=True,
                            scale_W=False # scaled true makes the Frob norm error much larger, why?
                        )
                        out = caldera_decom.Q + caldera_decom.L @ caldera_decom.R
                        import pdb; pdb.set_trace()
                        module.weight.data = out#.to(W.dtype) #set model weight to the quantized weight
                        if diagnose_singular_values:
                            #compute the svd of W converted to float 32
                            U, S, V = torch.svd(W.to(torch.float32))
                            #compute the svd of out
                            U_out, S_out, V_out = torch.svd(out.to(torch.float32))
                            #plot the singular values of W and out
                            plt.plot(S.cpu().numpy(), label='W')
                            plt.plot(S_out.cpu().numpy(), label='out')
                            plt.legend()
                            plt.show()
                            #print to pdf file name equal to module name
                            plt.savefig(f"{name}.pdf")
                        decomp_error = torch.norm(W - out, p='fro') / torch.norm(W, p='fro')
                        print(f"Error of the decomposition: {decomp_error}")
                        if decomp_error > error_threshold:
                            print(f"Error of the decomposition is greater than threshold for {name}. Skipping quantization")
                            module.weight.data = W
                            unquantized_language_param_count += W.numel()
                        else:
                            print(f"Applied CALDERA to {name}.weight, shape: {module.weight.data.shape}")
                            quantized_param_count += W.numel()
                    else:
                        # Hadamard transform over both rows and columns
                        transformedW, original_shape = hadamard_transform(W.to('cpu'), inverse=False)
                        # Inverse transform
                        #apply caldera with no Hessian
                        caldera_decom = caldera(
                            quant_params=quant_params,
                            W = torch.tensor(transformedW).to(torch.float32).to(device),# empty argument for H
                            H =  H.to(torch.float32).to(device),
                            device="cuda",
                            use_tqdm=True,
                            scale_W=False # scaled true makes the Frob norm error much larger, why?
                        )
                        out = caldera_decom.Q + caldera_decom.L @ caldera_decom.R
                        #covnert to numpy
                        out = out.cpu().numpy()
                        recoveredW = hadamard_transform(out, inverse=True, original_shape=original_shape)
                        recoveredW = torch.tensor(recoveredW).to(device)
                        #import pdb; pdb.set_trace()
                        module.weight.data = recoveredW.to(W.dtype)
                else:
                    unquantized_language_param_count += module.weight.numel()
                    print(f"Skipped quantization for {name} with shape {module.weight.size()}")
                    #pass
                    #module.weight.data = module.weight.data.to(torch.float16)

        else:
            print(f"Skipped quantization for {name}")
            if hasattr(module, 'weight'):
               vision_param_count += module.weight.numel()
               print(f"Skipped quantization for {name} with shape {module.weight.size()}")



# Load POPE dataset
ds = load_dataset("lmms-lab/POPE", "default", split="test")

def main():
    global Hessians
    model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float32,#,
        #low_cpu_mem_usage=True#,
        #=True
    ).to(0)
    processor = AutoProcessor.from_pretrained(model_id)
    print(model)
    if output_activations == True:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):  # Check if the module is a Linear layer
                module.register_forward_hook(lambda mod, inp, out, name=name: hook_fn(mod, inp, out, name))
                #print(f"Hook attached to: {name}")
    #calibration pass
    #check if Hessians.pt exists
    if os.path.exists(Hessians_path):
        #Hessians = torch.load(Hessians_path)
        print(f"'{Hessians_path}' exists.")
    else:
        print(f"'{Hessians_path}' does not exist. Recomputing Hessians...")
        target_layer_name = 'language_model.model.layers.0.mlp.up_proj'
        activation_sum = None
        # Loop through the dataset
        for idx, example in enumerate(ds):  # Replace with your dataset
            question = example['question']
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image"},
                    ],
                },
            ]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(images=example['image'], text=prompt, return_tensors="pt").to(0, torch.float32)
            
            # Perform a forward pass
            model.generate(**inputs, max_new_tokens=200, do_sample=False, pad_token_id = 151643)
            
            # Retrieve activations for the current sample
            if target_layer_name in layer_activations:
                activations = layer_activations[target_layer_name]  # Shape: [batch_size, hidden_dim]
                #reshape to a vector
                activations = activations.view(activations.size(2), -1)
                #convert to float32
                activations = activations.to(torch.float64)
                a_aT = torch.matmul(activations, activations.transpose(0, 1))  # Compute a * a^T
                if activation_sum is None:
                    activation_sum = a_aT
                else:
                    activation_sum += a_aT
            print(f"Computed Hessian for {target_layer_name} for {idx + 1} samples.")
            #if idx ==100:
            activation_sum = activation_sum / (idx + 1)
            Hessians[target_layer_name] = activation_sum
            print(f"Computed and saved Hessians for {target_layer_name}.")
            #save this to a file if the file does not exist
            torch.save(activation_sum, 'Hessians_full.pt')
                #break
    #terminate code
    if quantize:
        apply_CALDERA_quantization(model)
    
    # print the number of quantized and unquantized parameters
    print(f"Quantized parameters: {quantized_param_count}")
    print(f"Unquantized parameters: {unquantized_language_param_count}")
    # print the number of bits assumping unquantized parameters are 4 bits and quantized parameters are 2 bits
    print(f"Total number of bits: {quantized_param_count * 2 + unquantized_language_param_count * 4}")
    # print the number of bits assumping unquantized parameters are 4 bits and quantized parameters are 4 bits
    print(f"Prior total number of bits: {quantized_param_count * 4 + unquantized_language_param_count * 4}")
    # print the ratio of the number of bits to the prior number of bits
    if (quantized_param_count + unquantized_language_param_count) > 0:
        print(f"Ratio of the number of bits to the prior number of bits: {(quantized_param_count * 2 + unquantized_language_param_count * 4)/(quantized_param_count * 4 + unquantized_language_param_count * 4)}")
        print(f"Ratio of the 2 bit quantization to 4 bit quantization: {(quantized_param_count)/(quantized_param_count+unquantized_language_param_count)}")
    correct_predictions = 0
    total_samples = len(ds)

    # Loop through the test split of the POPE dataset
    for idx, example in enumerate(ds):
        print(f"Processing example {idx + 1}/{total_samples}")
        try:
            question = example['question']
            expected_answer = example['answer'].strip().lower()  # Normalize expected answer
            
            # Prepare the prompt for the model
            conversation = [
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image"},
                    ],
                },
            ]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(images=example['image'],text=prompt, return_tensors="pt").to(device)#.to(0, torch.float16)
            output = model.generate(**inputs, max_new_tokens=200, do_sample=False,pad_token_id = 151643)
            text_output = processor.decode(output[0][2:], skip_special_tokens=True)

            # Extract the predicted answer from the model output
            last_assistant_msg = text_output.split("assistant")[-1].strip().lower()
            match = re.search(r"\b(yes|no)\b", last_assistant_msg)
            predicted_answer = match.group(0) if match else None

            # Display results
            print(f"Question: {question}")
            print(f"Expected Answer: {expected_answer}")
            print(f"Model Output: {last_assistant_msg}")
            print(f"Predicted Answer: {predicted_answer}\n")

            # Check accuracy
            if predicted_answer is not None and predicted_answer.lower() == expected_answer.lower():
                correct_predictions += 1
                print("Correct prediction!")
            else:
                print("Incorrect prediction.")
                print(f"Expected: {expected_answer}")
                print(f"Predicted: {predicted_answer}")

            accuracy_current = correct_predictions / (idx + 1) * 100
            if (quantized_param_count + unquantized_language_param_count) > 0:
                print(f"Ratio of the 2 bit quantization to 4 bit quantization (language): {(quantized_param_count)/(quantized_param_count+unquantized_language_param_count)}")
                print(f"Ratio of the 2 bit quantization to 4 bit quantization (total): {(quantized_param_count)/(quantized_param_count+unquantized_language_param_count+vision_param_count)}")
            print(f"Current accuracy on POPE dataset: {accuracy_current:.2f}%\n")
            if idx>test_samples:
                break

        except Exception as e:
            print(f"Error processing example {idx + 1}: {e}")
            traceback.print_exc()
            continue  # Skip problematic example and move on

    # Calculate and print final accuracy
    accuracy = correct_predictions / (idx+1) * 100
    print(f"Final Accuracy on POPE dataset: {accuracy:.2f}%")
    print(f"Quantized parameters: {quantized_param_count}")
    print(f"Unquantized language parameters: {unquantized_language_param_count}")
    print(f"Vision parameters: {vision_param_count}")
    print(f"Ratio of the 2 bit quantization to 4 bit quantization (language): {(quantized_param_count)/(quantized_param_count+unquantized_language_param_count)}")
    print(f"Ratio of the 2 bit quantization to 4 bit quantization (total): {(quantized_param_count)/(quantized_param_count+unquantized_language_param_count+vision_param_count)}")
    # Save accuracy to txt file
    with open("accuracy.txt", "w") as f:
        f.write(f"Accuracy on POPE dataset: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
    log_file.close()
    sys.stdout = sys.__stdout__

