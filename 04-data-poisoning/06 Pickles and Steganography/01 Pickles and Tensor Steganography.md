# Pickles And Tensor Steganography

This is the "poisoned container" attack I mentioned earlier, the one that genuinely executes code, as opposed to the trojan which only poisoned behaviour. It has two independent pieces that combine: **pickle deserialization gives you code execution, and tensor steganography gives you a place to hide data.** Both target the model FILE, not the model's math.

The setting is the pre-trained model supply chain: everyone downloads models from hubs like Hugging Face or TensorFlow Hub, which is convenient but a real attack surface (documented in arxiv.org/abs/2107.08590). An attacker takes a benign model, tampers with the file, and re-shares it.

## Part 1: Pickle deserialization (the execution vector)

`pickle` is Python's standard way to serialize an object (turn it into bytes) and deserialize it (rebuild it). **Deserializing untrusted data with pickle is inherently dangerous.**

The mechanism is a method called `__reduce__`. When `pickle.load()` meets an object that defines `__reduce__`, it calls it to get instructions for rebuilding the object, and those instructions are basically "here is a callable and its arguments, run it." An attacker writes a class whose `__reduce__` returns something like `(os.system, ("some command",))` or `(exec, ("some code",))`. When you unpickle it, **pickle blindly runs that, which is arbitrary code execution on your machine.** The Python docs say it outright: "The pickle module is not secure. Only unpickle data you trust."

Now the PyTorch link: **`torch.save` and `torch.load` use pickle under the hood, so `torch.load` inherits the exact same code-execution risk.** Just loading a model file can run an attacker's code, before the model is ever used.

The one concrete detail worth memorising: ==`weights_only=True` versus `weights_only=False`.==

- `torch.load(path, weights_only=True)` uses a restricted unpickler that only allows basic types needed for weights (tensors, dicts, lists, tuples, strings, numbers, None) and refuses to run arbitrary classes or `__reduce__` code. In recent PyTorch this is now the default.
- `torch.load(path, weights_only=False)` behaves like raw `pickle.load` and **will execute embedded malicious code.** This attack specifically depends on someone calling it with `weights_only=False`.

## Part 2: Tensor steganography (the hiding medium)

A model has millions to billions of parameters, stored as `float32` numbers inside tensors, and the whole collection is the `state_dict`. **Tensor steganography hides data inside those numbers so that it is statistically invisible and barely changes the model's performance.** The hidden data might be the payload itself, malware config, or a trigger that the pickle-executed code then reads and uses. So steganography is the carrier, pickle is the execution, and they complement each other.

The trick is to alter only the least significant bits of the parameters.

## Why the LSBs (the float intuition, no binary grind)

A `float32` is 32 bits split into three parts: 1 sign bit (positive or negative), 8 exponent bits (the number's scale), and **23 mantissa bits (the number's precision)**. The only part that matters here is the mantissa, and within it, position matters a lot:

- **Flipping the lowest mantissa bit changes the value by almost nothing.** In the worked example (0.15625), flipping bit 0 shifted the value by about 0.0000000149, which is lost in the noise of a deep learning model's weights.
- **Flipping the highest mantissa bit changes the value a lot.** Flipping bit 22 shifted 0.15625 by 0.0625, which would visibly corrupt the model and expose the tampering.

That contrast is the whole point: **you hide your data in the least significant bits, so each weight changes by a negligible amount, the model still works normally, and a casual look at the weights sees nothing unusual.** Touching the high bits would break the model and give you away.

## How the two combine

Picture the finished malicious model: it loads and performs like a normal model, and its weights pass a statistical glance because the LSB edits look like noise. But it carries a hidden payload in those LSBs, and a malicious `__reduce__` that fires on `torch.load(..., weights_only=False)`, running code that can reassemble and use the hidden data. **Normal-looking file, normal-looking weights, code execution on load.**

## Real-world relevance

Be honest with your time here, in line with what real engagements show: this is a **supply-chain attack, it needs the victim to download and load your file**, so on a black-box remote engagement you almost certainly will not execute it. Treat it like the poisoning module: understand it, be able to describe and report it. That said, **the pickle-equals-RCE idea and the `weights_only` detail are genuinely important and memorable**, so those two facts are worth locking in even if you never run the attack.

Defences worth knowing (these make good report remediation text): only load models from trusted sources, use `weights_only=True`, prefer the **`safetensors`** format (it stores only tensors and literally cannot execute code, which kills the pickle vector), and scan model files with tools like **`picklescan` or `fickling`**. Also note the freshness point: recent PyTorch flipped the `weights_only` default to `True`, so the naive "just call `torch.load`" version of this attack fails on up-to-date installs and requires the loader to explicitly pass `weights_only=False`.

