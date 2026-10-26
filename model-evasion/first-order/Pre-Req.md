# The Math, Slowly, With One Tiny Example

Let's drop the jargon and use one made-up toy model you can do on paper. Every formula from my last message comes back here, built up from this example.

## Step 1: A toy "image classifier" with 3 pixels

Imagine an image with only **3 pixels**. Each pixel is a brightness from 0 (black) to 1 (white):

```
x = [0.5, 0.2, 0.4]
```

Our toy model computes a "cat score" with a simple recipe:

```
score = 2×pixel1 − 3×pixel2 + 1×pixel3
```

If the score is above 0 it says **cat**, and below 0 it says **dog**.

Plug in our image:

```
score = 2(0.5) − 3(0.2) + 1(0.4)
      = 1.0 − 0.6 + 0.4
      = 0.8 → positive → "cat" ✅ (correct)
```

**Your goal as the attacker:** change the pixels _a tiny bit_ so the score drops below 0 and the model says "dog."

## Step 2: What a "gradient" actually is

Ask one question per pixel: **"if I brighten this pixel a little, how much does the score move?"**

- Brighten pixel 1 by 0.1 → score goes **up** by 2 × 0.1 = 0.2. Its slope is **+2**.
- Brighten pixel 2 by 0.1 → score goes **down** by 3 × 0.1 = 0.3. Its slope is **−3**.
- Brighten pixel 3 by 0.1 → score goes **up** by 1 × 0.1 = 0.1. Its slope is **+1**.

Put those slopes in a list and you have the gradient:

```
gradient = [+2, −3, +1]
```

**That's all a gradient is: a list of slopes, one per pixel.** Each slope says which way, and how strongly, that pixel pushes the answer. A real model has 784 pixels, so its gradient is a list of 784 slopes. PyTorch calculates them for you, so you never do this by hand.

**Analogy:** you're standing on a foggy hillside. You can't see the whole mountain, but you can feel which way the ground tilts under your feet. The gradient is that tilt.

## Step 3: Use the slopes to push the score down

We want the score to go **down**. For each pixel, move it opposite to its slope:

|Pixel|Slope|To lower the score...|Direction|
|---|---|---|---|
|1|+2|make it darker|−1|
|2|−3|make it brighter|+1|
|3|+1|make it darker|−1|

**The budget rule (ε, "epsilon"):** no pixel may change by more than ε, so the change stays invisible. Let ε = 0.15.

Move every pixel by the full 0.15 in its helpful direction:

```
x_adv = [0.5 − 0.15, 0.2 + 0.15, 0.4 − 0.15]
      = [0.35, 0.35, 0.25]

score = 2(0.35) − 3(0.35) + 1(0.25)
      = 0.70 − 1.05 + 0.25
      = −0.1 → negative → "dog" ❌ (fooled!)
```

**No pixel moved more than 0.15, and the model flipped from cat to dog.** That is FGSM.

## Step 4: Why "sign" and not the slope size

`sign()` keeps only the **direction** (+1 or −1) and drops how big the slope is. Pixel 2's slope is −3 and pixel 3's is +1, but both still move exactly 0.15.

The reason is the budget rule: "each pixel may move _up to_ ε." If a pixel helps at all, holding back gains you nothing, so every helpful pixel moves the full amount. You only need the direction.

**Analogy:** every store gives you a discount and you can spend at most $10 in each one. You don't spend $3 in one store and $7 in another. You spend the full $10 in every store that helps.

## Step 5: Why such tiny changes work (the "add it up" effect)

The score dropped by exactly ε × (2 + 3 + 1) = 0.15 × 6 = **0.9** (from 0.8 to −0.1).

Now picture a real 28×28 image: **784 pixels**. Suppose each slope is only about 1 in size and ε = 0.1:

```
total drop ≈ 0.1 × 784 = 78.4
```

Each pixel change is invisible, but **784 tiny nudges all pointing the same "wrong" way add up to a huge change in the answer.** This is the "linearity hypothesis," and it's why more pixels make a model easier to fool.

## Step 6: Now the scary formula is just Steps 2-4

```
x_adv = x + ε · sign( ∇ₓ J(θ, x, y) )
```

Symbol by symbol:

|Symbol|Plain English|In our example|
|---|---|---|
|`x`|the original image (list of pixel values)|[0.5, 0.2, 0.4]|
|`x_adv`|the tricked image|[0.35, 0.35, 0.25]|
|`ε`|max change allowed per pixel (the budget)|0.15|
|`J`|the **loss** = "how wrong is the model"|goes up as the cat score goes down|
|`θ` (theta)|the model's weights (fixed; we don't touch them)|the 2, −3, 1 in the recipe|
|`y`|the correct label|"cat"|
|`∇ₓ`|"list of slopes, **one per pixel**" (the little ₓ means pixels, not weights)|the gradient|
|`sign`|keep only + or −||

Read aloud: **"Take the image and push every pixel by ε in whichever direction makes the model more wrong."**

**About the "+" in the formula:** it uses the slopes of _wrongness_ (loss), not the cat score. When the cat score goes down, wrongness goes up, so the loss slopes are the score slopes with the signs flipped: [−, +, −]. Adding ε × [−1, +1, −1] gives exactly the moves we made by hand. Same attack, written from the "wrongness" side.

**Clamping:** a pixel can't go below 0 or above 1. If a push gives 1.05, it gets cut back to 1.0.

## Step 7: Norms are just three rulers for "how big was my change"

Forget pixels for a second. You walk **3 blocks east and 4 blocks north** in a city:

- **L∞** = the biggest single move = **4**. "What's the most any one direction changed?"
- **L2** = the straight-line distance as the crow flies = √(3² + 4²) = √25 = **5**. It's Pythagoras, measuring overall size.
- **L0** = how many directions you moved in at all = **2**.

Same thing with our attack, where the change was δ = [−0.15, +0.15, −0.15]:

- **L∞** = 0.15 (the biggest single pixel change; **FGSM's budget is set with this ruler**)
- **L2** = √(0.15² + 0.15² + 0.15²) ≈ 0.26 (overall size; **DeepFool tries to keep this ruler small**)
- **L0** = 3 (all 3 pixels were touched)

## Quick self-check

Same image and model, but now ε = 0.2. What's the new score, and does it say cat or dog? Hint: the score drops by ε × 6.

If any step above lost you, tell me the step number and I'll redo just that part a different way.

-0.40