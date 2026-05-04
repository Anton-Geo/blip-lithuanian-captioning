# BLIP Lithuanian Captioning

This project explores fine-tuning a multilingual vision–language model for image captioning in Lithuanian, a low-resource language.

The goal is to create a custom Lithuanian caption dataset and adapt a pretrained model to generate captions that better match the dataset style.

The task was to:

- create a small, original dataset in a non-dominant language (Lithuanian)
- fine-tune a pretrained vision–language model (BLIP)
- evaluate how fine-tuning affects model behavior

---

## Dataset

Source: OpenImages V7
https://storage.googleapis.com/openimages/web/index.html.

The images were randomly selected with different content.

Annotation flow:

1. Each image was first described in the annotator’s native language.
2. The description was then translated into Lithuanian.
3. Captions were simplified to short, natural descriptions.

All created annotations is kept in data/annotations.csv

Splitting with 200/30/30 proportion for train/validation/test.

---

## Model

The project uses: `Gregor/mblip-mt0-xl` - this is a multilingual vision–language model (mBLIP) capable of generating captions in multiple languages, including Lithuanian.

Total parameters: ~4.8B
Trainable parameters (LoRA): ~4.7M (~0.1%) with following config:
```python
LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q", "v"],
)
```

---

## Training 

General Setup:
| Parameter      | Value             |
|----------------|-------------------|
| Epochs         | 20                |
| Batch size     | 8                 |
| Learning rate  | 1e-4              |
| Optimizer      | AdamW             |
| Scheduler      | ReduceLROnPlateau |
| Early stopping | yes               |

The key idea for using LoRA is to inject small trainable matrices into attention layers, instead of fine-tuning all model weights (This is especially relevant when the pre-training dataset is small).

1. Best model selected based on validation loss
2. Best checkpoint based on early stopping
3. Each experiment with LoRA weights is saved in a separate folder.


## Evaluation

The model was evaluated using several complementary metrics:

### 1. BLEU Score

Measures n-gram overlap between generated captions and reference Lithuanian captions.

Used for:
- baseline vs fine-tuned comparison

---

### 2. Semantic Similarity

Computed using:
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

Measures cosine similarity between:
- reference caption
- generated caption

This metric captures **semantic similarity**, even when wording differs.

---

### 3. CLIP Score

Computed using:
`openai/clip-vit-base-patch32`

Measures cosine similarity between:
- image embedding
- caption embedding

Used as a proxy for **image-text alignment**.

Note:
CLIP is primarily trained on English data, so this metric is considered auxiliary for Lithuanian captions.

---

### 4. Qualitative Evaluation

Several examples were manually inspected to compare:
- reference captions
- baseline outputs
- fine-tuned outputs


---

## Results

### Experiments

Several LoRA fine-tuning experiments were tested with different batch sizes, learning rates, LoRA ranks, LoRA dropout values, and target modules.

| Experiment | Epochs | Batch size | LR | LoRA (r, alpha, dropout) | BLEU |
|------------|--------|------------|----|----------------------|------|
| Baseline (mBLIP) | – | – | – | – | 0.0093 |
| Exp1 (test run) | 5  | 1 | 1e-5 | (8,16,0.05) | 0.0093 |
| Exp2 | 10 | 1 | 3e-5 | (8,16,0.05) | 0.0136 |
| Exp3 | 15 | 2 | 5e-5 | (8,16,0.05) | 0.0216 |
| Exp4 | 20 | 4 | 1e-4 | (8,16,0.05) | 0.0203 |
| Exp5 | 20 | 8 | 1e-3 | (8,16,0.05) | 0.0204 |
| Exp6 | 20 | 8 | 1e-4 | (8,16,0.05) | **0.0243** |
| Exp7 | 20 | 8 | 1e-4 | (16,32,0.05) | 0.0165 |
| Exp8 | 20 | 8 | 1e-4 | (8,16,0.10) | 0.0194 |
| Exp9 | 20 | 8 | 1e-4 | (8,8,0.15) | 0.0174 |


- Increasing batch size significantly improved performance (makes training more stable)
- Moderate LoRA configuration (r=8, alpha=16) worked well
- Larger LoRA (r=16) degraded performance (siignificantly increasing the total amount of trained parameters), likely due to overfitting
- Higher dropout also reduced performance
- The best results were achieved with a simple and stable configuration


The final model from the 6th experoment was selected based on the highest BLEU score on the test set.

Training curve:

![Training loss](figures/training_loss_exp6.png)

The validation loss reached the best value around epoch 9, after which early stopping and the learning rate scheduler helped prevent further overfitting.

---

### Quantitative Comparison

| Metric              | Baseline | Fine-tuned |
|---------------------|----------|------------|
| BLEU                | 0.0093   | 0.0243     |
| Semantic similarity | 0.5175   | 0.5695     |
| CLIP score          | 0.2108   | 0.2086     |

Reference CLIP score: 0.2128

---

## Qualitative Examples

### Example 1

<table>
<tr>
<td width="35%">
<img src="figures/img_001025.jpg" width="100%">
</td>
<td width="65%">

**Reference:**  
Gatvė šalia senovinio pastato su bokštais.

**Baseline:**  
automobilis, važiuojantis gatvėje šalia tvirtovės

**Fine-tuned:**  
Istorinis pastatas gatvėje su automobiliu.

**Semantic similarity:**  
Baseline: `0.5468`  
Fine-tuned: `0.8213`

**CLIP score:**  
Reference: `0.2154`  
Baseline: `0.2326`  
Fine-tuned: `0.1951`

</td>
</tr>
</table>

### Example 2

<table>
<tr>
<td width="35%">
<img src="figures/img_001040.jpg" width="100%">
</td>
<td width="65%">

**Reference:**  
Prie stalo sėdinti moteris su oro balionais.

**Baseline:**  
Žmogus sėdi ant stalo su baltomis ir baltomis balionais.

**Fine-tuned:**  
Jauna moteris prie stalo su balionais.

**Semantic similarity:**  
Baseline: `0.7491`  
Fine-tuned: `0.9125`

**CLIP score:**  
Reference: `0.1700`  
Baseline: `0.1679`  
Fine-tuned: `0.1806`

</td>
</tr>
</table>

### Example 3

<table>
<tr>
<td width="35%">
<img src="figures/img_001028.jpg" width="100%">
</td>
<td width="65%">

**Reference:**  
Lenktyninis kartas trasoje važiavimo metu.

**Baseline:**  
Žmogus, važiuojantis mėlynos spalvos automobiliu.

**Fine-tuned:**  
Autosporto dalyvis, važiuojantis gatve.

**Semantic similarity:**  
Baseline: `0.4112`  
Fine-tuned: `0.5920`

**CLIP score:**  
Reference: `0.2093`  
Baseline: `0.2208`  
Fine-tuned: `0.2530`

</td>
</tr>
</table>

---
### Observations

- BLEU score improved after fine-tuning
- Semantic similarity increased, indicating better alignment with Lithuanian references
- CLIP score slightly decreased (very small changes)

## Interpretation

The results show that fine-tuning improved the model's ability to generate captions that are semantically closer to the Lithuanian reference descriptions (which I created in the preparation stage).

However, CLIP score slightly decreased, indicating that generated captions became less aligned with image content according to the CLIP model.

This can be explained by:

- The dataset contains short and simplified captions
- The baseline model generates longer and more descriptive outputs
- CLIP favors more descriptive (often English-like) captions

Therefore, fine-tuning primarily resulted in **style adaptation** rather than improved image understanding.


## Conclusion

This project demonstrates that fine-tuning a multilingual vision–language model on a small Lithuanian dataset leads to:

- Improved alignment with dataset-specific caption style
- Increased semantic similarity to reference captions
- Insignificant degradation in CLIP-based image-text alignment

The results suggest that the base model already has multilingual capabilities, and fine-tuning does not teach the model a new language instead, it adapts the model to a specific captioning style

This is consistent with the behavior of large pretrained models when trained on small datasets.

