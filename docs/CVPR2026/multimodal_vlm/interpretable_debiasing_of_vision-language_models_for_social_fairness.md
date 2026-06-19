---
title: >-
  [Paper Note] Interpretable Debiasing of Vision-Language Models for Social Fairness
description: >-
  [CVPR 2026][LLM Safety][Interpretability] DeBiasLens is proposed to locate "social neurons" encoding social attributes by training Sparse Autoencoders (SAEs) on VLM encoders, then selectively deactivating these neurons during inference to mitigate bias. It reduces Max Skew by 9-16% on CLIP and gender bias ratios by 40-50% on InternVL2 while maintaining general
tags:
  - CVPR 2026
  - LLM Safety
  - Interpretability
date: 2026-05-08
content_hash: 5fd59277d6d1e23f
---
# Interpretable Debiasing of Vision-Language Models for Social Fairness

**Conference**: CVPR 2026  
**arXiv**: [2602.24014](https://arxiv.org/abs/2602.24014)  
**Code**: TBD  
**Area**: Multimodal VLM  
**Keywords**: VLM debiasing, social fairness, Sparse Autoencoders (SAE), interpretability, neuron modulation

## TL;DR
DeBiasLens is proposed to locate "social neurons" encoding social attributes by training Sparse Autoencoders (SAEs) on VLM encoders, then selectively deactivating these neurons during inference to mitigate bias. It reduces Max Skew by 9-16% on CLIP and gender bias ratios by 40-50% on InternVL2 while maintaining general performance.

## Background & Motivation
**Background**: VLMs/LVLMs inherit and amplify social biases from large-scale data—e.g., CLIP's "CEO" retrieval is biased towards males, and InternVL favors specific genders in ambiguous contexts. Existing debiasing methods include fine-tuning, prompt tuning, and pruning.

**Limitations of Prior Work**: Existing debiasing methods address surface bias symptoms without touching the internal propagation mechanisms of bias in representations. Although pruning attempts to find key parameters, debiasing often comes at the cost of general performance due to neuron polysemanticity (where a single neuron encodes both bias and useful knowledge).

**Key Challenge**: Bias and useful knowledge are entangled within model weights; direct modification of weights inevitably leads to performance degradation.

**Goal**: How to accurately locate and modulate bias-related monosemantic features within an interpretable framework without affecting useful knowledge.

**Key Insight**: Utilize SAEs to decompose entangled feature spaces into sparse, monosemantic neurons (satisfying monosemanticity), allowing bias-related "social neurons" to be independently located and manipulated.

**Core Idea**: SAE decouples polysemantic features into monosemantic ones → Filter neurons encoding specific social attributes → Deactivate these neurons during inference to eliminate bias.

## Method

### Overall Architecture
The core difficulty DeBiasLens solves is the entanglement of bias and useful knowledge within model weights. The strategy is to "cleanly decompose" the entangled feature space before performing targeted surgery. The pipeline consists of three steps: first, attach a Sparse Autoencoder (SAE) to the last layer of the VLM encoder to decouple dense, polysemantic encoder outputs into high-dimensional, monosemantic sparse representations; second, identify "social neurons" specifically encoding certain social attributes within this sparse space; finally, deactivate these neurons during inference and replace the original features with reconstructed features for downstream tasks. This process freezes the original model and only operates on the SAE.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Frozen VLM Image/Text Encoder<br/>Last Layer Dense Polysemantic Feature v"] --> B["SAE Decoupling<br/>Overcomplete Sparse Coding (Expansion Factor 8)<br/>Produces Monosemantic Sparse Code z"]
    B --> C["Social Neuron Detection<br/>High Consistency (τ=0.9) + Group Difference<br/>Identifies Attribute-Specific Neurons 𝒵B"]
    C --> D["Social Neuron Modulation Inference<br/>Set Activations in 𝒵B to γ → Reconstruction v̂"]
    D -->|Weighted Blending v'=αv̂+(1−α)v, α=0.6| E["Debiased Feature v'<br/>Input to Downstream T2I / Retrieval / VQA"]
```

### Key Designs

**1. SAE Decoupling: Decomposing Polysemantic Features into Monosemantic Neurons**

The root cause of debiasing failure lies in neuron polysemanticity—a single neuron may encode both gender and useful knowledge. Removing it damages general capabilities. SAE uses an overcomplete sparse coding $\phi(\mathbf{v}) = \sigma(\mathbf{W}_{enc}^\top(\mathbf{v} - \mathbf{b}_1))$ to project the encoder output $\mathbf{v}$ into a much higher-dimensional sparse space (expansion factor set to 8), forcing each active neuron to represent a single concept (monosemanticity). Training utilizes the multi-scale reconstruction loss of Matryoshka SAE. Crucially, **no social attribute labels are required during SAE training**—reconstruction on face or caption datasets allows the SAE to spontaneously capture social attribute directions. A direct validation signal is that after applying the SAE, the cosine similarity difference between image pairs with overlapping social attributes and random pairs increases significantly, indicating that social attributes are indeed encoded by specific neurons.

**2. Social Neuron Detection: Locating Attribute-Specific Neurons via Group Set Difference**

After decoupling, it is necessary to identify which neurons are responsible for gender, age, or race. The paper calculates the effectiveness of neurons for each social group $g$—a neuron is considered "stably active in this group" if its non-zero activation ratio reaches threshold $\tau=0.9$, forming the effective neuron set $\mathcal{E}_g$. High activation alone is insufficient because general neurons are active across all groups. Neurons truly encoding a specific group should only be active for that group. Thus, the group set difference is taken:

$$\mathcal{N}_g = \mathcal{E}_g \setminus \bigcup_{h \neq g} \mathcal{E}_h$$

By excluding neurons that are also effective for other groups and selecting the top neurons with the highest mean activation, the combination of "high consistency ($\geq\tau$)" and "group specificity (set difference)" isolates monosemantic neurons encoding only the social attributes of that group.

**3. Social Neuron Modulation Inference: Deactivating Neurons Without Modifying Weights**

Once social neurons are identified, their activations in the SAE sparse code are forcibly set to $\gamma$ (usually 0) during inference, resulting in a debiased sparse code $\mathbf{z}'$. This is then reconstructed into a debiased feature $\hat{\mathbf{v}} = \psi(\mathbf{z}')$ via the SAE decoder. To prevent reconstruction errors from harming downstream tasks, the final output is a weighted blend of the reconstructed and original features: $\mathbf{v}' = \alpha\hat{\mathbf{v}} + (1-\alpha)\mathbf{v}$, with $\alpha=0.6$. The entire process only modifies the SAE sparse representation and does not touch the original model weights, ensuring that bias is stripped away while general capabilities remain nearly intact—a key advantage over pruning or fine-tuning.

## Key Experimental Results

### Main Results (CLIP ViT-B/16 Gender Bias, Max Skew↓)

| Method | Interpretable? | Adj | Occup | Act | Ster |
|------|--------|------|-------|------|------|
| CLIP Baseline | - | 21.9 | 33.5 | 19.8 | 32.5 |
| Bend-VLM | ✗ | 10.8 | 10.2 | 9.8 | 9.1 |
| SANER | ✗ | 8.9 | 14.5 | 7.7 | - |
| **Ours (T)** | **✓** | **7.1** | **16.2** | **14.2** | **8.1** |
| **Ours (I)** | **✓** | 14.2 | 21.5 | 20.0 | 18.3 |

### LVLM Results

| Configuration | Gender Bias Reduction | Gen. Performance Drop |
|------|-------------|------------|
| DeBiasLens-InternVL2 (α=0.6) | 40-50% | Only 4-10 points |
| Pruning Methods | Similar | Larger drop |
| Prompt Engineering | Limited | Minimal |

### Key Findings
- DeBiasLens(T) performs best on adjectives and stereotype prompts without requiring attribute labels for training.
- Deactivating only the top-1 social neuron achieves performance comparable to deactivating all effective neurons, confirming that neurons do not interfere with each other.
- Gender neurons exhibit high specificity—deactivating them does not affect age bias; however, age neurons show crosstalk (40% of age neurons exhibit gender skew).
- Image encoders are more effective for high-resolution VLMs (ViT-L/14@336), while text encoders are more effective for standard resolutions.

## Highlights & Insights
- **Interpretability-driven debiasing** is a new paradigm: it does not mitigate bias outputs in a "black-box" manner but precisely locates and manipulates the internal mechanisms where bias arises.
- The monosemantic nature of SAEs makes them an ideal tool for debiasing: each neuron encodes a single concept, and deactivation does not trigger chain reactions.
- The framework is applicable to both encoder-only (CLIP) and encoder-decoder (InternVL2) VLMs, showing good versatility.
- Only middle representations are modified without changing model weights, making deployment simple.

## Limitations & Future Work
- The social neuron detection phase still requires social attribute labels to partition groups (though SAE training does not).
- Current validation focuses primarily on gender bias, with fewer ablations on age and race.
- The SAE expansion factor and threshold $\tau$ require hyperparameter tuning.
- Bias mitigation effectiveness across different cultures/languages has not been verified.

## Related Work & Insights
- **vs Bend-VLM**: Bend-VLM directly debiases embeddings via black-box operations; DeBiasLens uses interpretable neuron manipulation, making it transparent and auditable.
- **vs SANER**: SANER trains residual layers on the text encoder to erase attribute information; DeBiasLens performs selective deactivation after SAE decoupling, providing higher precision.
- **vs MMNeuron**: MMNeuron looks for attribute-specific neurons in pre-trained weights, but weight neurons are polysemantic; SAE neurons are monosemantic, enabling more precise debiasing.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to use SAE for interpretable VLM debiasing with a unique entry point.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple VLMs/LVLMs + multiple evaluation dimensions + neuron specificity validation.
- Writing Quality: ⭐⭐⭐⭐ Clear methodology; the "social neuron" concept is intuitive.
- Value: ⭐⭐⭐⭐⭐ Provides a new interpretable and auditable tool for AI fairness.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] A Closed-Form Solution for Debiasing Vision-Language Models with Utility Guarantees Across Modalities and Tasks](a_closedform_solution_for_debiasing_visionlanguage.md)
- [\[CVPR 2026\] FairLLaVA: Fairness-Aware Parameter-Efficient Fine-Tuning for Large Vision-Language Models](fairllava_fairness-aware_parameter-efficient_fine-tuning_for_large_vision-langua.md)
- [\[CVPR 2026\] VL-Eraser: Vacuum Distillation for Machine Unlearning in Vision-Language Models](vl-eraser_vacuum_distillation_for_machine_unlearning_in_vision-language_models.md)
- [\[CVPR 2026\] Test-Time Attention Purification for Backdoored Large Vision Language Models](test-time_attention_purification_for_backdoored_large_vision_language_models.md)
- [\[CVPR 2026\] Phantasia: Context-Adaptive Backdoors in Vision Language Models](phantasia_context-adaptive_backdoors_in_vision_language_models.md)

</div>

<!-- RELATED:END -->
