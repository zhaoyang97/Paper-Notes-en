---
title: >-
  [Paper Note] Forget-It-All: Multi-Concept Machine Unlearning via Concept-Aware Neuron Masking
description: >-
  [ICML 2026][Image Generation][Multi-concept Machine Unlearning] This paper proposes FIA, a training-free multi-concept unlearning framework. By combining "Contrastive Concept Saliency" and "Spatio-Temporal Sparse Filteri…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Multi-concept Machine Unlearning"
  - "Text-to-Image Diffusion"
  - "Concept-Sensitive Neurons"
  - "Neuron Masking"
  - "Training-Free"
date: 2026-05-08
content_hash: da414f91cf1f5f81
---

# Forget-It-All: Multi-Concept Machine Unlearning via Concept-Aware Neuron Masking

**Conference**: ICML 2026  
**arXiv**: [2601.06163](https://arxiv.org/abs/2601.06163)  
**Code**: https://github.com/kaiyuan02415/Forget-It-All (Available)  
**Area**: AI Safety / Diffusion Model Unlearning / Model Sparsity  
**Keywords**: Multi-concept Machine Unlearning, Text-to-Image Diffusion, Concept-Sensitive Neurons, Neuron Masking, Training-Free  

## TL;DR
This paper proposes FIA, a training-free multi-concept unlearning framework. By combining "Contrastive Concept Saliency" and "Spatio-Temporal Sparse Filtering," FIA localizes concept-sensitive neurons for each target concept. When fusing multi-concept masks, it explicitly preserves "concept-agnostic neurons" that respond to multiple concepts simultaneously, pruning only the genuinely concept-specific connections. On SD v1.5/v1.4, it achieves simultaneous unlearning of ten Imagenette classes (avg. forget accuracy 1.9%, overall score 86%), multiple artistic styles, and inappropriate content with a total sparsity rate of $<0.3\%$.

## Background & Motivation

**Background**: T2I diffusion models (e.g., Stable Diffusion) generate high-quality images but pose risks regarding copyright, privacy, and inappropriate content. Machine unlearning (MU) is considered a cost-effective solution. Current mainstream methods fall into two categories: fine-tuning-based methods (FMN, SalUn, AC, ESD, MACE, SPM) that update cross-attention or add LoRA to erase concepts; and training-free methods (UCE, SLD, ConceptPrune) that directly edit weights or inject safety guidance during inference.

**Limitations of Prior Work**: Most methods are designed for single-concept unlearning. Applying them sequentially to multiple concepts leads to two issues: (i) previously forgotten concepts may be "reactivated" or overall generation quality collapses; (ii) fine-tuning is extremely sensitive to hyperparameters, requiring re-tuning for every added concept, which increases computational overhead linearly and risks overfitting. Even specialized multi-concept methods (SPM/MACE/COGFD/SepME) rely on additional LoRA, concept graphs, or closed-form editing, making it difficult to balance unlearning effectiveness and generation quality.

**Key Challenge**: There is a conflict between "completely deleting $N$ concepts" and "preserving general generation capabilities." Many weights contribute to multiple conceptual expressions simultaneously. Simply taking the union of all candidate neuron masks would cause excessive "collateral damage" to neurons sharing low-level features.

**Goal**: (1) Identify truly "concept-sensitive" neurons for each target concept without fine-tuning or adding parameters; (2) Protect "concept-agnostic" neurons shared by multiple concepts during mask fusion to avoid quality degradation.

**Key Insight**: The authors reframe multi-concept unlearning as a **model sparsity** problem. Since a single concept only activates a small number of neurons, performing "concept-aware pruning" for each concept and merging masks with an intelligent fusion strategy can achieve multi-concept unlearning at an extremely low sparsity rate.

**Core Idea**: Use *contrastive and spatio-temporal joint* neuron saliency to isolate concept-specific neurons from shared neurons; prune the former and preserve the latter to "forget $N$ concepts without forgetting how to draw."

## Method

### Overall Architecture
FIA is a fully inference-time, training-free pipeline. Given a pre-trained diffusion model and a set of target concepts $\{c_1,\dots,c_C\}$:

1. For each concept $c$, sample $K=50$ images using "concept-containing prompts" and "base context prompts," collecting activations at each layer along a 50-step denoising trajectory.
2. Compute *Contrastive Concept Saliency* $S_{\ell,t,i,j}$ to quantify the relative contribution of each weight connection to concept $c$.
3. Aggregate these into *Time-Integrated Sensitivity* $A_{\ell,i,j}$ along the time dimension, followed by an intersection of intra-channel and layer-wide top-k selections to obtain the set of *Concept-Sensitive Neurons* $\mathcal{Q}_\ell^{(c)}$, represented as a binary mask.
4. Perform "Concept-Agnostic Aware Fusion" on the $C$ masks: neurons activated by $\geq \alpha C$ concepts are identified as *Concept-Agnostic Neurons* and preserved. Only "concept-specific" neurons belonging to a few target concepts are pruned.
5. Apply the final fused mask to zero out corresponding weights for inference. The total pruning rate is $<0.3\%$.

### Key Designs

1. **Contrastive Concept Saliency**:
    - **Function**: Quantifies the actual importance of a weight connection $(i,j)$ in generating the target concept without backpropagation.
    - **Mechanism**: First, a unified energy form is used: $U_{\ell,t,i,j}=|W_{\ell,i,j}|\cdot\|X_{\ell,t,j}\|_2\cdot \frac{|\langle X_{\ell,t,j},Y_{\ell,t,i}\rangle|}{\|X_{\ell,t,j}\|_2\cdot\|Y_{\ell,t,i}\|_2+\varepsilon}$, characterizing weight magnitude, input activity, and "input-output directional consistency" (the cosine term penalizes neurons with strong activation but incorrect noise direction). Then, using "concept prompts" (e.g., *a golf ball on the table*) and "base prompts" (e.g., *a table*), calculate the means $\mu_c, \mu_b$ and base variance $\sigma_b$ of $U$, defining $S_{\ell,t,i,j}=\max(0,\mu_c-\mu_b-\sigma_b)$.
    - **Design Motivation**: Magnitude or activation alone cannot distinguish "concept-specific" from "general features." Explicitly contrasting concept-vs-base and subtracting background variance acts as a statistical significance filter, leaving only connections with stable positive contributions to the concept.

2. **Spatio-Temporal Sparse Joint Localization**:
    - **Function**: Aggregates step-wise and position-wise $S$ into a set of neurons $\mathcal{Q}_\ell^{(c)}$ that stably respond to the target concept.
    - **Mechanism**: Temporal aggregation is performed as $A_{\ell,i,j}=\tfrac12\cdot\tfrac1T\sum_t S_{\ell,t,i,j}+\tfrac12\cdot\tfrac1T\sum_t \mathbf{1}[S_{\ell,t,i,j}>\tau_{\ell,t}]$, assigning equal weight to "average response intensity" and "activation frequency" to avoid selecting neurons with transient noise bursts. $\tau_{\ell,t}$ is an adaptive threshold based on the top-$r_1$ of each layer at each step. For spatial filtering: an intra-channel top-$k$ selection yields a local set $C_\ell$, and a global top-$K_g=r_2\cdot C_{out}\cdot C_{in}$ selection yields a global set $G_\ell$. Finally, $\mathcal{Q}_\ell^{(c)} = C_\ell \cap G_\ell$.
    - **Design Motivation**: Different denoising steps focus on different semantic features; single-step selection might mistake transient noise for concept neurons. Intra-channel top-$k$ ensures the budget isn't consumed by a few dominant channels, while global top-$k$ ensures strength; their intersection provides the stability and precision required for multi-concept fusion.

3. **Concept-Agnostic Neuron Protection for Multi-Mask Fusion**:
    - **Function**: Merges $C$ single-concept masks into a unified multi-concept pruning mask while preserving general generation capabilities.
    - **Mechanism**: First, calculate the hit count for each neuron: $s_{\ell,i,j}=\sum_{c=1}^C \mathrm{Mask}_\ell^{(c)}[i,j]$. Set a threshold $\tau_{ca}=\lceil \alpha C \rceil$ (where $\alpha \in (0,1]$ is the "concept-agnostic ratio"). Neurons with $s_{\ell,i,j}\geq \tau_{ca}$ are judged as *Concept-Agnostic* and forcibly preserved. Only "concept-specific" neurons where $0 < s_{\ell,i,j} < \tau_{ca}$ are pruned.
    - **Design Motivation**: The authors observed that a small group of neurons responds to almost all target concepts; these actually encode basic skills like color, shape, or composition. A naive union mask would erroneously prune them, causing CLIP/FID scores to collapse. A simple count and threshold identify and lock "general feature neurons," ensuring pruning is both thorough and restrained.

### Loss & Training
The method is completely training-free, requiring no gradient updates or learnable parameters. Only three sparsity rates need manual setting: temporal sparsity $r_1$, spatial sparsity $r_2$, and concept-agnostic ratio $\alpha$. Only 10 image samples per concept are needed to estimate $S$, and saliency collection is completed within 50 denoising steps. It can be deployed on a single A6000 GPU.

## Key Experimental Results

### Main Results

**Multi-Object Unlearning (10 Imagenette classes simultaneously, SD v1.5)**:

| Method | Avg. Forget Acc. ↓ | CLIP_coco ↑ | Remarks |
| :--- | :--- | :--- | :--- |
| SD v1.5 (Original) | 90.34 | 31.42 | Unforgotten |
| CP (Training-free pruning) | 7.34 | 27.93 | Good unlearning, quality collapse |
| UCE (Closed-form edit) | 8.62 | 29.25 | Training-free baseline |
| SalUn (Fine-tuned) | 23.17 | 29.93 | Top fine-tuning method |
| SPM (LoRA) | 47.29 | 30.77 | Fine-tuning |
| MACE (LoRA+CFR) | 78.22 | 31.05 | Specialized multi-concept method |
| **FIA (Ours)** | **1.9** | 29.56 | Training-free, near-complete unlearning |

**Imagenette Forget First 5 / Preserve Last 5 (overall = harmonic mean(P, 1-F))**:

| Method | Forget Acc. ↓ | Preserve Acc. ↑ | Overall ↑ |
| :--- | :--- | :--- | :--- |
| CP | 2.7 | 52.4 | 68.1 |
| UCE | 5.5 | 71.9 | 81.7 |
| MACE | 58.5 | 78.2 | 54.2 |
| SalUn | 22.3 | 77.4 | 77.5 |
| **FIA** | **2.1** | **76.7** | **86.0** |

**Inappropriate Content Unlearning (I2P, NudeNet detection, SD v1.4)**: FIA reduced total detections of exposed body parts from 743 in the base model to **32** (next best was MACE at 111), while maintaining FID (14.02) and CLIP (31.18) scores comparable to baselines.

**Multi-Artist Style Unlearning (Van Gogh / Monet / Picasso / Da Vinci / Dali simultaneously)**:

| Method | CLIP_a (Style Similarity) ↓ | FSR (Forget Success) ↑ | COCO CLIP ↑ |
| :--- | :--- | :--- | :--- |
| CP | 27.90 | 79.6 | 29.76 |
| MACE | 30.98 | 57.4 | 30.14 |
| SPM | 31.10 | 40.0 | 31.33 |
| **FIA** | **27.45** | **83.4** | 30.56 |

### Ablation Study

| Configuration | Key Metrics | Explanation |
| :--- | :--- | :--- |
| Full FIA | Forget Acc 1.9 / CLIP 29.56 | Complete model |
| Temporal sparsity only | Slight forget increase, significant quality drop | Selected dominant neurons across all layers, hurting generality |
| Spatial sparsity only | Incomplete unlearning | Selected strong general neurons rather than concept-specific ones |
| No agnostic protection | Large CLIP drop | Shared neurons pruned by mistake; quality collapse |
| Increased total sparsity | Forget ≈ constant, quality drops monotonically | FIA is already at the most economical pruning point |

### Key Findings
- The three modules are irreplaceable: Contrastive saliency determines "accuracy," spatio-temporal sparsity determines "stability," and concept-agnostic protection determines "generation capability."
- Total pruning rate is less than 0.3% of the model—confirming that information for multi-concept unlearning is highly concentrated in very few neurons, an interesting discovery regarding model sparsity.
- As the number of target concepts increases from 2 to 10, FIA's unlearning accuracy remains linearly low, while baselines detoriate rapidly.
- The same hyperparameters work across three tasks (objects, styles, inappropriate content), validating the "plug-and-play" promise.

## Highlights & Insights
- **Contrastive + Statistical Significance Pruning Criteria**: Using the difference between concept and base prompt means minus background variance upgrades traditional "importance scores" to a "concept-specificity t-test." This elegant idea could be applied to other semantic pruning scenarios (e.g., style unlearning in LLMs).
- **Observation on Concept-Agnostic Neurons**: Using a simple count to identify "shared foundations" from "concept-specific" neurons avoids the complex engineering of LLM concept graphs or explicit anchors used in prior work.
- **Training-free + 0.3% Sparsity**: This implies no GPU fine-tuning, no added parameters, and instant rollback (by keeping the original mask), making it highly compliant for regulatory purposes.
- **Transferability**: The framework's core—building contrastive response distributions followed by intersection and shared protection—can be applied to LLM unlearning or vision encoder feature pruning.

## Limitations & Future Work
- When target concepts are extremely numerous and semantically overlapping, it may be difficult to find enough "concept-specific" neurons, leading to incomplete unlearning.
- Validated only on SD v1.4/1.5 and SDXL. Whether the sparsity assumptions hold for DiT-based models (PixArt, SD3, Flux) or video diffusion remains unverified.
- Contrastive saliency depends on manually designed base prompts. It is challenging to design neutral prompts for abstract concepts (e.g., "violence"), potentially causing estimation bias.
- Neuron pruning is a non-learnable binary decision; risks of "reactivation" (e.g., via adversarial prompts or textual inversion) still exist.
- Future directions: Gated masks instead of hard 0/1 (inference-time adaptive shutdown based on prompts), extending contrastive saliency to self-attention/MLP layers, and making the agnostic ratio $\alpha$ data-driven.

## Related Work & Insights
- **vs ConceptPrune (CP)**: Both follow the "training-free pruning" route, but CP is for single concepts and relies on fixed thresholds, causing interference in multi-concept settings. FIA improves forget accuracy from 7.34 to 1.9 and CLIP from 27.93 to 29.56.
- **vs MACE / SPM (LoRA-based)**: These require training adapters for each concept and suffer from cumulative quantization errors in closed-form editing. FIA is more economical and effective.
- **vs UCE / SPEED (Closed-form edit)**: These modify cross-attention weights and rely on precise embeddings, often hurting quality. FIA leaves weights untouched and only zeros out selective neurons.
- **vs SalUn (Gradient saliency)**: SalUn requires backpropagation and fine-tuning. FIA approximates neuron contribution via forward activation contrast, saving time and increasing stability.

## Rating
- Novelty: ⭐⭐⭐⭐ The observation of "concept-agnostic neurons" and the statistical construction of contrastive saliency provide a fresh perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive validation across three tasks, multiple baselines, SDXL generalization, and extensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and intuitive illustrations; some notations are slightly dense initially.
- Value: ⭐⭐⭐⭐⭐ Training-free, 0.3% sparse, and plug-and-play—a highly practical baseline for model compliance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Orthogonal Concept Erasure for Diffusion Models](orthogonal_concept_erasure_for_diffusion_models.md)
- [\[CVPR 2026\] Neighbor-Aware Localized Concept Erasure in Text-to-Image Diffusion Models](../../CVPR2026/image_generation/neighbor-aware_localized_concept_erasure_in_text-to-image_diffusion_models.md)
- [\[AAAI 2026\] Mass Concept Erasure in Diffusion Models with Concept Hierarchy](../../AAAI2026/image_generation/mass_concept_erasure_in_diffusion_models_with_concept_hierarchy.md)
- [\[ICML 2026\] Diagnosing and Correcting Concept Omission in Multimodal Diffusion Transformers](diagnosing_and_correcting_concept_omission_in_multimodal_diffusion_transformers.md)
- [\[ICLR 2026\] Concept-TRAK: Understanding how diffusion models learn concepts through concept-level attribution](../../ICLR2026/image_generation/concept-trak_understanding_how_diffusion_models_learn_concepts_through_concept-l.md)

</div>

<!-- RELATED:END -->
