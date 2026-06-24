---
title: >-
  [Paper Note] Synergistic Prompting for Robust Visual Recognition with Missing Modalities
description: >-
  [ICCV 2025][Multimodal VLM][missing modality] This paper proposes the Synergistic Prompting (SyP) framework, which employs a dynamic adapter to generate input-adaptive scaling factors that modulate a base prompt (dynamic prompt), synergizing with a static prompt that captures shared cross-modal features. SyP achieves robust visual recognition under missing-modality conditions and consistently outperforms SOTA methods such as DCP on MM-IMDb, Food101, and Hateful Memes.
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "missing modality"
  - "dynamic prompt"
  - "synergistic prompting"
  - "CLIP"
  - "multi-modal learning"
date: 2026-05-08
content_hash: 804f1c0d08c3b647
---

# Synergistic Prompting for Robust Visual Recognition with Missing Modalities

**Conference**: ICCV 2025
**arXiv**: [2507.07802](https://arxiv.org/abs/2507.07802)  
**Code**: N/A  
**Area**: Multimodal VLM / Missing Modality Learning / Prompt Learning
**Keywords**: missing modality, dynamic prompt, synergistic prompting, CLIP, multi-modal learning

## TL;DR
This paper proposes the Synergistic Prompting (SyP) framework, which employs a dynamic adapter to generate input-adaptive scaling factors that modulate a base prompt (dynamic prompt), synergizing with a static prompt that captures shared cross-modal features. SyP achieves robust visual recognition under missing-modality conditions and consistently outperforms SOTA methods such as DCP on MM-IMDb, Food101, and Hateful Memes.

## Background & Motivation

**Background**: Multimodal learning has achieved notable progress in cross-modal retrieval, VQA, and related tasks, primarily leveraging large-scale paired datasets and pretrained multimodal Transformers (e.g., CLIP). In real-world deployments, however, sensor failures, privacy constraints, and data collection difficulties frequently result in incomplete modality inputs.

**Taxonomy of Existing Approaches**:
- **Joint learning**: Models cross-modal correlations by aligning latent feature spaces, but relies on masking or imputation strategies that introduce noise.
- **Cross-modal generation**: Attempts to reconstruct missing modalities from available ones, but modal heterogeneity leads to poor reconstruction quality and high computational overhead.
- **Prompt-based methods**: Leverage learnable prompts to adapt pretrained models in a parameter-efficient manner, yet suffer from two fundamental limitations.

**Two Core Limitations of Existing Prompt Methods**:
- **(1) Static prompts lack flexibility**: All inputs share identical prompt embeddings regardless of which modality is missing or the degree of missingness, making adaptation to dynamic real-world scenarios impossible.
- **(2) Lack of inter-layer synergy**: Simple prompt tuning fails to adequately exploit multimodal dependencies encoded in hierarchical model representations, leading to unreliable performance when critical modalities are absent.

**Key Challenge**: A fundamental mismatch exists between the "one-size-fits-all" design of static prompts and the highly dynamic nature of modality-missing patterns in real-world scenarios. When a critical modality is absent, the model must adaptively amplify the contribution of available modalities—a capability that static approaches cannot provide.

**Key Insight**: The paper combines dynamic prompts (input-adaptive) with static prompts (preserving pretrained knowledge) into a "synergistic prompting" strategy that simultaneously ensures flexibility and stability.

## Method

### Overall Architecture
SyP is built upon CLIP's dual-stream architecture (image encoder + text encoder), freezing the backbone and updating only the prompts and FC layers. For each input, the modality-missing type $m \in \{c, m_1, m_2\}$ (complete, missing image, missing text) determines a modality-specific synergistic prompt $P_m^I$ (image branch) and $P_m^T$ (text branch), which are prepended to the input token sequence.

### Key Design 1: Dynamic Adapter

The dynamic adapter computes input-adaptive scaling factors based on the **available modality features of the current input** to dynamically modulate the strength of the base prompt.

**Feature concatenation**: Image features $X_I \in \mathbb{R}^{d_I}$ and text features $X_T \in \mathbb{R}^{d_T}$ are concatenated into a joint feature vector:

$$X_C = [X_I, X_T]$$

**Scaling factor computation**: A bottleneck MLP computes the scaling factor:

$$S_d = \sigma\left(\text{MLP}\left(\text{ReLU}\left(\frac{1}{r}W_1 X_C + b_1\right)\right)W_2 + b_2\right)$$

where $r$ is the dimensionality reduction ratio and sigmoid constrains $S_d \in [0,1]$. Larger $S_d$ amplifies prompt influence; smaller $S_d$ attenuates it, enabling adaptive modulation according to the relevance of each modality.

**Dynamic prompt generation**: The scaling factor is applied element-wise to the base prompt:

$$P_{I,D} = P_{I,B} \odot S_d, \quad P_{T,D} = P_{T,B} \odot S_d$$

When a modality is absent, the scaling factor increases the weight of the corresponding prompt; when modalities are complete, it reduces it, achieving adaptive modality-weight allocation.

### Key Design 2: Synergistic Prompting Strategy

**Static prompt**: Captures shared features between image and text modalities, projected into each modality space via learned linear projection functions:

$$P_{I,S} = G_I(P_S), \quad P_{T,S} = G_T(P_S)$$

**Final synergistic prompt**: Dynamic and static prompts are summed element-wise:

$$P_m^I = P_{I,D} + P_{I,S}, \quad P_m^T = P_{T,D} + P_{T,S}$$

This combination ensures the model simultaneously exploits modality-specific adaptive adjustments (dynamic prompt) and shared cross-modal features (static prompt).

### Key Design 3: Inter-Layer Prompt Propagation

Prompts are recursively propagated across Transformer layers; the prompt at layer $i$ is derived from the preceding layer's prompt via a transformation function:

$$P_m^{I,R_i} = F_i(P_m^{I,R_{i-1}}) = \text{LN}(\text{FC}(\text{GeLU}(\text{FC}(P_m^{I,R_{i-1}}))))$$

This ensures that each layer's prompt integrates learned representations from the previous layer together with current-layer input features, capturing hierarchical multimodal representations.

### Loss & Training

Standard task-specific loss functions are employed. The final multimodal prompt feature is obtained by concatenating image and text prompts followed by an FC layer:

$$P_{\text{final}}^{(i)} = \text{FC}(P_m^{I,R_{i-1}} \| P_m^{T,R_{i-1}})$$

The total training loss is the sum of per-sample task losses: $\mathcal{L}_{\text{total}} = \sum_{i=1}^{N} \mathcal{L}_i(P_{\text{final}}^{(i)})$

## Key Experimental Results

### Datasets & Setup
- **MM-IMDb**: 25,959 image-text movie samples, multi-label genre classification, F1-Macro.
- **UPMC Food-101**: 101 food categories with noisy image-text pairs, Top-1 Accuracy.
- **Hateful Memes**: 10,000+ hateful meme detection samples, AUROC.
- Backbone: CLIP ViT-B/16; prompt length $L_p=36$; applied to $M=6$ layers.
- AdamW, lr=1e-3, weight decay=2e-2, 20 epochs, batch size=32.

### Main Results

| Dataset | Missing Rate | Missing Modality | DCP | SyP (Ours) | Gain |
|--------|--------|----------|-----|------------|------|
| MM-IMDb (F1) | 50% | Both (balanced) | 52.32 | 55.02 | +2.70 |
| MM-IMDb (F1) | 70% | Both (balanced) | 51.42 | 52.90 | +1.48 |
| MM-IMDb (F1) | 90% | Both (balanced) | 48.04 | 49.63 | +1.59 |
| Food101 (Acc) | 50% | Both (balanced) | 85.24 | 86.17 | +0.93 |
| Food101 (Acc) | 70% | Both (balanced) | 81.87 | 82.45 | +0.58 |
| Food101 (Acc) | 90% | Both (balanced) | 79.87 | 81.03 | +1.16 |
| Hateful Memes (AUROC) | 50% | Both (balanced) | 66.02 | 68.16 | +2.14 |
| Hateful Memes (AUROC) | 70% | Both (balanced) | 66.08 | 68.42 | +2.34 |
| Hateful Memes (AUROC) | 90% | Both (balanced) | 66.78 | 68.93 | +2.15 |

SyP surpasses SOTA DCP across all datasets, missing rates, and missing patterns, with the most pronounced improvements on Hateful Memes (+2–7 percentage points).

### Ablation Study

| Variant | Hateful Memes | Food101 | MM-IMDb |
|------|--------------|---------|---------|
| w/o Synergistic Prompts (classifier fine-tuning only) | 57.35 | 71.59 | 44.63 |
| Dynamic Prompt only | 66.37 | 82.90 | 51.21 |
| Static Prompt only | 65.62 | 83.06 | 50.34 |
| **SyP (Dynamic + Static Synergy)** | **68.16** | **86.17** | **54.72** |

| Variant | Hateful Memes | Food101 | MM-IMDb |
|------|--------------|---------|---------|
| Base Prompt only | 64.27 | 81.68 | 48.95 |
| Base Prompt + Dynamic Adapter | 66.37 | 82.90 | 51.21 |
| Synergistic Prompts (w/o Dynamic Adapter) | 66.19 | 84.85 | 51.90 |
| **Synergistic Prompts + Dynamic Adapter** | **68.16** | **86.17** | **54.72** |

### Key Findings
1. The dynamic–static synergy outperforms either strategy alone by 3–4 percentage points, demonstrating that "synergy" is the critical factor.
2. The dynamic adapter contributes an additional ~2 percentage points on top of synergistic prompts, validating the effectiveness of adaptive scaling.
3. At high missing rates (90%), SyP exhibits substantially smaller performance degradation than baselines, demonstrating superior robustness.
4. Dataset-specific characteristics differ: MM-IMDb and Food101 rely more heavily on textual semantics (text missingness is more detrimental), while Hateful Memes is more sensitive to image missingness.

## Highlights & Insights
1. **The complementary dynamic–static design is elegant**: Dynamic prompts handle adaptive adjustment (flexibility), while static prompts maintain the pretrained knowledge base (stability); their element-wise summation is simple yet effective.
2. **The scaling factor design has clear intuition**: When a modality is absent, its input is a zero tensor; the concatenated joint feature naturally skews toward the available modality, and the sigmoid MLP output adjusts the scaling factor accordingly—without any explicit missing-modality detection mechanism.
3. **Parameter efficiency**: The CLIP backbone is frozen; only prompts and FC layers are trained, introducing minimal additional parameters.

## Limitations & Future Work
1. Validation is limited to $M=2$ modality settings; extension to three or more modalities has not been explored.
2. Missing modalities are replaced by zero-filled tensors; more sophisticated imputation strategies remain unexplored.
3. Experiments are confined to classification tasks; generative tasks (captioning, VQA) are not addressed.
4. Sensitivity analysis of hyperparameters such as the dimensionality reduction ratio $r$ of the dynamic adapter is insufficient.

## Related Work & Insights
- **Missing modality learning**: Joint learning (SMIL, ShaSpec), cross-modal generation (LEL, MTP), prompt-based methods (MMP, DePT, DCP).
- **Prompt Learning**: CoOp, MaPLe, DePT, DCP, and related prompt tuning methods.
- **Multimodal pretraining**: CLIP, ViT, and related backbone architectures.

## Rating
- **Novelty**: 3/5 (The dynamic–static prompt combination has moderate novelty, but the overall framework is relatively engineering-driven.)
- **Technical Depth**: 3/5 (The method is clean but lacks theoretical analysis.)
- **Experimental Thoroughness**: 4/5 (Three datasets, multiple missing rates, and comprehensive ablation studies.)
- **Writing Quality**: 3/5 (Generally clear but somewhat verbose.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] MoRA: Missing Modality Low-Rank Adaptation for Visual Recognition](../../ICLR2026/multimodal_vlm/mora_missing_modality_low-rank_adaptation_for_visual_recognition.md)
- [\[CVPR 2025\] Recognition-Synergistic Scene Text Editing](../../CVPR2025/multimodal_vlm/recognition-synergistic_scene_text_editing.md)
- [\[ICML 2026\] Calibrated Multimodal Representation Learning with Missing Modalities](../../ICML2026/multimodal_vlm/calibrated_multimodal_representation_learning_with_missing_modalities.md)
- [\[ICCV 2025\] Generalizable Object Re-Identification via Visual In-Context Prompting](generalizable_object_re-identification_via_visual_in-context_prompting.md)
- [\[CVPR 2026\] Beyond Missing Modalities: Hypergraph Guided Diffusion for Uncertainty-Aware Multimodal Emotion Recognition](../../CVPR2026/multimodal_vlm/beyond_missing_modalities_hypergraph_conditioned_diffusion_for_uncertainty-aware.md)

</div>

<!-- RELATED:END -->
