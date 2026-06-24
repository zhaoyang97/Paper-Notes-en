---
title: >-
  [Paper Note] Mimic In-Context Learning for Multimodal Tasks
description: >-
  [CVPR 2025][Multimodal VLM][In-Context Learning] This paper mathematically analyzes the "shifting effect" of in-context demonstrations (ICDs) on self-attention in ICL. It proposes the MimIC method, which simulates ICL behavior by inserting a learnable shift vector and a query-dependent scaling factor into each attention head. With only 0.26M parameters, MimIC outperforms 32-shot ICL and all existing shift vector methods on VQA and image captioning tasks.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "In-Context Learning"
  - "shift vector"
  - "multi-head attention"
  - "query dependency"
  - "layer alignment"
  - "parameter-efficient"
date: 2026-05-08
content_hash: 024d0cceb70d239b
---

# Mimic In-Context Learning for Multimodal Tasks

**Conference**: CVPR 2025  
**arXiv**: [2504.08851](https://arxiv.org/abs/2504.08851)  
**Code**: [GitHub](https://github.com/Kamichanw/MimIC)  
**Area**: Multimodal VLM  
**Keywords**: In-Context Learning, shift vector, multi-head attention, query dependency, layer alignment, parameter-efficient

## TL;DR

This paper mathematically analyzes the "shifting effect" of in-context demonstrations (ICDs) on self-attention in ICL. It proposes the MimIC method, which simulates ICL behavior by inserting a learnable shift vector and a query-dependent scaling factor into each attention head. With only 0.26M parameters, MimIC outperforms 32-shot ICL and all existing shift vector methods on VQA and image captioning tasks.

## Background & Motivation

Large Multimodal Models (LMMs) can generalize to new tasks from a few examples through In-Context Learning (ICL), but the collaborative effects of multimodal data make ICL performance extremely sensitive to ICD configurations (selection, ordering). $\rightarrow$ Directly increasing the number of ICDs leads to a dramatic surge in computational cost due to excessive image tokens, and current LMMs typically only support up to 32-shot. $\rightarrow$ Key Challenge: How to achieve equivalent or even superior ICL performance without requiring actual ICDs? $\rightarrow$ Prior works have found that mathematically, ICDs are equivalent to adding a "shift vector" to the query hidden states. However, existing methods (TV/FV/LIVE) suffer from three approximation flaws: (1) the shift vector is placed after the FFN rather than after the attention layer, (2) all heads share the same shift vector, and (3) the shift magnitude is query-independent. $\rightarrow$ Core Idea: Achieve a more rigorous approximation of the shifting effect of ICL by inserting learnable vectors inside each attention head, coupled with a query-dependent scaling factor and a layer-wise alignment loss to achieve more precise ICL emulation.

## Method

### Overall Architecture

MimIC replaces all self-attention heads in the original LMM with MimIC Attention Heads. Inside each head, a learnable shift vector $\mathbf{v} \in \mathbb{R}^{d_h}$ and a linear layer $f(\cdot)$ are inserted to approximate the shifting effect brought by ICDs. During training, the original LMM processes $\{X_D, X\}$ to generate ICL hidden states $\mathcal{H}'$, while the MimIC LMM processes only $X$ to generate the shifted hidden states $\mathcal{H}$, co-optimized via an alignment loss and a task loss. During inference, the MimIC LMM is used directly without any ICDs.

### Key Designs

1. **Shift Vector Insertion Location—After Attention Layer Instead of After FFN**:
    - Function: Apply the shifting effect immediately after the attention computation.
    - Mechanism: Mathematical derivation (Eq. 2) shows that the effect of ICDs occurs during the self-attention stage, which decomposes into standard attention + a shift term; prior methods incorrectly insert the vector after the FFN.
    - Design Motivation: Inserting after attention allows each head to learn its own shift direction in its independent representation space, which is more mathematically consistent.

2. **Independent Learnable Shift Vector for Each Head**:
    - Function: Assign an independent $\mathbf{v} \in \mathbb{R}^{d_h}$ to each attention head.
    - Mechanism: In the multi-head attention of Transformers, each head has its own representation space; sharing a single shift vector ignores the differences between heads.
    - Design Motivation: Ablation studies confirm that Head-sharing $\mu$ yields a 1.75% lower accuracy on VQAv2 compared to MimIC, indicating that the per-head design is crucial.

3. **Query-Dependent Shift Magnitude $\tilde{\mu}(\mathbf{q}, \mathbf{K})$**:
    - Function: Dynamically scale the shift vector based on the current query.
    - Mechanism: Use a linear layer $f: \mathbb{R}^{d_h} \to \mathbb{R}$ to approximate $\log Z_1(\mathbf{q}, \mathbf{K}_D)$, and then compute $\tilde{\mu} = \tilde{Z_1}(\mathbf{q})/(\tilde{Z_1}(\mathbf{q}) + Z_2(\mathbf{q}, \mathbf{K}))$.
    - Design Motivation: As shown in Eq. 3, $\mu$ in raw ICL depends on both the query and the ICD keys. A fixed, query-independent $\mu$ fails to distinguish the shift magnitudes required by different queries.

### Loss & Training

Total loss $\mathcal{L} = \mathcal{L}_{\text{align}} + \lambda \mathcal{L}_{\text{gt}}$, where:

- **Layer-wise Alignment Loss**: $\mathcal{L}_{\text{align}} = \frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{l_q}\|\mathbf{h}_{i,j} - \mathbf{h}'_{i,j}\|_2^2$, ensuring that the hidden states of MimIC LMM at each layer align with those of the ICL LMM.
- **Language Modeling Loss** $\mathcal{L}_{\text{gt}}$: Standard ground truth cross-entropy loss.
- $\lambda=0.5$. During training, 32 (for Idefics1) or 8 (for Idefics2) samples are randomly selected as ICDs for each step. Only 1000 training samples are required.
- Optimizer: AdamW, learning rate $5 \times 10^{-3}$, cosine annealing + 10% warmup.

## Key Experimental Results

### Main Results

| Dataset | Metric | MimIC | 32-shot ICL | LIVE | LoRA | Gain (vs ICL) |
|--------|------|-------|-------------|------|------|-------------|
| VQAv2 (Idefics1) | Accuracy | **59.64** | 56.18 | 53.71 | 55.60 | +3.46% |
| OK-VQA (Idefics1) | Accuracy | **52.05** | 48.48 | 46.05 | 47.06 | +3.57% |
| COCO Caption (Idefics1) | CIDEr | **114.89** | 105.89 | 112.76 | 97.75 | +9.00 |
| VQAv2 (Idefics2) | Accuracy | **69.29** | 66.20 | 67.60 | 66.54 | +3.09% |
| OK-VQA (Idefics2) | Accuracy | **58.74** | 57.68 | 54.86 | 55.05 | +1.06% |
| COCO Caption (Idefics2) | CIDEr | **132.87** | 122.51 | 126.04 | 116.69 | +10.36 |

MimIC requires only 0.26M parameters (compared to 25M for LoRA), which is double the parameters of LIVE but substantially outperforms it.

### Ablation Study

| Configuration | VQAv2 | OK-VQA | COCO | Description |
|------|-------|--------|------|------|
| MimIC (full) | 59.64 | 52.05 | 114.89 | Full method |
| Head-sharing $\mu$ | 57.89 | 50.86 | 111.98 | All heads share $\mu$, -1.75% |
| Query-sharing $\mu$ | 57.95 | 50.94 | 112.48 | Fixed $\mu$ independent of query, -1.69% |

| Method | L2 Distance (VQAv2) | L2 Distance (OK-VQA) | Description |
|------|-------------|---------------|------|
| Zero-shot | 42.97 | 41.21 | Furthest from ICL |
| LIVE | 33.79 | 34.12 | Aligned using KL divergence |
| MimIC† (KL) | 32.13 | 29.76 | MimIC replacing L2 with KL |
| MimIC | **30.17** | **28.25** | L2 alignment is more effective |

### Key Findings

- MimIC requires only 200 training samples to outperform 32-shot ICL, whereas LIVE requires roughly 8 times more data.
- Under 1-shot training, MimIC can match the generalization ability of 32-shot ICL, suggesting that it uncovers a generalized shifting pattern.
- Hallucination Analysis: The CHAIRs/CHAIRi metrics of MimIC (8.51/5.74) are significantly lower than those of 32-shot ICL (16.78/9.77), while achieving a higher recall (43.30 vs 42.59).

## Highlights & Insights

- Rigorous mathematical derivation: Starting from the decomposition of self-attention, it precisely pinpoints three approximation flaws of prior methods and corrects them one by one.
- Extremely high parameter efficiency: Only 0.26M parameters (approx. 1% of LoRA) while achieving comprehensive outperformance across all tasks.
- Significant inference efficiency gain: Eliminates the need to process long sequences of ICDs, enabling direct zero-shot inference.
- Excellent hallucination mitigation: Produces fewer hallucinations than standard ICL and other baselines.

## Limitations & Future Work

- Evaluation is limited to Idefics1/2, without testing on other mainstream LMMs such as LLaVA and Qwen-VL.
- Generating target hidden states for alignment still requires executing ICL on the original LMM during training, which incurs high training overhead.
- MimIC parameters need to be trained separately for each task; cross-task generalization capability remains unverified.
- The linear layer $f(\cdot)$ approximation of $\log Z_1$ might lose accuracy under extreme feature distributions.

## Related Work & Insights

- **LIVE** [Peng 2024]: The most direct predecessor, which uses learnable vectors after the FFN to simulate ICL. MimIC substantially outperforms it through a more precise attention-level approximation.
- **Task Vector / Function Vector**: Training-free heuristic methods that show limited performance on multimodal tasks.
- Insight: Tightly coupling theoretical analysis (mathematical derivation) with empirical design. "The devil is in the details"—subtle design differences (insertion location, per-head vs. shared) can lead to massive performance discrepancies.

## Rating
- Novelty: ⭐⭐⭐⭐ Although the concept of shift vectors is not brand new, identifying flaws via rigorous mathematical analysis and correcting them represents a significant incremental innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Highly comprehensive; covers two LMMs, three tasks, exhaustive ablation studies, L2 distance analysis, hallucination analysis, etc.
- Writing Quality: ⭐⭐⭐⭐⭐ The mathematical derivation is clear, diagrams are intuitive, and the storyline is fluent.
- Value: ⭐⭐⭐⭐ Holds distinct value in the lines of ICL efficiency and robustness, though practical impact depends on validation across more mainstream LMMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] HiFICL: High-Fidelity In-Context Learning for Multimodal Tasks](hificl_high-fidelity_in-context_learning_for_multimodal_tasks.md)
- [\[CVPR 2025\] SymDPO: Boosting In-Context Learning of Large Multimodal Models with Symbol Demonstration Direct Preference Optimization](symdpo_boosting_in-context_learning_of_large_multimodal_models_with_symbol_demon.md)
- [\[CVPR 2025\] Context-Aware Multimodal Pretraining](context-aware_multimodal_pretraining.md)
- [\[CVPR 2025\] DynRefer: Delving into Region-level Multimodal Tasks via Dynamic Resolution](dynrefer_delving_into_region-level_multimodal_tasks_via_dynamic_resolution.md)
- [\[CVPR 2025\] Cropper: Vision-Language Model for Image Cropping through In-Context Learning](cropper_vision-language_model_for_image_cropping_through_in-context_learning.md)

</div>

<!-- RELATED:END -->
