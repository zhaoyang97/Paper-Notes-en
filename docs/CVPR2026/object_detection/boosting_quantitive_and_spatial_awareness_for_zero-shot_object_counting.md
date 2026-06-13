---
title: >-
  [Paper Note] Boosting Quantitive and Spatial Awareness for Zero-Shot Object Counting
description: >-
  [CVPR 2026][Object Detection][zero-shot counting] The QICA framework addresses the lack of quantity awareness and spatial insensitivity in zero-shot object counting by using a quantity-conditioned Synergistic Prompting S…
tags:
  - "CVPR 2026"
  - "Object Detection"
  - "zero-shot counting"
  - "vision-language model"
  - "prompt tuning"
  - "cost aggregation"
  - "quantity awareness"
date: 2026-05-08
content_hash: 1cb3803896e3d173
---

# Boosting Quantitive and Spatial Awareness for Zero-Shot Object Counting

**Conference**: CVPR 2026  
**arXiv**: [2603.16129](https://arxiv.org/abs/2603.16129)  
**Code**: Coming soon  
**Area**: LLM / NLP (Other)  
**Keywords**: zero-shot counting, vision-language model, prompt tuning, cost aggregation, quantity awareness

## TL;DR
The QICA framework addresses the lack of quantity awareness and spatial insensitivity in zero-shot object counting by using a quantity-conditioned Synergistic Prompting Strategy (SPS) to jointly adapt vision-language encoders, combined with a Cost Aggregation Decoder (CAD) operating on similarity maps to preserve zero-shot transferability, achieving zero-shot SOTA on FSC-147 (MAE 12.41) with strong cross-domain generalization.

## Background & Motivation

**Background**: Zero-shot object counting (ZSOC) aims to enumerate arbitrary-category objects using only text descriptions. Mainstream methods leverage VLMs such as CLIP to compute vision-text similarity maps, then use CNN/Transformer decoders to predict density maps.

**Limitations of Prior Work**: (1) **Lack of quantity awareness** — text prompts specify only categories without quantity information; models excel at recognizing "what" but fail to understand "how many," especially in dense scenes. (2) **Spatial insensitivity + feature space distortion** — directly fine-tuning VLM encoders leads to severe overfitting to training categories, corrupting the pre-trained feature space and harming zero-shot generalization.

**Key Challenge**: Accurate counting requires adapting encoders to learn quantity-sensitive features, but fine-tuning corrupts zero-shot generalization — creating an adaptation-vs-generalization dilemma.

**Goal**: (1) Enable fine-grained quantity discrimination; (2) Achieve effective adaptation without corrupting the pre-trained feature space.

**Key Insight**: Introduce quantity-conditioned prompts for encoders to implicitly learn quantity information, while operating on similarity maps (rather than feature space) to avoid feature distortion.

**Core Idea**: During training, factual/counterfactual quantity prompts teach the model to distinguish different quantities; at inference, only category prompts are used for zero-shot counting.

## Method

### Overall Architecture
Input: image $I$ and text description $T$; output: density map $D$. The framework comprises three core modules: (1) SPS jointly adapts frozen CLIP visual and language encoders via quantity-conditioned learnable prompts; (2) CAD performs spatial aggregation on vision-text similarity maps with multi-scale upsampling to predict density maps; (3) $\mathcal{L}_{MQA}$ enforces quantity consistency at both encoder and decoder levels. Key design: quantity-information pathways active during training are disabled at inference, retaining only category semantic pathways.

### Key Designs

1. **Synergistic Prompting Strategy (SPS)**

    - Function: Jointly adapts visual and language encoders via quantity-conditioned learnable prompts
    - Mechanism: Maps quantity value $q$ to continuous embedding $\epsilon_q$, added to per-layer learnable prompts $\Pi^j$ to generate quantity-aware text prompts $\hat{\Pi}^j_k = \Pi^j + \epsilon_{q_k}\mathbf{1}^T$. A coupling function $\Phi^j$ projects language prompts to visual prompts $\Psi^j_k = \Phi^j(\hat{\Pi}^j_k)$, enabling bidirectional gradient flow. During training, both factual (true quantity) and counterfactual (deviated quantity) prompts are generated per image
    - Design Motivation: Independent unimodal prompts cannot achieve cross-modal quantity-aware coordination. The coupling function establishes a direct language→vision gradient pathway, enabling both encoders to jointly adapt toward quantity awareness

2. **Cost Aggregation Decoder (CAD)**

    - Function: Operates directly on vision-text cosine similarity maps to produce density maps via spatial aggregation
    - Mechanism: Computes per-position cosine similarity between dense visual features $\mathbf{V}$ and category text embeddings $\mathbf{T}^{\text{cat}}_k$ to obtain similarity maps $\mathbf{S}_k$; then applies embedding layer → Swin Transformer spatial aggregation → multi-scale upsampling (with skip connections and similarity gating) → prediction head for final density map
    - Design Motivation: Operating in feature space corrupts the pre-trained manifold and causes overfitting, while aggregating on similarity maps (a scalar field) preserves embedding space integrity, enabling encoder fine-tuning without harming generalization

3. **Multi-level Quantity Alignment Loss ($\mathcal{L}_{MQA}$)**

    - Function: Enforces quantity consistency constraints at both encoder and decoder levels
    - Mechanism: Encoder level uses ranking loss to ensure the true quantity hypothesis has the highest global similarity $\alpha_0 > \alpha_i$, with closer quantities yielding higher similarity; decoder level uses auxiliary MSE loss to constrain density map integrals to match corresponding quantity values. Total loss: $\mathcal{L}_{MQA} = \|D^0 - D^{GT}\|_2^2 + \lambda_1 \mathcal{L}^{qty}_{enc} + \lambda_2 \mathcal{L}^{qty}_{dec}$
    - Design Motivation: Decoder-only supervision is insufficient; encoder-level ranking constraints implicitly encode quantity information in the feature space during training, enabling accurate counting at inference without quantity prompts

### Loss & Training
- Density map MSE + encoder ranking loss ($\lambda_1=0.1$) + decoder counting loss ($\lambda_2=0.05$)
- During training, $K$ quantity hypotheses (factual + counterfactual) are generated per image with independent forward passes sharing encoder parameters
- At inference, no quantity information is needed — only category text input

## Key Experimental Results

### Main Results

**FSC-147 Zero-Shot Counting**

| Method | Backbone | Val MAE↓ | Val RMSE↓ | Test MAE↓ | Test RMSE↓ |
|--------|----------|----------|-----------|-----------|------------|
| CounTX | ViT-B/16 | 17.76 | 65.21 | 16.70 | 105.21 |
| VLCounter | ViT-B/16 | 18.06 | 65.13 | 17.05 | 106.16 |
| T2ICount | SD-v1.5 | 13.78 | 58.78 | 11.76 | 97.86 |
| CountGD | GDINO-Swin-B | 12.14 | 47.51 | 14.76 | 120.42 |
| **QICA** | ViT-B/16 | **13.82** | **60.24** | **13.05** | 104.17 |
| **QICA†** | ViT-L/14 | **12.98** | **56.35** | **12.41** | - |

### Ablation Study

| Configuration | Val MAE | Test MAE | Note |
|---------------|---------|----------|------|
| Baseline (CLIP + Conv decoder) | ~18 | ~17 | No quantity awareness |
| + SPS (text prompts only) | ~16 | ~15 | Limited improvement with unimodal prompts |
| + SPS (synergistic prompts) | ~15 | ~14 | Significant boost from bimodal coupling |
| + CAD | ~14 | ~13.5 | Spatial aggregation further optimizes |
| + $\mathcal{L}_{MQA}$ (full model) | 13.82 | 13.05 | Multi-level constraints yield final results |

### Key Findings
- QICA significantly outperforms all zero-shot methods with the same backbone (ViT-B/16): CounTX MAE 16.70 → QICA 13.05
- Cross-domain generalization tests (CARPK, ShanghaiTech-A) surpass all baselines, confirming no overfitting
- The coupling function in SPS improves MAE by ~1.5 over independent prompting, demonstrating bimodal synergy is crucial
- CAD reduces MAE by ~1–2 points compared to feature-space decoding while preserving zero-shot capability
- Quantity ranking loss contributes more significantly in dense scenes (high object counts)

## Highlights & Insights
- **Elegant train-inference consistency design**: During training, the full quantity-aware embedding $T^{full}$ produces category embedding $T^{cat}$ via projection; at inference, semantically equivalent category embeddings are naturally produced — quantity knowledge is "distilled" into the visual encoder's implicit representations
- **Operating on similarity maps rather than feature space**: This design choice is critical — CAD operates on a scalar field (similarity map), causing zero damage to the pre-trained feature space, solving the longstanding VLM fine-tuning problem
- **Factual/counterfactual quantity prompts**: Teaching quantity awareness through contrastive learning with "correct quantity" vs "wrong quantity" is more discriminative than simply adding quantity labels

## Limitations & Future Work
- Still depends on pre-trained VLM category recognition capabilities; may fail for extremely rare objects unseen by the VLM
- Multiple forward passes for K quantity hypotheses increase training overhead
- Density map MSE loss may be imbalanced in extremely sparse or extremely dense scenes
- The quantity awareness mechanism could be extended to open-set detection/segmentation tasks

## Related Work & Insights
- **vs CLIP-Count/CounTX**: They freeze encoders to avoid overfitting but also sacrifice adaptation capability; QICA resolves this dilemma by operating CAD on similarity maps
- **vs CountGD**: CountGD uses a larger GDINO backbone + visual exemplars; QICA achieves comparable performance using only text + ViT-B
- **vs T2ICount**: Stable Diffusion-based methods have stronger generative priors but higher inference cost and different backbones make fair comparison difficult

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of quantity-aware prompts and cost aggregation decoder is creative, with elegant train-inference consistency design
- Experimental Thoroughness: ⭐⭐⭐⭐ FSC-147 + cross-domain validation + rich ablations
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, detailed method description
- Value: ⭐⭐⭐⭐ Practical contribution to zero-shot counting

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CoPS: Conditional Prompt Synthesis for Zero-Shot Anomaly Detection](cops_conditional_prompt_synthesis_for_zero-shot_anomaly_detection.md)
- [\[CVPR 2026\] MoECLIP: Patch-Specialized Experts for Zero-shot Anomaly Detection](moeclip_patch-specialized_experts_for_zero-shot_anomaly_detection.md)
- [\[CVPR 2026\] AnomalyVFM -- Transforming Vision Foundation Models into Zero-Shot Anomaly Detectors](anomalyvfm_--_transforming_vision_foundation_models_into_zero-shot_anomaly_detec.md)
- [\[CVPR 2026\] VisualAD: Language-Free Zero-Shot Anomaly Detection via Vision Transformer](visualad_language-free_zero-shot_anomaly_detection_via_vision_transformer.md)
- [\[CVPR 2026\] Back to Point: Exploring Point-Language Models for Zero-Shot 3D Anomaly Detection](back_to_point_exploring_point-language_models_for_zero-shot_3d_anomaly_detection.md)

</div>

<!-- RELATED:END -->
