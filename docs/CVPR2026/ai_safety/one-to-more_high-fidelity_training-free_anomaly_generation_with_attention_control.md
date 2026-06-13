---
title: >-
  [Paper Note] One-to-More: High-Fidelity Training-Free Anomaly Generation with Attention Control
description: >-
  [CVPR 2026][AI Safety][anomaly generation] O2MAG proposes a training-free few-shot anomaly generation method that synthesizes diverse and realistic anomalies from a single reference anomaly image via a tri-branch diffusi…
tags:
  - "CVPR 2026"
  - "AI Safety"
  - "anomaly generation"
  - "training-free"
  - "self-attention grafting"
  - "diffusion models"
  - "industrial anomaly detection"
date: 2026-05-08
content_hash: de83199b3d9fb31f
---

# One-to-More: High-Fidelity Training-Free Anomaly Generation with Attention Control

**Conference**: CVPR 2026
**arXiv**: [2603.18093](https://arxiv.org/abs/2603.18093)  
**Code**: N/A  
**Area**: AI Safety / Anomaly Detection
**Keywords**: anomaly generation, training-free, self-attention grafting, diffusion models, industrial anomaly detection

## TL;DR

O2MAG proposes a training-free few-shot anomaly generation method that synthesizes diverse and realistic anomalies from a single reference anomaly image via a tri-branch diffusion process with self-attention grafting (TriAG). It incorporates Anomaly Guidance Optimization (AGO) to align textual semantics and Dual Attention Enhancement (DAE) to ensure complete mask-region filling. The method significantly outperforms existing approaches on downstream anomaly detection benchmarks using MVTec-AD.

## Background & Motivation

1. **Background**: Industrial anomaly detection suffers from severe data imbalance—normal images are abundant while anomaly images are scarce. Existing anomaly synthesis methods fall into two categories: training-based (e.g., DreamBooth fine-tuning, textual inversion) and training-free approaches.
2. **Limitations of Prior Work**: Training-based methods incur substantial computational and storage overhead and are prone to overfitting under few-shot settings. The only prior training-free method, AnomalyAny, manipulates only cross-attention and fails to precisely control anomaly semantics and spatial layout, resulting in insufficiently realistic generations.
3. **Key Challenge**: Industrial defects are extremely rare in Stable Diffusion's training data; simple text prompts cannot accurately describe defect semantics, causing generated images to deviate from the true anomaly distribution.
4. **Goal**: Leverage the intrinsic priors of diffusion models to synthesize diverse and realistic anomalies from a single reference anomaly image without any training.
5. **Key Insight**: PCA analysis of self-attention maps reveals that anomaly foregrounds and normal backgrounds are naturally separated in attention space, enabling cross-branch information transfer by manipulating self-attention K/V features.
6. **Core Idea**: Tri-branch parallel diffusion combined with mask-guided self-attention grafting, extracting foreground defect features from the reference anomaly branch and background features from the normal image branch.

## Method

### Overall Architecture

O2MAG consists of three parallel diffusion processes: a reference anomaly branch (noise obtained by DDIM inversion of the reference anomaly image), a normal image branch (inverted from a normal image), and a target anomaly branch (initialized with normal image noise). Self-attention grafting injects anomaly features into designated mask regions. An AGO module optimizes text embeddings, and a DAE module enhances attention within the mask region.

### Key Designs

1. **Tri-branch Attention Grafting (TriAG)**:

    - **Function**: Transfers visual features of the reference anomaly into a specified region of the target image while preserving the normal background.
    - **Mechanism**: The target branch's Q is kept unchanged; K and V are replaced selectively. Inside the target mask $M_T$, K/V are sourced from the reference anomaly branch (querying only foreground anomaly features within reference mask $M_R$); outside the mask, K/V are sourced from the normal branch (querying only background features outside the mask). The final attention output is $\text{Attn}^*_{T} = M_T \odot \text{Attn}_{fg} + (1-M_T) \odot \text{Attn}_{bg}$.
    - **Design Motivation**: PCA visualizations of self-attention confirm natural separation between anomalous and normal regions in attention space, enabling precise content transfer via K/V manipulation.

2. **Anomaly Guidance Optimization (AGO)**:

    - **Function**: Bridges the gap between text-encoded anomaly semantics and the true visual appearance of real defects.
    - **Mechanism**: The diffusion model is frozen; only the text embedding $\mathbf{e}$ is optimized by minimizing the reconstruction loss $\mathbf{e}^* = \arg\min_{\mathbf{e}} \mathbb{E}[\|\epsilon - \epsilon_\theta(x_t, t, \mathbf{e})\|^2]$, shifting the text embedding from the normal semantic space toward the anomaly space. Optimization runs for 500 steps using Adam with a learning rate of $3 \times 10^{-3}$.
    - **Design Motivation**: Industrial defects are rarely represented in SD's training data, and text prompts such as "a photo of cable with bent wire" cannot accurately encode defect appearance; data-driven embedding alignment is therefore necessary.

3. **Dual Attention Enhancement (DAE)**:

    - **Function**: Ensures that anomalies fully fill the target mask region, preventing weak or incomplete defect generation.
    - **Mechanism**: During denoising at selected timesteps, both self-attention and cross-attention weights within the mask region are amplified, producing stronger model responses to the defect area.
    - **Design Motivation**: Small defect regions tend to be overlooked or attenuated during generation and require active attention reinforcement.

### Loss & Training

No training is required. AGO performs lightweight optimization of text embeddings at inference time only (500 steps). The full pipeline is built on 50-step DDIM sampling with SD v1.5.

## Key Experimental Results

### Main Results

| Method | AP-I (Detection)↑ | AUC-P (Localization)↑ | F1-P (Localization)↑ | Accuracy (Classification)↑ |
|---|---|---|---|---|
| DFMGAN | 93.5 | 86.7 | 59.2 | - |
| AnomalyDiffusion | 99.3 | 98.9 | 78.2 | - |
| DualAnoDiff | 99.4 | 99.1 | 82.6 | 78.5 |
| SeaS | 99.3 | 98.7 | 79.1 | - |
| **O2MAG** | **99.7** | **99.3** | **84.6** | **90.6** |

O2MAG achieves state-of-the-art performance across all metrics; its classification accuracy exceeds the best training-based method by 12.1%.

### Ablation Study

| Configuration | AP-I | F1-P | Note |
|---|---|---|---|
| Full O2MAG | 99.7 | 84.6 | Complete model |
| w/o AGO | 98.9 | 81.2 | Contribution of text embedding optimization |
| w/o DAE | 99.3 | 82.4 | Contribution of attention enhancement |
| w/o TriAG mask | 97.5 | 75.8 | Mask guidance is the core component |

### Key Findings

- Mask guidance in TriAG is the most critical component; removing it causes foreground/background confusion.
- AGO contributes substantially to classification performance (+12.1%), as accurate anomaly semantics are essential for category discrimination.
- A training-free method surpasses training-based methods on downstream anomaly detection for the first time, demonstrating the expressive power of SD's self-attention priors.
- KID and IC-LPIPS metrics confirm that O2MAG achieves superior generation quality and diversity.

## Highlights & Insights

- **Anomaly–Normal Separation in Self-Attention**: PCA visualizations reveal that SD's self-attention naturally encodes semantic separation between foreground and background, a finding generalizable to other image editing tasks.
- **Training-Free Surpasses Training-Based**: The results demonstrate that carefully designed attention manipulation can outperform fine-tuning-based methods, substantially lowering the barrier to anomaly synthesis.
- **Tri-Branch Parallel Design**: By decoupling the anomaly source, background source, and generation target into separate branches, the method achieves precise region-level control.

## Limitations & Future Work

- AGO still requires 500 optimization steps, increasing inference latency.
- The method relies on predefined anomaly masks, which may be difficult to obtain in real-world scenarios.
- Generation quality is limited for extremely small defects.
- Future work could integrate segmentation models such as SAM to automate mask generation.

## Related Work & Insights

- **vs. AnomalyAny**: AnomalyAny manipulates only cross-attention; O2MAG operates on self-attention K/V, achieving finer-grained control.
- **vs. DualAnoDiff**: DualAnoDiff requires training a dedicated defect branch; O2MAG is entirely training-free.
- **vs. MasaCtrl**: O2MAG extends MasaCtrl's attention manipulation paradigm by introducing a tri-branch structure and mask guidance specifically designed for anomaly synthesis.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Tri-branch attention grafting is novel, though it builds on existing attention manipulation concepts.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation on MVTec-AD with multi-dimensional metrics.
- **Writing Quality**: ⭐⭐⭐⭐ Method description is clear and visualizations are thorough.
- **Value**: ⭐⭐⭐⭐ Substantially lowers the training barrier for industrial anomaly synthesis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Training-Free Coverless Multi-Image Steganography with Access Control](../../ICML2026/ai_safety/training-free_coverless_multi-image_steganography_with_access_control.md)
- [\[CVPR 2026\] Your Classifier Can Do More: Towards Balancing the Gaps in Classification, Robustness, and Generation](your_classifier_can_do_more_towards_balancing_the.md)
- [\[AAAI 2026\] Matrix-Free Two-to-Infinity and One-to-Two Norms Estimation](../../AAAI2026/ai_safety/matrix-free_two-to-infinity_and_one-to-two_norms_estimation.md)
- [\[ICML 2026\] SORA: Free Second-Order Attacks in Fast Adversarial Training](../../ICML2026/ai_safety/sora_free_second-order_attacks_in_fast_adversarial_training.md)
- [\[ICLR 2026\] AP-OOD: Attention Pooling for Out-of-Distribution Detection](../../ICLR2026/ai_safety/ap-ood_attention_pooling_for_out-of-distribution_detection.md)

</div>

<!-- RELATED:END -->
