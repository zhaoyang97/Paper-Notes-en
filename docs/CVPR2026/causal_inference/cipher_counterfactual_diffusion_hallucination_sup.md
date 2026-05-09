---
title: >-
  [Paper Note] Fighting Hallucinations with Counterfactuals: Diffusion-Guided Perturbations for LVLM Hallucination Suppression
description: >-
  [CVPR 2026][Causal Inference][LVLM hallucination] This paper proposes CIPHER, a training-free test-time hallucination suppression method. It generates semantically altered yet structurally preserved counterfactual images via a diffusion model, applies SVD decomposition to the representation differences between original and counterfactual images in LVLM hidden layers to extract a hallucination subspace, and then projects hidden states onto the orthogonal complement of this subspace during inference. CIPHER is the first method to localize and mitigate LVLM hallucinations by intervening on the visual modality.
tags:
  - CVPR 2026
  - Causal Inference
  - LVLM hallucination
  - counterfactual image
  - diffusion model
  - feature projection
  - SVD
  - training-free
date: 2026-05-08
content_hash: 733224b69ba21490
---

# Fighting Hallucinations with Counterfactuals: Diffusion-Guided Perturbations for LVLM Hallucination Suppression

**Conference**: CVPR 2026
**arXiv**: [2603.10470](https://arxiv.org/abs/2603.10470)
**Code**: [Project Page](https://hamidreza-dastmalchi.github.io/cipher-cvpr2026/)
**Area**: Causal Inference
**Keywords**: LVLM hallucination, counterfactual image, diffusion model, feature projection, SVD, training-free

## TL;DR

This paper proposes CIPHER, a training-free test-time hallucination suppression method. It generates semantically altered yet structurally preserved counterfactual images via a diffusion model, applies SVD decomposition to the representation differences between original and counterfactual images in LVLM hidden layers to extract a hallucination subspace, and then projects hidden states onto the orthogonal complement of this subspace during inference. CIPHER is the first method to localize and mitigate LVLM hallucinations by intervening on the visual modality.

## Background & Motivation

**Background**: Large vision-language models (LVLMs) such as LLaVA, MiniGPT-4, and mPLUG-Owl2 exhibit strong performance on multimodal tasks, yet frequently produce hallucinations — generating descriptions inconsistent with visual inputs, such as fabricating non-existent objects, or mischaracterizing attributes and scenes.

**Limitations of Prior Work**: (1) Training-based methods (e.g., additional supervision signals, architectural modifications) require expensive annotations and retraining, limiting scalability. (2) Post-processing methods (e.g., Woodpecker, LURE) rely on external models for detection and correction, increasing system complexity and limiting generalization. (3) Contrastive decoding test-time methods (e.g., DoLa, VCD, OPERA) require multiple forward passes, incurring high inference overhead (throughput reduced to 0.05–0.42 items/s, far below greedy decoding at 0.70). (4) Existing feature-level intervention methods (e.g., Nullu) extract hallucination directions solely through text perturbation, overlooking hallucinations induced by the visual modality itself.

**Key Challenge**: LVLM hallucinations arise not only from language model generation biases (text-induced) but also from weak visual grounding and modality misalignment (vision-induced). However, existing methods almost exclusively address the former — text perturbation produces weaker and less stable hallucination signals (linear probe accuracy of only 0.73–0.80), whereas vision-induced hallucination directions are more structured and easier to isolate.

**Goal**: To specifically identify and suppress hallucination directions induced by the visual modality, achieving more thorough hallucination mitigation in an efficient, training-free manner.

**Key Insight**: Rather than perturbing text to identify hallucination directions (as in Nullu), the paper perturbs images — using a diffusion model to generate counterfactual images that are semantically altered but structurally preserved — and treats the representational difference between original and counterfactual images within the LVLM as the hallucination direction.

**Core Idea**: By "making images lie" (diffusion-based editing to produce counterfactual images), CIPHER localizes the feature directions of visual hallucinations within LVLMs and eliminates them via orthogonal projection at inference time.

## Method

### Overall Architecture

CIPHER operates in two stages: (1) an offline stage — constructing the OHC-25K counterfactual dataset and extracting hallucination subspace bases for each layer via SVD decomposition; and (2) an inference stage — projecting hidden states onto the orthogonal complement of the hallucination subspace at each decoding step. The entire process requires no modification of model parameters, no additional training, and no extra forward passes.

### Key Designs

1. **OHC-25K Counterfactual Dataset Construction**
   - **Function**: Produces images with incorrect visual content but preserved structure, enabling precise localization of vision-induced hallucination directions.
   - **Mechanism**: From the MSCOCO training set, $M=5000$ image-caption pairs $\{(\boldsymbol{I}_i, \mathcal{C}_i)\}$ are selected. GPT-3.5 perturbs each caption to produce a hallucinated version $\tilde{\mathcal{C}}_i$ (injecting plausible but non-existent objects). Each original image is VAE-encoded into latent $\boldsymbol{z}_0 = \mathcal{E}(\boldsymbol{I}_i)$, partially forward-diffused to $\tilde{\boldsymbol{z}}_{t_h} = \sqrt{\bar{\alpha}_{t_h}}\boldsymbol{z}_0 + \sqrt{1-\bar{\alpha}_{t_h}}\boldsymbol{\epsilon}$ at $t_h = 0.5T$, then reverse-diffused conditioned on the hallucinated caption $\tilde{\boldsymbol{z}}_{t-1} = f_\theta(\tilde{\boldsymbol{z}}_t, t, \tilde{\mathcal{C}}_i)$. Each image yields $B=5$ variants; the counterfactual images paired with original captions form 25K pairs.
   - **Design Motivation**: Diffusion at an intermediate step ($0.5T$) preserves global structure while injecting semantically inconsistent elements, precisely simulating "visual hallucination" — visually plausible yet factually incorrect content.

2. **Hallucination Subspace Estimation (SVD Decomposition)**
   - **Function**: Extracts principal hallucination directions from representational differences across a large number of samples.
   - **Mechanism**: For each sample $i$, the mean hidden states of caption tokens $\boldsymbol{h}_\ell^{(i)}$ and $\tilde{\boldsymbol{h}}_\ell^{(i)} = \frac{1}{B}\sum_{j=1}^{B}\tilde{\boldsymbol{h}}_\ell^{(i,j)}$ are extracted at layer $\ell$ of the LVLM for the original and counterfactual pairs, respectively. The difference $\boldsymbol{\delta}_\ell^{(i)} = \tilde{\boldsymbol{h}}_\ell^{(i)} - \boldsymbol{h}_\ell^{(i)}$ is computed. All differences are stacked into matrix $\boldsymbol{\Delta}_\ell \in \mathbb{R}^{M \times d}$, which is decomposed via SVD as $\boldsymbol{\Delta}_\ell = \boldsymbol{U}_\ell \boldsymbol{\Sigma}_\ell \boldsymbol{V}_\ell^\top$. The top $r$ right singular vectors $\boldsymbol{V}_{\ell,r} = [\boldsymbol{v}_{\ell,1}, \ldots, \boldsymbol{v}_{\ell,r}]$ form the hallucination basis.
   - **Design Motivation**: Visual hallucination feature differences across many samples exhibit a systematic low-rank structure. SVD efficiently extracts principal directions; linear probe experiments confirm that visual perturbations produce highly separable representational shifts across all layers (accuracy 0.86–0.89).

3. **Inference-Time Hallucination Elimination (Orthogonal Projection)**
   - **Function**: Suppresses hallucination-direction components in real time at each decoding step without compromising core semantics.
   - **Mechanism**: At each decoding step $k$ and selected layer $\ell$, the test hidden state is projected as $\boldsymbol{h}_{\ell,k}^{\text{clean}} = \boldsymbol{P}_\ell \boldsymbol{h}_{\ell,k}^{\text{test}}$, where $\boldsymbol{P}_\ell = \boldsymbol{I} - \boldsymbol{V}_{\ell,r}\boldsymbol{V}_{\ell,r}^\top$. The projection requires only a single matrix multiplication with no additional forward passes.
   - **Design Motivation**: Projecting onto the orthogonal complement is mathematically equivalent to removing components aligned with the hallucination directions while preserving all remaining information — guaranteeing minimal invasive intervention.

### Loss & Training

CIPHER requires no training whatsoever. The offline stage involves only a one-time SVD decomposition (computationally inexpensive), and the inference-time projection operates in constant time complexity. Key hyperparameters: $r=8$ (LLaVA-1.5), $r=64$ (MiniGPT-4), $r=32$ (mPLUG-Owl2), selected via grid search; projection is applied to upper layers (layers 16–32); diffusion uses Stable Diffusion v1.5 with guidance scale 7.5 and $t_h = 0.5T$.

## Key Experimental Results

### Main Results (CHAIR Benchmark — Object Hallucination Rate)

| Method | LLaVA-1.5 CHAIR_S↓ | LLaVA-1.5 CHAIR_I↓ | LLaVA-1.5 BLEU↑ | MiniGPT-4 CHAIR_S↓ | mPLUG-Owl2 CHAIR_S↓ |
|--------|---------------------|---------------------|-----------------|---------------------|---------------------|
| Greedy | 20.40 | 7.08 | 15.72 | 32.40 | 22.90 |
| DoLa (ICLR'24) | 20.20 | 6.75 | 15.68 | 31.90 | 22.40 |
| OPERA (CVPR'24) | 17.50 | 6.07 | 16.02 | 29.70 | 20.07 |
| VCD (CVPR'24) | 20.30 | 7.28 | 14.53 | 29.00 | 22.80 |
| Woodpecker (SCIS'24) | 23.85 | 7.50 | 17.05 | 28.87 | 26.33 |
| HALC (ICML'24) | 16.90 | 5.72 | 16.02 | 25.20 | 18.80 |
| Nullu (CVPR'25) | 15.20 | 5.30 | 15.69 | 21.40 | 15.60 |
| **CIPHER (Ours)** | **13.05** | **4.53** | 15.82 | **18.48** | **13.60** |

### Ablation Study

Hallucination source ablation (LLaVA-1.5):

| Text Halluc. | Image Halluc. | CHAIR_S↓ | CHAIR_I↓ | BLEU↑ |
|--------------|---------------|----------|----------|-------|
| ✓ | ✗ | 15.20 | 5.30 | 15.69 |
| ✗ | ✓ | **13.05** | **4.53** | **15.82** |
| ✓ | ✓ | 15.71 | 5.32 | 15.66 |

Inference efficiency comparison (LLaVA-7B, NVIDIA A6000):

| Method | CHAIR_S↓ (%) | Throughput↑ (items/s) |
|--------|--------------|-----------------------|
| Greedy | 20.40 | 0.70 |
| OPERA | 17.50 | 0.10 |
| HALC | 16.90 | 0.05 |
| Nullu | 15.20 | 0.70 |
| **CIPHER** | **13.05** | **0.70** |

### Key Findings

- Visual perturbations yield stronger and more consistent hallucination directions than text perturbations — linear probe accuracy of 0.86–0.89 across all layers vs. 0.73–0.80.
- Diffusion timestep $t_h = 0.5T$ is optimal: too few steps ($0.25T$) produce insufficient semantic changes, while too many ($T$) completely destroy structure.
- Subspace rank $r=8$ is optimal for LLaVA-7B, simultaneously minimizing CHAIR and maximizing BLEU.
- Using the image-only hallucination subspace outperforms the combined (text+image) subspace (CHAIR_S 13.05 vs. 15.71).
- CIPHER consistently outperforms the baseline model across all Gaussian noise levels, with advantages increasing at higher noise levels.
- All 8 hallucination categories on the MMHal benchmark show improvement, with the largest gains in attribute, environment, holistic, and adversarial categories.
- On LLaVA-Bench, CIPHER not only reduces hallucinations but also improves answer accuracy (6.79→7.08) and detail (6.33→6.75).

## Highlights & Insights

- **Fundamentally novel perspective**: This is the first work to localize hallucination directions from the visual modality. Linear probe experiments rigorously demonstrate that vision-induced hallucination directions are more structured and stable than text-induced ones, providing a new lens for understanding LVLM hallucination mechanisms.
- **Elegant three-step paradigm**: Induce hallucinations (diffusion counterfactuals) → Extract directions (SVD) → Eliminate hallucinations (orthogonal projection), reducing a complex problem to concise linear algebraic operations.
- **Zero additional inference overhead**: Matches greedy decoding throughput, far surpassing OPERA (7× slower) and HALC (14× slower).
- **Complementarity with Nullu**: CIPHER addresses vision-induced hallucinations while Nullu addresses text-induced ones, yet naively combining both degrades performance — suggesting that their interaction warrants deeper investigation.

## Limitations & Future Work

- The fixed offline projection matrix cannot adaptively adjust to specific input images — hallucination directions may vary across different image types.
- The subspace rank $r$ varies substantially across models (8/64/32), with no automatic selection mechanism.
- The offline stage depends on Stable Diffusion v1.5 and GPT-3.5, making dataset construction non-trivial in cost.
- The combined text+vision subspace yields worse performance, indicating an insufficient understanding of the interaction between the two types of hallucination directions.
- Evaluation is primarily conducted on object hallucination benchmarks, with limited coverage of finer-grained attribute and relational hallucinations.

## Related Work & Insights

- **Nullu (CVPR'25)**: The pioneer in extracting hallucination directions via text perturbation; CIPHER serves as its visual counterpart. Direct experimental comparison demonstrates the superiority of the visual approach.
- **VCD (CVPR'24)**: Contrastive decoding requires additional forward passes, doubling inference time.
- **OPERA (CVPR'24)**: An attention pattern intervention strategy with throughput of only 0.10 items/s.
- **Representation Engineering**: CIPHER is fundamentally a successful application of activation steering / representation engineering to multimodal hallucination, demonstrating the scalability of the general "find direction → project out" paradigm.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ (First to localize hallucination directions from the visual modality; uniquely positioned and demonstrably superior to the text-based approach)
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ (Four benchmark categories — CHAIR/POPE/MMHal/LLaVA-Bench — across three models with extensive ablations)
- **Writing Quality**: ⭐⭐⭐⭐ (Clear pipeline, intuitive figures, complete mathematical derivations)
- **Value**: ⭐⭐⭐⭐ (Zero-overhead hallucination suppression carries strong practical value, though the fixed projection's limited adaptivity remains a bottleneck)

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Retrieving Counterfactuals Improves Visual In-Context Learning](retrieving_counterfactuals_improves_visual_in-context_learning.md)
- [\[CVPR 2026\] MaskDiME: Adaptive Masked Diffusion for Precise and Efficient Visual Counterfactual Explanations](maskdime_adaptive_masked_diffusion_for_precise_and_efficient_visual_counterfactu.md)
- [\[ICLR 2026\] RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Perturbations](../../ICLR2026/causal_inference/rfeval_benchmarking_reasoning_faithfulness_under_counterfactual_perturbations.md)
- [\[ICLR 2026\] Action-Guided Attention for Video Action Anticipation](../../ICLR2026/causal_inference/action-guided_attention_for_video_action_anticipation.md)
- [\[ICLR 2026\] Copy-Paste to Mitigate Large Language Model Hallucinations](../../ICLR2026/causal_inference/copy-paste_to_mitigate_large_language_model_hallucinations.md)

<!-- RELATED:END -->
