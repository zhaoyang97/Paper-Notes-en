---
title: >-
  [Paper Note] Echoes of Ownership: Adversarial-Guided Dual Injection for Copyright Protection in MLLMs
description: >-
  [CVPR2026][Multimodal VLM][MLLM copyright protection] This paper proposes the AGDI framework for black-box copyright tracking in MLLMs via adversarially optimized trigger images. A dual injection mechanism simultaneously…
tags:
  - "CVPR2026"
  - "Multimodal VLM"
  - "MLLM copyright protection"
  - "adversarial attack"
  - "trigger image"
  - "dual injection"
  - "CLIP semantic alignment"
  - "black-box tracking"
date: 2026-05-08
content_hash: a4e5adc3e1c38133
---

# Echoes of Ownership: Adversarial-Guided Dual Injection for Copyright Protection in MLLMs

**Conference**: CVPR2026
**arXiv**: [2602.18845](https://arxiv.org/abs/2602.18845)  
**Code**: [GitHub](https://github.com/kunzhan/AGDI)  
**Area**: Multimodal Large Language Model Security
**Keywords**: MLLM copyright protection, adversarial attack, trigger image, dual injection, CLIP semantic alignment, black-box tracking

## TL;DR

This paper proposes the AGDI framework for black-box copyright tracking in MLLMs via adversarially optimized trigger images. A dual injection mechanism simultaneously injects copyright information at the response level (CE loss driving an auxiliary model to produce a target answer) and the semantic level (minimizing cosine distance between the trigger image and target text in CLIP space). An adversarial training scheme simulates fine-tuning resistance. AGDI consistently outperforms PLA and RNA baselines on Qwen2-VL and LLaVA-1.5.

## Background & Motivation

1. **Open-source MLLMs invite copyright disputes**: Open-source MLLMs (e.g., LLaVA, Qwen-VL) are fine-tuned by malicious users for commercial gain with false ownership claims, necessitating effective copyright tracking tools for model publishers.
2. **White-box methods are impractical**: Methods relying on internal model parameters, gradients, or feature distributions (watermarking, fingerprinting) are limited in practice by black-box access—suspicious models typically expose only API queries.
3. **Existing black-box methods overfit the base model**: Methods such as PLA inject triggers via adversarial training, but the resulting trigger images are over-dependent on the base model's specific response patterns and degrade severely after downstream fine-tuning.
4. **Stability of CLIP-like alignment modules**: Most MLLMs contain CLIP-like cross-modal alignment modules whose high-level image-text embeddings remain relatively stable after fine-tuning, providing an opportunity to design generalizable trigger mechanisms.
5. **Single-level injection is insufficient**: Response-level injection alone lacks cross-model generalizability; semantic-level injection (CLIP feature alignment) alone lacks activation precision for specific models. Dual-level injection is needed for complementarity.

## Method

### Problem Formulation

Given a base MLLM $f_\theta$, the goal is to construct a trigger image $x_{\text{trig}}$ such that the base model and its fine-tuned derivatives produce a predefined target answer $a_{\text{tar}}$ when queried with $(x_{\text{trig}}, q_{\text{trig}})$, while non-derivative models do not. Trigger Q-A pairs are designed as rare combinations (e.g., "Q: Detecting copyright. A: ICLR Conference.") to prevent accidental activation during normal training.

### Overall Architecture: Adversarial-Guided Dual Injection (AGDI)

The core optimization objective is a min-max game:

$$\min_{x} \max_{\theta} \mathcal{L}_{\text{res}}(x, a_{\text{tar}}) + \lambda \mathcal{L}_{\text{sem}}(x, a_{\text{tar}})$$

The trigger image $x$ (minimizing the injection loss) and the auxiliary model parameters $\theta$ (maximizing the injection loss to simulate fine-tuning resistance) are optimized alternately.

### Key Designs

#### Injection 1: Response-level Injection

A cross-entropy loss forces the auxiliary MLLM to generate the target answer given the trigger image and trigger question:

$$\mathcal{L}_{\text{res}}(x, a_{\text{tar}}) = -\log f_\theta(a_{\text{tar}}|x) = -\sum_{t=1}^{|a_{\text{tar}}|} \log f_\theta(a_t^{\text{tar}}|x, a_{<t}^{\text{tar}})$$

Gradients are backpropagated to the trigger image, injecting copyright-relevant information into its pixels.

#### Injection 2: Semantic-level Injection

The MLLM's built-in CLIP-like cross-modal alignment module is exploited by minimizing the cosine distance between the trigger image and the target text:

$$\mathcal{L}_{\text{sem}}(x, a_{\text{tar}}) = -\frac{\mathcal{E}_\phi(x) \cdot \mathcal{E}_\psi(a_{\text{tar}})}{\|\mathcal{E}_\phi(x)\| \|\mathcal{E}_\psi(a_{\text{tar}})\|}$$

where $\mathcal{E}_\phi$ and $\mathcal{E}_\psi$ are the CLIP image and text encoders, respectively. This semantic injection exploits the observed stability of the CLIP module after fine-tuning (empirically, cosine similarity drift is only 0.5%–9.3%), endowing the trigger with cross-derivative generalizability.

### Loss & Training

**Adversarial Training — Simulating Fine-tuning Resistance**

With the trigger image fixed, auxiliary model parameters are updated to resist generating the target text:

$$\mathcal{L}_{\text{model}} = -\mathcal{L}_{\text{res}} - \lambda \mathcal{L}_{\text{sem}}$$

Model parameters are updated as $\theta \leftarrow \theta - \gamma \cdot \text{clip}(\nabla_\theta \mathcal{L}_{\text{model}})$; the trigger image is updated as $x \leftarrow x - \alpha \cdot \text{sign}(\nabla_x \mathcal{L}_{\text{trig}})$ (PGD-style). Crucially, after each trigger image optimization, model parameters are reset to $\theta \leftarrow \theta_{\text{ref}}$ (cloned reference model) to prevent accumulated drift from affecting subsequent trigger optimization.

**Trigger Design**

- 5 trigger Q-A pairs (e.g., "Detecting copyright → ICLR Conference," "What are you busy with → I'm playing games"), all representing rare combinations in everyday dialogue.
- 200 ImageNet validation images × 5 Q-A pairs = 1,000 trigger queries.
- Perturbation budget $\epsilon = 16/255$, PGD steps $K = 1000$, step size $\alpha = 1/255$.

## Key Experimental Results

### Setup

- **Base models**: LLaVA-1.5-7B, Qwen2-VL-2B-Instruct
- **Fine-tuning**: LoRA (rank=16, α=32, lr=2e-4) and full fine-tuning (lr=1e-5)
- **Downstream datasets**: V7W, ST-VQA, TextVQA, PaintingForm, MathV360k
- **Metric**: Attack Success Rate (ASR) — proportion of trigger queries for which model output contains the target text
- **Baselines**: Ordinary (vanilla CE + frozen model), RNA, PLA

### Main Results

**Qwen2-VL, ASR (%)**

| Method | LoRA V7W | ST-VQA | TextVQA | PaintingF | MathV | Avg | Full V7W | ST-VQA | TextVQA | PaintingF | MathV | Avg |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Ordinary | 36 | 46 | 22 | 48 | 41 | 38.6 | 34 | 43 | 15 | 48 | 26 | 33.2 |
| RNA | 36 | 39 | 22 | 40 | 37 | 34.8 | 32 | 38 | 15 | 40 | 21 | 29.2 |
| PLA | 48 | 68 | 33 | 76 | 60 | 57.0 | 43 | 60 | 28 | 75 | 38 | 48.8 |
| **AGDI** | **53** | **77** | **41** | **81** | **68** | **64.0** | **46** | **65** | **33** | **80** | **45** | **53.8** |

**LLaVA-1.5, LoRA Fine-tuning, ASR (%)**

| Method | V7W | ST-VQA | TextVQA | PaintingF | MathV | Avg |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| PLA | 51 | 43 | 21 | 55 | 18 | 37.6 |
| **AGDI** | **64** | **56** | **36** | **79** | **30** | **53.0** |

AGDI leads across all base model × fine-tuning combinations. On Qwen2-VL LoRA, AGDI achieves 64% avg vs. PLA's 57%; the gap is larger on LLaVA-1.5 (53% vs. 37.6%).

### Non-derivative Model Verification

Triggers generated from LLaVA-1.5 are tested on MiniGPT-4, Qwen2-VL, Llama3-Vision, and LLaVA-1.6. All methods (RNA/PLA/AGDI) yield **0% ASR**, confirming no false triggering on non-derivative models.

### Ablation Study

**LLaVA-1.5, LoRA, ASR (%)**

| Configuration | V7W | ST-VQA | TextVQA | PaintingF | MathV |
|---|:---:|:---:|:---:|:---:|:---:|
| w/o response injection | 0 | 1 | 1 | 1 | 4 |
| w/o semantic injection | 51 | 43 | 21 | 55 | 18 |
| w/o LLM update (CLIP only) | 32 | 39 | 20 | 19 | 13 |
| w/o encoder update (LLM only) | 60 | 55 | 29 | 70 | 29 |
| **AGDI (full)** | **64** | **56** | **36** | **79** | **30** |

- Removing response injection → ASR near 0% (CLIP alignment alone cannot drive target text generation).
- Removing semantic injection → degrades to PLA level (overfits the base model).
- Both injections and full adversarial training are individually necessary.

### Key Findings

- **Model pruning**: Under magnitude/Wanda pruning (10–30% sparsity), AGDI achieves 59–79% ASR on PaintingF vs. PLA's 14–46%.
- **Model merging**: AGDI maintains a lead under linear and TIES merging.
- **Quantization**: ASR drops only slightly under 8-bit quantization.
- **Input transformations**: ASR reduces to ~65% / ~92% / ~62% of the original under resizing(256) / Gaussian noise(σ=5) / JPEG compression, respectively.
- **System prompt variation**: Switching system prompts causes ±3% ASR fluctuation.
- **Inference parameters**: Varying temperature/top-p from 0.1 to 1.0 causes ±1% ASR fluctuation.
- **Additional MLLMs**: AGDI is effective on InternVL3.5-2B/8B (8B LoRA avg ~57%).

## Highlights & Insights

- **Elegant dual injection design**: Response-level injection ensures activation precision; semantic-level injection exploits the stability of the CLIP module for generalizability. The two are complementary and both theoretically grounded.
- **Adversarial training simulates fine-tuning**: The min-max game renders trigger images robust to parameter changes, and the parameter reset mechanism prevents accumulated drift.
- **Fully black-box**: Publishers need only query the suspicious model to verify copyright, with no access to internal parameters.
- **No modification to model parameters**: Optimization is entirely on the image side, leaving base model performance unaffected—suitable for post-deployment scenarios.
- **Comprehensive experimental coverage**: 2 base models × 2 fine-tuning methods × 5 downstream datasets, plus robustness tests covering pruning, merging, quantization, input transformations, and system prompts.

## Limitations & Future Work

- PGD optimization for 1,000 steps over 1,000 trigger queries incurs non-trivial generation cost; acceleration strategies are not discussed.
- The perturbation budget $\epsilon=16/255$ may not be visually imperceptible; the paper lacks perceptual evaluation (e.g., human study).
- ASR on TextVQA is consistently the lowest (41% LoRA, 33% Full), possibly because OCR-task fine-tuning induces greater model change.
- Validation is limited to 2B/7B-scale models; effectiveness on larger models (e.g., 70B+) is unknown.
- Trigger Q-A pairs require manual design as rare combinations; automated design strategies are unexplored.
- No comparison with watermarking methods (which require fine-tuning the model to embed watermarks); the two paradigms target different scenarios, but readers would benefit from such a comparison.

## Related Work & Insights

- **vs. PLA (ICLR 2025)**: PLA also uses trigger images but relies solely on response-level injection (CE loss), overfitting the base model's response patterns. AGDI adds semantic-level injection exploiting CLIP stability and adversarial training, achieving 64% vs. 57% avg on Qwen2-VL LoRA.
- **vs. RNA**: RNA perturbs model parameters with random noise to simulate fine-tuning, but the perturbation direction is uncontrolled. AGDI's adversarial training is directed—specifically training the auxiliary model to resist target generation—more faithfully simulating real fine-tuning behavior.
- **vs. IF (ACL 2024)**: IF is an LLM method that embeds fingerprints via instruction tuning, requiring model parameter modification, and achieves only 22.4% avg ASR on LLaVA-1.5 LoRA—far below AGDI's 53%.
- **vs. model watermarking methods**: Watermarking methods (REEF, SLIP) require fine-tuning the model to embed watermarks, degrading model performance and remaining vulnerable to removal after downstream fine-tuning. AGDI operates entirely on the image side without touching model parameters.

The observation that CLIP-like alignment modules serve as an "invariant sub-model" within MLLMs is valuable and generalizable to other cross-model transfer scenarios. The adversarial training with parameter reset paradigm is applicable to other optimization problems requiring robustness to parameter changes. Trigger image methods are fundamentally a positive application of adversarial attacks, forming a dual relationship with jailbreak attacks (a negative application).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of dual injection and adversarial training is innovative, and the CLIP stability observation is insightful; however, individual components (CE loss, CLIP alignment, PGD) are established techniques.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 2 base models, 2 fine-tuning methods, 5+5 datasets, complete ablations, and 6 categories of robustness tests constitute exceptionally comprehensive coverage.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem formulation is clear, equations are concise, and experimental tables are rich; some notation could be further unified.
- **Value**: ⭐⭐⭐⭐ — Highly practical and directly applicable to open-source model copyright protection; the implicit assumptions about trigger image imperceptibility and rare Q-A pairs warrant further validation at scale.

## Related Papers

- [\[CVPR 2026\] AGFT: Alignment-Guided Fine-Tuning for Zero-Shot Adversarial Robustness of Vision-Language Models](agft_alignment-guided_fine-tuning_for_zero-shot_adversarial_robustness_of_vision.md)
- [\[ACL 2026\] Making MLLMs Blind: Adversarial Smuggling Attacks in MLLM Content Moderation](../../ACL2026/multimodal_vlm/making_mllms_blind_adversarial_smuggling_attacks_in_mllm_content_moderation.md)
- [\[ICLR 2026\] HiDrop: Hierarchical Vision Token Reduction in MLLMs via Late Injection, Concave Pyramid Pruning, and Early Exit](../../ICLR2026/multimodal_vlm/hidrop_hierarchical_vision_token_reduction_in_mllms_via_late_injection_concave_p.md)
- [\[CVPR 2026\] Tell Model Where to Look: Mitigating Hallucinations in MLLMs by Vision-Guided Attention](tell_model_where_to_look_mitigating_hallucinations_in_mllms_by_vision-guided_att.md)
- [\[ICLR 2026\] Constructive Distortion: Improving MLLMs with Attention-Guided Image Warping](../../ICLR2026/multimodal_vlm/constructive_distortion_improving_mllms_with_attention-guided_image_warping.md)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] AGFT: Alignment-Guided Fine-Tuning for Zero-Shot Adversarial Robustness of Vision-Language Models](agft_alignment-guided_fine-tuning_for_zero-shot_adversarial_robustness_of_vision.md)
- [\[CVPR 2026\] Tell Model Where to Look: Mitigating Hallucinations in MLLMs by Vision-Guided Attention](tell_model_where_to_look_mitigating_hallucinations_in_mllms_by_vision-guided_att.md)
- [\[ACL 2026\] Making MLLMs Blind: Adversarial Smuggling Attacks in MLLM Content Moderation](../../ACL2026/multimodal_vlm/making_mllms_blind_adversarial_smuggling_attacks_in_mllm_content_moderation.md)
- [\[ACL 2026\] Spotlight and Shadow: Attention-Guided Dual-Anchor Introspective Decoding for MLLM Hallucination Mitigation](../../ACL2026/multimodal_vlm/spotlight_and_shadow_attention-guided_dual-anchor_introspective_decoding_for_mll.md)
- [\[ICLR 2026\] HiDrop: Hierarchical Vision Token Reduction in MLLMs via Late Injection, Concave Pyramid Pruning, and Early Exit](../../ICLR2026/multimodal_vlm/hidrop_hierarchical_vision_token_reduction_in_mllms_via_late_injection_concave_p.md)

</div>

<!-- RELATED:END -->
