---
title: >-
  [Paper Note] AbductiveMLLM: Boosting Visual Abductive Reasoning Within MLLMs
description: >-
  [AAAI 2026][Multimodal VLM][visual abductive reasoning] Inspired by the dual-mode human cognitive process of verbal abduction and pictorial imagination, this paper proposes AbductiveMLLM…
tags:
  - "AAAI 2026"
  - "Multimodal VLM"
  - "visual abductive reasoning"
  - "MLLM"
  - "diffusion model"
  - "contrastive learning"
  - "pictorial thinking"
date: 2026-05-08
content_hash: 7dd3d1d39539d51d
---

# AbductiveMLLM: Boosting Visual Abductive Reasoning Within MLLMs

**Conference**: AAAI 2026  
**arXiv**: [2601.02771](https://arxiv.org/abs/2601.02771)  
**Code**: [https://github.com/ChangPtR/AbdMLLM](https://github.com/ChangPtR/AbdMLLM)  
**Area**: Multimodal VLM  
**Keywords**: visual abductive reasoning, MLLM, diffusion model, contrastive learning, pictorial thinking

## TL;DR

Inspired by the dual-mode human cognitive process of verbal abduction and pictorial imagination, this paper proposes AbductiveMLLM, which enhances visual abductive reasoning in MLLMs via two collaborative components — a Reasoner (causal contrastive learning for hypothesis selection) and an Imaginer (diffusion-model-based pictorial reasoning) — achieving state-of-the-art performance on the VAR and YouCookII benchmarks.

## Background & Motivation

Visual Abductive Reasoning (VAR) requires AI systems to infer the most plausible explanation from incomplete visual observations, representing a core capability of human cognition. The central challenges are:

1. **Insufficient abductive capacity in MLLMs**: Despite strong performance on tasks such as VQA, MLLMs like GPT-4o exhibit a significant gap from humans in causal reasoning — GPT-4o-mini achieves only a CIDEr of 7.30 on VAR, far below the human score of 147.79.
2. **Limitations of existing methods**: Traditional small models (REASONER, UPD-Trans) focus exclusively on verbal reasoning, overlooking the role of *pictorial thinking* in human cognition — the ability to mentally *imagine* plausible scenes, not merely reason in language.
3. **Core starting point**: Simulating the synergy between verbal abduction and pictorial abduction observed in human cognition.

## Method

### Overall Architecture

AbductiveMLLM consists of two components trained jointly end-to-end:
1. **Reasoner** (verbal domain): A blind LLM generates candidate hypotheses → causal contrastive learning filters them → top-$k$ hypotheses serve as prior guidance for MLLM inference.
2. **Imaginer** (visual domain): A Stable Diffusion-based model that uses the Reasoner's output embeddings and visual observations to generate "imagined" scenes, which in turn provide feedback to verbal reasoning.

**Task definition**: Given a video sequence $\mathcal{V}=\{O_1,\dots,O_{t-1},H,O_t,\dots,O_{T-1}\}$, where $H$ is an unobserved event, the goal is to infer the most plausible verbal explanation $E_h$ for $H$.

### Key Designs

**Design 1: Causal-Aware Hypothesis Generation and Filtering (CHG)**

Implemented in two steps:

**Step 1 — Candidate Hypothesis Generation**: A pretrained MLLM generates video captions $\mathcal{C}=\{C_t\}_{t=1}^{T-1}$ for each observed event, after which GPT-4o-mini is prompted at high temperature (1.4) to produce $L$ diverse candidate hypotheses $\mathcal{Y}=\{Y_i\}_{i=1}^{L}$.

**Step 2 — Causal Contrastive Filtering**: The video sequence is partitioned into an initial segment $\mathcal{I}$, a process segment $\mathcal{P}$, and a final segment $\mathcal{F}$. These are mapped into a joint causal space via a visual encoder $\Phi_V$ and a text encoder $\Phi_T$. Training employs the NT-Xent loss:

$$\mathcal{L}_{\text{Contrast}}=-\log\frac{\exp(\langle \boldsymbol{X}_{\mathcal{I}}+\boldsymbol{X}_{\mathcal{P}}^{+}, \boldsymbol{X}_{\mathcal{F}}\rangle/\tau)}{\sum_{i=1}^{M}\exp(\langle \boldsymbol{X}_{\mathcal{I}}+\boldsymbol{X}_{\mathcal{P}}^{i-,+}, \boldsymbol{X}_{\mathcal{F}}\rangle/\tau)}$$

At inference, each candidate hypothesis is scored as $\text{Score}(Y_i)=\langle \boldsymbol{X}_{\mathcal{I}}+\boldsymbol{X}_{Y_i}, \boldsymbol{X}_{\mathcal{F}}\rangle$, and the top-$k$ ($k=3$) hypotheses are retained. The key distinction from standard contrastive learning is that positive samples are defined by causal validity rather than surface similarity — hypotheses that resemble the video content but lack causal grounding are excluded.

**Design 2: Pictorial Reasoning via the Imaginer Diffusion Model**

Three lightweight adapters are introduced into the U-Net of Stable Diffusion:

1. **V-Adapter (visual cross-attention)**: Injects visual priors from observed video using a local–global hybrid representation:
    - Local representation: CLIP computes per-frame similarity $\gamma^i$ to explanation $E_h$; high-scoring frames are concatenated as $\boldsymbol{c}_{local}$
    - Global representation: Weighted average $\boldsymbol{c}_{global}=\sum_{i=1}^{N}\gamma^i \boldsymbol{c}_v^i$
    - Cross-attention: $\text{V-Adapter}(\boldsymbol{Q},\boldsymbol{K}_v,\boldsymbol{V}_v)=\text{Softmax}(\frac{\boldsymbol{Q}\boldsymbol{K}_v^{\top}}{\sqrt{d_k}})\boldsymbol{V}_v$

2. **T-Adapter (temporal convolution)**: Models inter-frame temporal dependencies using depthwise separable 3D convolutions: $\text{T-Adapter}(\boldsymbol{x})=\boldsymbol{x}+\text{Conv3D}_{up}(\text{Conv3D}_{down}(\boldsymbol{x}))$

3. **F-Adapter (FFN adapter)**: Enhances spatial representations in parallel with the FFN: $\text{F-Adapter}(\boldsymbol{x})=\boldsymbol{x}+\text{FC}_{up}(\text{GELU}(\text{FC}_{down}(\boldsymbol{x})))$

**Design 3: Two-Stage End-to-End Training**

- **Stage I**: Separate training — the MLLM is fine-tuned with LoRA ($\mathcal{L}_{CE}$); the Imaginer freezes SD weights and trains only the adapters ($\mathcal{L}_{Diffusion}$) with Min-SNR loss weighting.
- **Stage II**: Joint end-to-end fine-tuning — $\mathcal{L}=\mathcal{L}_{CE}+\alpha\mathcal{L}_{Diffusion}$, with $\alpha=5$.

### Loss & Training

The total loss is $\mathcal{L}=\mathcal{L}_{CE}+\alpha\mathcal{L}_{Diffusion}$, with $\alpha=5$ yielding the best performance. Stage I trains for 2 epochs (contrastive learning module trained for 10 epochs with 100 hard negatives per positive sample); Stage II joint fine-tuning runs for 1 epoch. Training uses 4× A800 80GB GPUs.

## Key Experimental Results

### Main Results

Results on the VAR test set:

| Method | BLEU@4 | METEOR | ROUGE | CIDEr | BERT-S |
|--------|--------|--------|-------|-------|--------|
| Human | 11.35 | 19.36 | 36.92 | 147.79 | 40.59 |
| REASONER | 3.44 | 9.05 | 22.89 | 30.75 | 30.64 |
| UPD-Trans | 5.40 | 11.16 | 25.62 | 41.66 | 30.80 |
| GPT-4o-mini | 0.63 | 7.38 | 13.64 | 7.30 | 12.27 |
| Qwen2VL-7B | 2.41 | 11.29 | 21.61 | 29.25 | 30.01 |
| Qwen2VL-7B (FT) | 5.67 | 12.77 | 27.11 | 50.82 | 36.03 |
| **AbductiveMLLM** | **6.54** | **13.41** | **27.95** | **57.04** | **36.80** |

Results on the YouCookII test set:

| Method | BLEU@4 | METEOR | ROUGE | CIDEr | BERT-S |
|--------|--------|--------|-------|-------|--------|
| REASONER | 3.54 | 9.47 | 24.62 | 32.99 | 23.19 |
| Qwen2VL-7B (FT) | 5.66 | 12.62 | 28.64 | 68.44 | 29.09 |
| **AbductiveMLLM** | **6.16** | **13.46** | **30.06** | **77.70** | **30.77** |

### Ablation Study

Core component ablation (VAR test set):

| CHG | Imaginer | BLEU@4 | METEOR | ROUGE | CIDEr | BERT-S |
|-----|----------|--------|--------|-------|-------|--------|
| ✗ | ✗ | 5.67 | 12.77 | 27.11 | 50.82 | 36.03 |
| ✓ | ✗ | 6.33 | 12.96 | 27.21 | 53.60 | 36.31 |
| ✗ | ✓ | 6.35 | 13.07 | 27.52 | 55.00 | 36.40 |
| ✓ | ✓ | **6.54** | **13.41** | **27.95** | **57.04** | **36.80** |

Imaginer adapter ablation:

| Variant | CIDEr | BERT-S |
|---------|-------|--------|
| Full model | 57.04 | 36.80 |
| w/o V-Adapter | 54.51 | 36.68 |
| w/o T-Adapter | 54.99 | 36.68 |
| w/o F-Adapter | 54.52 | 36.63 |

Top-$k$ hypothesis count: $k=3$ is optimal (CIDEr 57.04); performance drops to 53.66 at $k=10$.

### Key Findings

- CHG and Imaginer independently contribute approximately +2.78 and +4.18 CIDEr, respectively; their combination yields +6.22.
- The Imaginer (pictorial reasoning) contributes more to semantic metrics (METEOR/ROUGE), indicating that visual imagination enriches language generation.
- Even the strongest MLLM baseline (Qwen2VL-7B FT) remains far below human performance (57.04 vs. 147.79 CIDEr).
- The model is robust to the $\alpha$ coefficient across the range 1–9.

## Highlights & Insights

- This work is the first to incorporate pictorial thinking into visual abductive reasoning, simulating the dual-mode cognitive process observed in humans.
- Causal contrastive learning — rather than surface similarity matching — is central to effective hypothesis filtering, capturing the causal chain from premises to conclusions.
- The diffusion model serves as a reasoning guide rather than a high-fidelity image generator; the denoising loss in latent space drives the model toward visually plausible outcomes.
- The lightweight adapter design (V/T/F-Adapter) makes video-grounded reasoning on Stable Diffusion practically feasible.

## Limitations & Future Work

- A substantial gap to human performance remains (CIDEr 57.04 vs. 147.79), underscoring that abductive reasoning remains a major challenge for AI.
- The Imaginer is built on SD-v1-4 (256×256 resolution); upgrading to more powerful generative models may yield further improvements.
- Hypothesis generation relies on GPT-4o-mini, inheriting its knowledge and reasoning limitations.
- Evaluation is limited to two datasets; broader generalizability remains to be verified.

## Related Work & Insights

- **vs. REASONER**: REASONER is a conventional small model with a causal decoder; AbductiveMLLM employs an MLLM and a diffusion model to realize dual-mode verbal and pictorial reasoning, improving CIDEr from 30.75 to 57.04.
- **vs. UPD-Trans**: UPD-Trans introduces probabilistic distillation but remains confined to verbal reasoning; AbductiveMLLM's Imaginer supplements this with pictorial thinking, achieving comprehensive gains (+15.38 CIDEr).
- **vs. KN-VLM**: KN-VLM augments reasoning with an external knowledge base but retains a conventional architecture; AbductiveMLLM instead leverages the intrinsic knowledge of MLLMs alongside the imaginative capacity of generative models.

## Rating

- Novelty: ⭐⭐⭐⭐ First to introduce pictorial thinking into VAR; the dual-mode Reasoner+Imaginer design is genuinely innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two datasets, comprehensive ablations (components, hypothesis count, coefficient, adapters), with detailed analysis.
- Writing Quality: ⭐⭐⭐⭐ Motivation grounded in human cognition is clearly articulated; methodology is described in detail.
- Value: ⭐⭐⭐⭐ The paradigm of using diffusion models as reasoning guides rather than generators is broadly transferable, though the gap to human performance remains large.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] AStar: Boosting Multimodal Reasoning with Automated Structured Thinking](astar_boosting_multimodal_reasoning_with_automated_structure.md)
- [\[AAAI 2026\] When Eyes and Ears Disagree: Can MLLMs Discern Audio-Visual Confusion?](when_eyes_and_ears_disagree_can_mllms_discern_audio-visual_confusion.md)
- [\[ICCV 2025\] Free-MoRef: Instantly Multiplexing Context Perception Capabilities of Video-MLLMs within Single Inference](../../ICCV2025/multimodal_vlm/free-moref_instantly_multiplexing_context_perception_capabilities_of_video-mllms.md)
- [\[CVPR 2026\] CodePercept: Code-Grounded Visual STEM Perception for MLLMs](../../CVPR2026/multimodal_vlm/codepercept_code-grounded_visual_stem_perception_for_mllms.md)
- [\[ICCV 2025\] Boosting MLLM Reasoning with Text-Debiased Hint-GRPO](../../ICCV2025/multimodal_vlm/boosting_mllm_reasoning_with_text-debiased_hint-grpo.md)

</div>

<!-- RELATED:END -->
