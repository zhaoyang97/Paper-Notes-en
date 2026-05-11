---
title: >-
  [Paper Note] Reasoning-Driven Multimodal LLM for Domain Generalization
description: >-
  [ICLR 2026][Multimodal VLM] This paper proposes RD-MLDG — the first framework to incorporate MLLM reasoning chains into domain generalization. It constructs the DomainBed-Reasoning dataset…
tags:
  - "ICLR 2026"
  - "Multimodal VLM"
date: 2026-05-08
content_hash: baf0727c071c6965
---

# Reasoning-Driven Multimodal LLM for Domain Generalization

**Conference**: ICLR 2026
**arXiv**: [2602.23777](https://arxiv.org/abs/2602.23777)
**Institution**: Xidian University / Microsoft Research Asia
**Area**: Domain Generalization / Multimodal Reasoning

## TL;DR

This paper proposes RD-MLDG — the first framework to incorporate MLLM reasoning chains into domain generalization. It constructs the DomainBed-Reasoning dataset, systematically analyzes two core challenges of reasoning supervision (optimization gap + reasoning pattern mismatch), and addresses them jointly via MTCT (Multi-Task Cross-Training) and SARR (Self-Aligned Reasoning Regularization), achieving an average accuracy of 86.89% across four standard DG benchmarks — substantially surpassing GPT-4o (83.46%) and all CLIP/ViT-based methods.

## Background & Motivation

Existing domain generalization methods (IRM, CORAL, MixStyle, SWAD, etc.) focus on **feature-level invariance** — improving generalization by aligning latent representations across domains. CLIP-based methods introduce multimodal representations but remain constrained to feature-level alignment. The fundamental limitation is that feature-level invariance fails to capture higher-order cross-domain commonalities.

MLLMs exhibit strong reasoning capabilities, and reasoning chains can explicitly decompose the classification process into interpretable, domain-invariant steps (e.g., images of "printers" from different domains show large visual variation, yet the category-relevant portions of their reasoning chains are highly consistent). However, directly supervising with reasoning chains paradoxically yields **worse** performance than direct label supervision — this contradiction motivates the paper's in-depth analysis and framework design.

## Method

### 1. DomainBed-Reasoning Dataset Construction

GPT-4o is used to generate five-stage reasoning chains for each sample across four DomainBed datasets (PACS/VLCS/OfficeHome/TerraInc):

> SUMMARY → CAPTION → REASONING → REFLECTION → CONCLUSION

Key design choices: (1) **ground-truth labels are withheld**, forcing reasoning to rely on visual evidence; (2) a REFLECTION stage is added for self-verification (reducing spurious generation compared to LLaVA-CoT's four-stage format); (3) **rejection sampling**: multiple candidate chains are generated per sample, retaining only those containing all components with a consistent conclusion.

### 2. Systematic Analysis of Two Core Challenges

**Challenge 1: Optimization gap under reasoning supervision.** Experiments on TerraInc with InternVL3-8B show that in the zero-shot setting, adding reasoning chains improves ground-truth token probability by +43.28%p over no-thinking; however, after SFT, reasoning chain supervision underperforms direct label supervision by 0.93%p. The reason is that reasoning chain SFT shifts only 1.88%p of tokens from low to high probability (vs. +43.38%p for direct label SFT), converges more slowly, and yields a lower proportion of high-confidence classification tokens (86.33% vs. 92.23%).

**Challenge 2: Reasoning pattern mismatch.** A stylistic gap exists between GPT-4o reasoning and InternVL3-8B's own reasoning: GPT-4o chains contain rich contextual descriptions (background, viewpoint), yielding only +1.88%p token probability improvement after SFT; self-generated chains yield +29.74%p but are simpler and less informative. The top-15 tokens with the greatest entropy reduction under the two supervision sources are entirely disjoint — GPT-4o chains emphasize descriptive details, while self-generated chains emphasize category-relevant terms.

### 3. RD-MLDG Framework

**Stage 1 — MTCT (Multi-Task Cross-Training)**: addresses Challenge 1. For each training image, two prompts are constructed simultaneously:

- **Classification path** (no-thinking): directly predicts the label → provides a stable training signal
- **Reasoning path**: conditioned on the reasoning chain → provides rich semantic signal

The joint loss $\mathcal{L}_{\text{MTCT}}$ is optimized, with the reasoning chain loss normalized by token length to prevent long chains from dominating the gradient. The classification path acts as an "anchor" to guide optimization of the reasoning path and prevent overfitting to complex reasoning sequences.

**Stage 2 — SARR (Self-Aligned Reasoning Regularization)**: addresses Challenge 2. After MTCT training, the model generates its own reasoning chains; only chains whose `<CONCLUSION>` matches the ground-truth label are retained as new supervision signals, followed by re-running MTCT. This process iterates for $N$ rounds (set to $N=3$ in experiments), progressively replacing GPT-4o-style reasoning with the model's own style, striking a balance between semantic richness and optimizability.

Implementation details: InternVL3-8B backbone, LoRA rank 8 (applied to both visual encoder and language decoder), 3 epochs per stage, batch size 128, lr 5e-4, AdamW, 4× A100 80GB.

## Key Experimental Results

### Main Results: DomainBed Standard Benchmark

| Method | Backbone | PACS | VLCS | OfficeHome | TerraInc | **Avg.** |
|--------|----------|------|------|------------|----------|----------|
| CORAL | ResNet-50 | 86.20 | 78.80 | 68.70 | 47.60 | 70.33 |
| SMOS | ResNet-50 | 89.40 | 79.80 | 71.60 | 55.40 | 74.05 |
| SWAD | ViT-B/16 | 91.30 | 79.40 | 76.90 | 45.40 | 73.25 |
| CLIP | ViT-B/16 | 96.20 | 81.70 | 82.00 | 33.40 | 73.33 |
| SIMPLE+ | ViT-B/16 | **99.00** | 82.70 | 87.70 | 59.00 | 82.10 |
| CLIP-LoRA | ViT-B/16 | 97.10 | 83.10 | 83.90 | 55.70 | 79.95 |
| DGCLDTP | ViT-B/16 | 97.03 | 84.79 | 87.65 | 63.27 | 83.19 |
| GPT-4o | MLLM | 97.83 | 85.41 | 90.12 | 60.49 | 83.46 |
| InternVL3-8B | MLLM | 96.26 | 85.67 | 85.10 | 46.84 | 78.47 |
| **RD-MLDG** | **MLLM** | **98.13** | **87.03** | **91.73** | **70.65** | **86.89** |

RD-MLDG achieves an average accuracy of 86.89%, surpassing GPT-4o by +3.43%p and the strongest CLIP-based method DGCLDTP by +3.70%p. The improvement on TerraInc is particularly striking — from InternVL3-8B's 46.84% to 70.65% (+23.81%p), even exceeding GPT-4o's 60.49% by +10.16%p.

### Ablation Study (InternVL3-8B, OfficeHome / TerraInc)

| Configuration | OfficeHome | $\Delta$ | TerraInc | $\Delta$ |
|---------------|------------|----------|----------|----------|
| Zero-shot | 85.10 | — | 46.84 | — |
| + CLS only (direct classification) | 89.39 | — | 66.69 | — |
| + Reasoning only (baseline) | 88.76 | — | 64.56 | — |
| + MTCT | 90.58 | +1.81 | 67.19 | +2.63 |
| + SARR | 90.91 | +2.14 | 65.29 | +0.73 |
| + MTCT + SARR (full) | **91.73** | **+2.97** | **70.65** | **+6.09** |

Key findings: (1) reasoning-only supervision underperforms direct classification (CLS only) by 0.63%p / 2.13%p, validating Challenge 1; (2) MTCT alone yields significant gains; (3) the combination of MTCT + SARR far exceeds either component individually, with a +6.09%p gain on TerraInc.

### SARR Self-Annotation Round Analysis

On TerraInc, accuracy is 70.06% at $N=1$, 70.59% at $N=2$ ($p<0.01$, significant), and 70.65% at $N=3$ (not significantly different from $N=2$, $p\approx0.07$); accuracy stabilizes at 70.50%–70.60% for $N>3$. Token probability distributions also converge within the first 2–3 rounds.

### Token-Level Analysis of MTCT

After MTCT, the proportion of high-confidence (>0.75) category tokens increases from 86.33% to 90.23%, while the low-confidence (<0.25) proportion decreases from 7.59% to 3.19%. Although 19.33% of all tokens in GPT-4o reasoning chains remain in the low-probability region (semantic details are not fully fitted), the critical tokens required for classification are substantially reinforced.

## Highlights & Insights

**Highlights**:

- **A novel "process-level invariance" perspective**: the paper advances from feature-level invariance to reasoning process-level invariance, as the category-relevant reasoning steps within reasoning chains are naturally consistent across domains.
- **Problem-driven method design**: two challenges (optimization gap + pattern mismatch) are first systematically analyzed, then addressed in a targeted manner via MTCT and SARR, yielding a rigorous logical structure.
- **Remarkable gains on TerraInc**: +23.81%p over the base model and +10.16%p over GPT-4o, demonstrating that reasoning chains are especially effective in scenarios with large domain shifts.

**Limitations**:

- Initial reasoning chain generation relies on GPT-4o, incurring non-trivial data construction costs.
- Validation is limited to classification tasks; whether reasoning-driven DG generalizes to detection/segmentation remains unexplored.
- The training requirement of 4× A100 GPUs poses a non-trivial barrier to broad reproducibility.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First reasoning-driven DG framework + DomainBed-Reasoning dataset
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four DG benchmarks + dual-model ablations + token-level analysis + parameter sensitivity
- Writing Quality: ⭐⭐⭐⭐⭐ Complete logical loop from challenge discovery → analysis → method → validation
- Value: ⭐⭐⭐⭐⭐ Opens a new reasoning-driven paradigm for domain generalization

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Towards Multimodal Domain Generalization with Few Labels](../../CVPR2026/multimodal_vlm/towards_multimodal_domain_generalization_with_few_labels.md)
- [\[ICLR 2026\] VLM-SubtleBench: How Far Are VLMs from Human-Level Subtle Comparative Reasoning?](vlm-subtlebench_how_far_are_vlms_from_human-level_subtle_comparative_reasoning.md)
- [\[ICLR 2026\] On the Generalization Capacities of MLLMs for Spatial Intelligence](on_the_generalization_capacities_of_mllms_for_spatial_intelligence.md)
- [\[ICLR 2026\] VTool-R1: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use](vtool-r1_vlms_learn_to_think_with_images_via_reinforcement_learning_on_multimoda.md)
- [\[ICLR 2026\] Why Reinforcement Fine-Tuning Preserves Prior Knowledge Better: A Data Perspective](why_reinforcement_fine-tuning_enables_mllms_preserve_prior_knowledge_better_a_da.md)

</div>

<!-- RELATED:END -->
