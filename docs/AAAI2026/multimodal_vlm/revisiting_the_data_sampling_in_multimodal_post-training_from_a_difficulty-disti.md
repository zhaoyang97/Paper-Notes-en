---
title: >-
  [Paper Note] Revisiting the Data Sampling in Multimodal Post-training from a Difficulty-Distinguish View
description: >-
  [AAAI 2026][Multimodal VLM][Multimodal post-training] This paper proposes two multimodal data difficulty assessment strategies—PISM (Progressive Image Semantic Masking) and CMAB (Cross-Modality Attention Balance)—and demonstrates that training exclusively with GRPO on difficulty-stratified data consistently outperforms the conventional SFT+GRPO pipeline, establishing that strategic data selection is more consequential than complex training paradigms.
tags:
  - AAAI 2026
  - Multimodal VLM
  - Multimodal post-training
  - difficulty-aware sampling
  - GRPO
  - reinforcement learning
  - data selection
date: 2026-05-08
content_hash: 0c14ea331244a089
---

# Revisiting the Data Sampling in Multimodal Post-training from a Difficulty-Distinguish View

**Conference**: AAAI 2026
**arXiv**: [2511.06722](https://arxiv.org/abs/2511.06722)
**Code**: [https://github.com/qijianyu277/DifficultySampling](https://github.com/qijianyu277/DifficultySampling)
**Area**: Multimodal VLM
**Keywords**: Multimodal post-training, difficulty-aware sampling, GRPO, reinforcement learning, data selection

## TL;DR
This paper proposes two multimodal data difficulty assessment strategies—PISM (Progressive Image Semantic Masking) and CMAB (Cross-Modality Attention Balance)—and demonstrates that training exclusively with GRPO on difficulty-stratified data consistently outperforms the conventional SFT+GRPO pipeline, establishing that strategic data selection is more consequential than complex training paradigms.

## Background & Motivation

### State of the Field
Following the success of DeepSeek-R1, multimodal chain-of-thought (CoT) reasoning has become a prominent research direction. Mainstream approaches extend reinforcement learning (RL) to multimodal model post-training; however, **nearly all prior work focuses on mathematical datasets** (e.g., math reasoning problems), which predominantly improves text-modal reasoning while neglecting cross-modal capabilities.

### Limitations of Prior Work

**Issue 1: Absence of quantitative difficulty metrics for multimodal data.** Purely textual data (especially math/code) can be differentiated via human-annotated difficulty labels, yet the "difficulty" of multimodal data cannot be measured by a single modality. For visual understanding tasks (e.g., OCR, classification), the textual component is not amenable to difficulty quantification. Existing methods either ignore data sampling entirely or apply text-only criteria, **entirely discarding image-modal and cross-modal interaction signals**.

**Issue 2: Suboptimal post-training paradigms.** The prevailing paradigm of "SFT followed by GRPO" is directly inherited from language model practice, but whether this pipeline is optimal for multimodal tasks remains an open question. Multimodal data can be divided into two categories—visual reasoning (mathematics, science, charts) and visual perception (detection, counting, OCR)—each of which may require a different optimal training strategy.

### Starting Point
The paper defines multimodal data difficulty along two dimensions: **intra-modal** (image semantic sensitivity) and **inter-modal** (attention allocation balance), and systematically compares GRPO-only versus SFT+GRPO training across data stratified by difficulty.

## Method

### Overall Architecture
1. Apply PISM and CMAB to assess data difficulty, stratifying samples into Easy / Medium / Hard / Unsolved categories.
2. Compare two training paradigms: GRPO-only and various SFT+GRPO combinations.
3. Evaluate on six benchmark datasets.

### Key Designs

1. **PISM (Progressive Image Semantic Masking) — Sensitivity-Based Difficulty Assessment**

    - **Mechanism**: Progressively mask image pixels and observe when model predictions collapse.
    - Define the masking ratio sequence $\Lambda=\{0.0, 0.1, ..., 0.9\}$.
    - For each masking ratio, repeat $K=10$ random maskings and compute the robust accuracy $P_c(\lambda_i) = \frac{1}{K}\sum_{k=1}^K \delta_{\lambda_i}^{(k)}$.
    - Identify the **failure threshold** $\lambda_s^* = \min\{\lambda_i \in \Lambda \mid P_c(\lambda_i) < \tau\}$, where $\tau=0.1$.
    - **Difficulty classification**:
        - Hard: $\lambda_s^* \leq 0.4$ (collapses under mild masking → heavy reliance on visual details)
        - Medium: $0.4 < \lambda_s^* < 0.7$
        - Easy: $\lambda_s^* \geq 0.7$ (remains correct under heavy masking → answerable from textual cues)
        - Unsolved: incorrect even on the original unmasked image
    - **Design Motivation**: If visual information is critical, even minor image corruption should cause model errors—these constitute "hard" samples.

2. **CMAB (Cross-Modality Attention Balance) — Attention-Based Difficulty Assessment**

    - **Mechanism**: Analyze the model's attention allocation between image tokens and text tokens during response generation.
    - Compute the per-layer attention ratio for each generated token: $\rho^{(l,t)} = S_{img}^{(l,t)} / S_{txt}^{(l,t)}$.
    - Compute the geometric mean across layers (excluding the first and last): $\rho_t = \exp\left(\frac{1}{L_{layers}-2}\sum_{l=2}^{L_{layer}-1}\log(\rho^{(l,t)}+\epsilon)\right)$.
    - Compute the sample-level attention balance: $\bar{\rho} = \frac{1}{T}\sum_{t=1}^T \rho_t$.
    - **Difficulty classification**:
        - Easy: $\bar{\rho} < 0.1$ or $\bar{\rho} > 1.9$ (single-modality dominance; no complex cross-modal reasoning required)
        - Medium: $0.1 \leq \bar{\rho} < 0.4$ or $1.6 < \bar{\rho} \leq 1.9$
        - Hard: $0.4 \leq \bar{\rho} \leq 1.6$ (balanced exploitation of both modalities → genuine cross-modal reasoning)
    - **Design Motivation**: When attention is evenly distributed across image and text, both modalities are indispensable—these are the truly "hard" multimodal samples.

3. **Training Paradigm Comparison**

    - GRPO-only: apply GRPO directly on difficulty-stratified data.
    - SFT+GRPO: perform SFT on one difficulty subset, then apply GRPO on another.
    - All possible difficulty combinations are exhaustively evaluated (e.g., mid→hard, hard→mid, rand→hard).

### Loss & Training
- Backbone model: Qwen2.5VL-7B
- SFT implemented with the LLaMA-Factory framework
- GRPO implemented with the Swift framework
- Hardware: NVIDIA A800-SXM4 (8×80 GB) × 5 nodes + NVIDIA H20 (8×96 GB) × 2 nodes

## Key Experimental Results

### Main Results (PISM Difficulty Stratification — Visual Reasoning Data)

| Training Paradigm | MathVista | MMVet | OCRBench | HBench | MMMU | MMStar |
|---------|-----------|-------|----------|--------|------|--------|
| GRPO-only (fullset) | 53.4 | 41.7 | 76.2 | 67.4 | 0.440 | 0.607 |
| SFT(mid)+GRPO(hard) | 67.3 | 40.6 | 75.0 | 68.5 | 0.507 | 0.609 |
| SFT(hard)+GRPO(mid) | 67.3 | 39.3 | 74.2 | 67.6 | 0.502 | 0.608 |
| GRPO-only (random) | 68.2 | 53.3 | 77.3 | 68.3 | 0.541 | 0.637 |
| **GRPO-only (mid+hard)** | **68.3** | **48.3** | **77.8** | **68.8** | **0.547** | **0.639** |

### Ablation Study (PISM vs. CMAB — Visual Reasoning Data)

| Strategy | MathVista | MMVet | MMMU | MMStar | Notes |
|------|-----------|-------|------|--------|------|
| PISM: GRPO(mid+hard) | 68.3 | 48.3 | 0.547 | 0.639 | Superior on perception tasks |
| CMAB: GRPO(mid+hard) | **69.0** | 48.6 | 0.542 | 0.628 | Superior on reasoning tasks |
| CMAB: GRPO(random) | 68.2 | 43.6 | **0.556** | **0.642** | Random sampling also competitive |

**Data distribution (PISM)**: 20,633 visual perception samples → Easy 7,827 / Medium 4,872 / Hard 1,454 / Unsolved 6,480
**Data distribution (CMAB)**: 27,133 visual reasoning samples → Easy 2,170 / Medium 3,604 / Hard 2,166 / Unsolved 19,193

### Key Findings

1. **GRPO-only consistently outperforms SFT+GRPO**: Across all benchmarks, difficulty-stratified GRPO-only surpasses all SFT+GRPO combinations, challenging the widely held assumption that SFT is a prerequisite for GRPO.
2. **SFT induces "pseudo-CoT"**: SFT relies on human-designed reasoning templates, potentially encouraging superficial pattern matching rather than genuine logical reasoning and increasing hallucination risk.
3. **Data quality > data quantity**: GRPO(mid+hard) with ~6k samples outperforms GRPO(fullset) with ~27k samples, yielding a 14.9% improvement on MathVista.
4. **PISM and CMAB are complementary**: PISM is stronger on perception-oriented tasks (OCRBench, MMVet), while CMAB excels on reasoning-oriented tasks (MathVista, MMMU).
5. **Simplified training pipeline**: GRPO-only not only achieves superior performance but also eliminates the computational overhead of the SFT stage.

## Highlights & Insights

- **Defining "difficulty" from a multimodal perspective constitutes the paper's core contribution.** Unlike simple reliance on problem-level difficulty labels or rejection sampling, PISM and CMAB capture difficulty characteristics intrinsic to multimodal data from the dimensions of visual sensitivity and attention balance, respectively.
- **The counterintuitive finding that SFT is unnecessary** carries significant practical implications, as it enables a substantially simplified training pipeline.
- The experimental design is exceptionally systematic: all SFT+GRPO difficulty combinations are exhaustively evaluated, ruling out coincidental results.
- The design of PISM draws inspiration from **adversarial robustness evaluation**—measuring model dependency by perturbing the input.

## Limitations & Future Work

- Validation is limited to Qwen2.5VL-7B; generalizability across additional models requires further investigation.
- PISM incurs high computational cost, requiring 100 inference passes per sample (10 masking ratios × 10 repetitions).
- CMAB requires access to intermediate-layer attention weights and is therefore inapplicable to API-based models.
- Difficulty thresholds (e.g., $\lambda_{hard}=0.4$, $\lambda_{easy}=0.7$) must be manually specified.
- Fusion strategies combining PISM and CMAB (e.g., weighted aggregation of both difficulty scores) remain unexplored.

## Related Work & Insights

- DeepSeek-R1 established the GRPO post-training paradigm but did not investigate multimodal data sampling.
- VLM-R1, Visual-RFT, and related works introduce RL-based reasoning into multimodal settings without considering data difficulty.
- The paper resonates with curriculum learning but reaches a more radical conclusion by questioning whether any curriculum is necessary.
- Implication: future multimodal post-training pipelines should **prioritize data diagnosis before selecting a training strategy**.

## Rating
- Novelty: ⭐⭐⭐⭐ (PISM and CMAB are creative and well-motivated difficulty metric designs)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (highly systematic paradigm comparison across 6 benchmarks and 10+ training configurations)
- Writing Quality: ⭐⭐⭐⭐ (clear structure, though some sections are notation-dense)
- Value: ⭐⭐⭐⭐⭐ (directly actionable guidance for multimodal post-training practice)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] DIVA-GRPO: Enhancing Multimodal Reasoning through Difficulty-Adaptive Variant Advantage](../../ICLR2026/multimodal_vlm/diva-grpo_enhancing_multimodal_reasoning_through_difficulty-adaptive_variant_adv.md)
- [\[CVPR 2026\] World-Env: Leveraging World Model as a Virtual Environment for VLA Post-Training](../../CVPR2026/multimodal_vlm/rehearsevla_simulated_post-training_for_vlas_with_physically-consistent_world_mo.md)
- [\[NeurIPS 2025\] Enhancing Outcome Reward-Based RL Training of MLLMs with Self-Consistency Sampling](../../NeurIPS2025/multimodal_vlm/enhancing_the_outcome_reward-based_rl_training_of_mllms_with_self-consistency_sa.md)
- [\[ICCV 2025\] SCAN: Bootstrapping Contrastive Pre-training for Data Efficiency](../../ICCV2025/multimodal_vlm/scan_bootstrapping_contrastive_pre-training_for_data_efficiency.md)
- [\[AAAI 2026\] Aligning the True Semantics: Constrained Decoupling and Distribution Sampling for Cross-Modal Alignment](aligning_the_true_semantics_constrained_decoupling_and_distr.md)

</div>

<!-- RELATED:END -->
