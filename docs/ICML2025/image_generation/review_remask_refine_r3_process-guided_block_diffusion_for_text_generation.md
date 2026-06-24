---
title: >-
  [Paper Note] Review, Remask, Refine (R3): Process-Guided Block Diffusion for Text Generation
description: >-
  [ICML2025 (MOSS Workshop)][Image Generation][masked diffusion model] Proposed the R3 (Review, Remask, Refine) framework, which leverages a Process Reward Model (PRM) at inference time to evaluate intermediate generated blocks of a masked-diffusion model, proportionally remasks and regenerates low-quality blocks, achieving training-free targeted error correction, and obtaining significant improvements on mathematical reasoning tasks with extremely low PRM call budgets.
tags:
  - "ICML2025 (MOSS Workshop)"
  - "Image Generation"
  - "masked diffusion model"
  - "process reward model"
  - "iterative refinement"
  - "remasking"
  - "text generation"
  - "inference-time scaling"
date: 2026-05-08
content_hash: 0b228785cd884e8a
---

# Review, Remask, Refine (R3): Process-Guided Block Diffusion for Text Generation

**Conference**: ICML2025 (MOSS Workshop)  
**arXiv**: [2507.08018](https://arxiv.org/abs/2507.08018)  
**Author**: Nikita Mounier, Parsa Ideahpour (University of Pennsylvania)
**Area**: Image Generation  
**Keywords**: masked diffusion model, process reward model, iterative refinement, remasking, text generation, inference-time scaling

## TL;DR

Proposed the R3 (Review, Remask, Refine) framework, which leverages a Process Reward Model (PRM) at inference time to evaluate intermediate generated blocks of a masked-diffusion model, proportionally remasks and regenerates low-quality blocks, achieving training-free targeted error correction, and obtaining significant improvements on mathematical reasoning tasks with extremely low PRM call budgets.

## Background & Motivation

Masked Diffusion Models (MDMs) have recently emerged as a powerful paradigm for text generation, progressively generating coherent text through iterative demasking/denoising, with representative models including LLaDA and BD3-LM. The core advantage of such models lies in their iterative nature, which allows backtracking and correction of generated content. However, the critical question remains **which parts should be remasked**—random remasking is highly inefficient and lacks targeted error-correction capabilities.

Limitations of Prior Work:
- **ReMDM**: Allows remasking of generated tokens at inference time, but lacks an intelligent remasking strategy, failing to precisely locate errors.
- **d1 Framework**: Utilizes SFT + RL for post-training pre-trained masked diffusion models to enhance inference capability, but incurs additional training costs.
- **Best-of-N Sampling**: Generates N candidates for each block and selects the best, which performs well but has an extremely high number of PRM calls (requiring N evaluations for every generated block), leading to massive computational overhead.
- **Outcome Reward Model (ORM)**: Only evaluates the final output, incapable of providing fine-grained feedback on intermediate generation steps.

Key Insight: Process Reward Models (PRMs) can evaluate the quality of intermediate steps, making them naturally suited for guiding iterative generation—blocks with low PRM scores require more correction, while blocks with high PRM scores should be preserved. This signal can be converted into an intelligent remasking strategy.

## Method

### Overall Architecture

R3 consists of a three-stage cyclic inference-time error correction workflow:
1. **Review**: Score the generated text blocks using a PRM.
2. **Remask**: Proportionally remask more tokens in low-quality blocks based on their PRM scores.
3. **Refine**: Regenerate the masked parts using the diffusion model and select the best candidate to replace the original text.

Both core components are off-the-shelf pre-trained models and do not require any fine-tuning:
- Base Maskable Diffusion Model $M_{\text{diff}}$: Given a sequence containing masked tokens, it predicts the content of the masked positions.
- Process Reward Model $M_{\text{PRM}}$: Evaluates the quality score of a text block $x_b$ conditional on context $C_b$, represented as $S_b = M_{\text{PRM}}(x_b | C_b)$, where $S_b \in [0,1]$.

### Windowed Evaluation and Batch Refinement Mechanism

R3 constructs text block by block, applying a windowed evaluation and refinement strategy:

**Step 1 — Generate Current Block**: Generate block $x_j$ using $M_{\text{diff}}$, and append it to the existing sequence $X^{(j+1)} = X^{(j)} \oplus x_j$.

**Step 2 — Window Review**: Every $K$ blocks, score the blocks in the active window $W_j = \{x_{j-K+1}, \ldots, x_j\}$ sequentially using the PRM, yielding a set of scores $\mathcal{S}_{W_j}$.

**Step 3 — Trigger Refinement**: If the minimum score in the window satisfies $\min(\mathcal{S}_{W_j}) < \tau_{\text{thresh}}$ (default 0.8), refinement is triggered.

**Step 4 — Proportional Remasking and Candidate Generation**:
- For each block $x_b$ in the window, calculate the remasking probability based on its PRM score $S_b$.
- The remasking proportion is $\rho_b = \beta_I \cdot \tilde{P}_R(S_b)$, where $\beta_I$ is the intensity factor.
- Generate $N_S$ candidate refined versions.

**Step 5 — Candidate Scoring and Selection**: Score all candidates with the PRM and select the best candidate to replace the original window.

### Mapping from PRM Scores to Remasking Probabilities

This is a key design of R3, mapping the PRM score $S_b$ to a remasking probability $P_R(S_b)$:

1. Calculate the intermediate quality metric: $q_b = \exp(-\alpha_B \cdot S_b)$, where $\alpha_B$ (default 10.0) controls the steepness of the exponential decay.
2. Normalize within the window to $[p_{\min}, 1]$:

$$P_R(S_b) = p_{\min} + (1 - p_{\min}) \cdot \frac{q_b - \min_{b' \in W} q_{b'}}{\max_{b' \in W} q_{b'} - \min_{b' \in W} q_{b'} + \epsilon}$$

where $p_{\min}$ (e.g., 0.01) ensures that even high-scoring blocks have a tiny probability of being re-evaluated. This mapping ensures that low PRM scores (high $q_b$) correspond to high remasking probabilities, capturing the intuitive concept of **lower scores, more refinement**.

### Computational Efficiency Analysis

Let the total number of blocks be $N_{\text{total}}$ and the window size be $K$:
- **Best-of-N**: Each block requires $N_S$ PRM calls $\rightarrow N_{\text{total}} \times N_S$ calls in total (e.g., $16 \times 5 = 80$ calls).
- **R3 Best Case**: Requires only $\lceil N_{\text{total}} / K \rceil$ window evaluations (e.g., only 2 calls when $K=8$).
- **R3 Worst Case**: Every window triggers refinement $\rightarrow 2 \times \lceil N_{\text{total}} / K \rceil$ window evaluations (e.g., 4 calls).

## Key Experimental Results

### Experimental Setup
- **Base Diffusion Model**: LLaDA-8B-Instruct
- **Process Reward Model**: Qwen2.5-Math-PRM-7B
- **Evaluation Dataset**: 127 multi-step derivation problems from the MATH 500 dataset
- **Hyperparameters**: Sampling temperature 0.8, PRM threshold $\tau_{\text{thresh}} = 0.8$, remask intensity $\beta_I = 0.8$, number of candidates $N_S = 5$, $\alpha_B = 10.0$, 16 blocks $\times$ 32 tokens = 512 total tokens, 128 demasking steps.

### Table 1: MATH 500 Subset (127 Problems) Accuracy

| Method | Correct Count | Accuracy |
|---|---|---|
| Simple Diffusion (pass@1) | 37 / 127 | 29.13% |
| R3 (K=4) | 42 / 127 | 33.07% |
| R3 (K=6) | 44 / 127 | 34.65% |
| R3 (K=8) | 54 / 127 | **42.52%** |
| Block-wise Best-of-N (BoN) | 61 / 127 | 48.03% |

### Table 2: Comparison of PRM Call Counts Across Methods (16-Block Sequence)

| Method | PRM Calls | Accuracy |
|---|---|---|
| Simple Diffusion | 0 | 29.13% |
| R3 (K=8) | 2-4 | 42.52% |
| R3 (K=6) | 3-6 | 34.65% |
| R3 (K=4) | 4-8 | 33.07% |
| Block-wise BoN (N=5) | 80 | 48.03% |

## Key Findings

1. **PRM-Guided Targeted Correction is Effective**: R3 (K=8) improves the accuracy of the simple diffusion baseline from 29.13% to 42.52%, representing an absolute gain of **13.4%**.
2. **Efficiency Far Exceeds Brute-Force Search**: R3 (K=8) requires only 2-4 PRM calls compared to 80 calls for BoN. R3 achieves **~88% of BoN's accuracy with only ~5% of its PRM computational overhead**.
3. **Larger Windows Perform Better**: K=8 significantly outperforms K=4/K=6, indicating that larger context windows help the PRM make more accurate quality assessments, leading to better refinement results.
4. **Qualitative Case Validation**: In a trigonometry problem, R3 successfully identified and corrected a computational error (e.g., correcting $b = 2\pi/\pi = 3$ to $b = 2$).

## Highlights & Insights

- **Extremely Simple Design**: The entire framework relies purely on the combination of two off-the-shelf pre-trained models, requiring no training or fine-tuning, and works out-of-the-box.
- **Elegant Balance Between Computation and Quality**: Through windowed evaluation and conditional-triggered refinement, redundant evaluation is avoided for every block, concentrating PRM calls on regions that truly require correction.
- **Score-Driven Soft Remasking**: Instead of binary options of remasking all or nothing, the remasking proportion is determined proportionally based on the PRM score, preserving information from high-quality parts.
- **High Generality**: Theoretically applicable to any combination of masked-diffusion models and PRMs, and is not limited to a specific domain.

## Limitations & Future Work

- **Limited Experimental Scale**: Evaluated only on a 127-problem subset of MATH 500, lacking validation on larger scales and broader tasks (e.g., GSM8K, code generation, general text generation).
- **Workshop Paper Depth**: As a MOSS Workshop paper, the methodology and experiments are somewhat preliminary, lacking systematic ablation studies (e.g., the impacts of $\alpha_B$, $\beta_I$, and $\tau_{\text{thresh}}$).
- **Persistent Gap with BoN**: R3 achieves a top performance of 42.52% compared to 48.03% for BoN; whether the efficiency advantage diminishes under larger computational budgets remains unexplored.
- **Dependency on PRMs**: The effectiveness highly depends on the quality of the PRM, which currently primarily targets verifiable tasks like mathematical reasoning, casting doubt on its applicability in open-domain text generation.
- **Fixed Window Size**: $K$ is a fixed hyperparameter; more flexible strategies, such as adaptive window sizes or hierarchical evaluation, are yet to be explored.
- **Lack of Comparison with Other Inference-Time Scaling Methods**: No comparisons with other inference-time methods leveraging diffusion characteristics (such as d1, SVDD).

## Related Work & Insights

- **LLaDA** (Nie et al., 2025): A large language diffusion model that generates text via iterative denoising through masked diffusion, serving as one of the base models for R3.
- **BD3-LM** (Arriola et al., 2025): A block diffusion language model that combines autoregressive block-level generation with intra-block discrete diffusion, where R3 can be directly applied.
- **ReMDM** (Wang et al., 2025): Allows remasking of generated tokens at inference time; R3 builds on this by introducing PRM-guided intelligent remasking.
- **d1** (Zhao et al., 2025): Post-trains masked diffusion models using SFT + RL to enhance reasoning capabilities, complementing the training-free approach of R3.
- **Qwen2.5-Math-PRM** (Zhang et al., 2025): The process reward model used in this paper, providing step-level quality assessment.

**Insight**: The core idea of R3 can be generalized—introducing external quality evaluators into any iterative generation model for targeted refinement, not limited to diffusion models. For instance, similar strategies could be applied to autoregressive models under a draft-then-revise paradigm.

## Rating

- Novelty: ⭐⭐⭐⭐ — The idea of PRM-guided remasking is intuitive, clear, and reasonable, but the method itself is relatively simple and lacks deep technical innovation.
- Experimental Thoroughness: ⭐⭐⭐ — Evaluated only on a single dataset with 127 problems, lacking ablation studies and more baseline comparisons, making the experimental scale quite small.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clearly structured with rigorous method descriptions, assisted by algorithmic pseudocode and qualitative case studies for easier comprehension.
- Value: ⭐⭐⭐⭐ — Proposes a practical, training-free inference-time alignment framework; though the experiments are preliminary, the direction is valuable and establishes a clear baseline for future work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DRiffusion: Draft-and-Refine Process Parallelizes Diffusion Models with Ease](../../CVPR2026/image_generation/driffusion_draft-and-refine_process_parallelizes_diffusion_models_with_ease.md)
- [\[ICML 2025\] Origin Identification for Text-Guided Image-to-Image Diffusion Models](origin_identification_for_text-guided_image-to-image_diffusion_models.md)
- [\[ICCV 2025\] Text Embedding Knows How to Quantize Text-Guided Diffusion Models](../../ICCV2025/image_generation/text_embedding_knows_how_to_quantize_text-guided_diffusion_models.md)
- [\[ICCV 2025\] Rethink Sparse Signals for Pose-guided Text-to-Image Generation](../../ICCV2025/image_generation/rethink_sparse_signals_for_pose-guided_text-to-image_generation.md)
- [\[ICCV 2025\] TeRA: Rethinking Text-guided Realistic 3D Avatar Generation](../../ICCV2025/image_generation/tera_rethinking_text-guided_realistic_3d_avatar_generation.md)

</div>

<!-- RELATED:END -->
