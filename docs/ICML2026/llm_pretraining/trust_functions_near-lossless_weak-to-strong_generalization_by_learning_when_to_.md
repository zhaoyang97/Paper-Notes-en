---
title: >-
  [Paper Note] Trust Functions: Near-Lossless Weak-to-Strong Generalization by Learning When to Trust the Weak Teacher
description: >-
  [ICML 2026][LLM Pretraining][Weak-to-Strong Generalization] This paper reframes "Weak-to-Strong Generalization" as a **data selection** problem. It proposes the "Trust Function…
tags:
  - "ICML 2026"
  - "LLM Pretraining"
  - "Weak-to-Strong Generalization"
  - "Trust Functions"
  - "Data Filtering"
  - "Teacher Hidden States"
  - "Superalignment"
date: 2026-05-08
content_hash: 997ddcd0b1682372
---

# Trust Functions: Near-Lossless Weak-to-Strong Generalization by Learning When to Trust the Weak Teacher

**Conference**: ICML 2026  
**arXiv**: [2606.01000](https://arxiv.org/abs/2606.01000)  
**Code**: Mentioned in the paper (Code / Website links)  
**Area**: Alignment RLHF / Weakly Supervised Learning / Data Selection  
**Keywords**: Weak-to-Strong Generalization, Trust Functions, Data Filtering, Teacher Hidden States, Superalignment

## TL;DR
This paper reframes "Weak-to-Strong Generalization" as a **data selection** problem. It proposes the "Trust Function," a lightweight MLP that reads the last-layer hidden states of a teacher model to predict the reliability of weak labels. By selecting only high-trust samples to train a strong student, the method achieves near-lossless or even super-recovery performance relative to ground-truth supervision across multiple tasks and can be iterated into a "Weak-to-Strong Chain" to amplify gains.

## Background & Motivation
**Background**: As LLMs approach or exceed human levels in complex tasks, the traditional assumption that "humans provide reliable supervision" collapses. Superalignment shifts toward using a weak teacher $\pi_{\mathcal{W}}$ to train a stronger student $\pi_{\mathcal{S}}$. Pioneering work by Burns et al. showed that weak supervision allows students to surpass teachers, but a gap persistent remains compared to ground-truth (GT) supervision.

**Limitations of Prior Work**: Pseudo-labels from weak teachers contain two types of systematic errors: (i) incorrect labels are inherited by strong models along the data's geometric structure; (ii) task-relevant directions not present in the weak teacher's representation space cannot be transferred. Consequently, weak supervision often leads to instability or degradation under distribution drift, failing to close the gap to GT levels.

**Key Challenge**: Existing attempts at "data selection" generally rely on **output-layer heuristics**—such as entropy, multi-model consistency, or self-assessment. These signals are poorly calibrated on complex tasks (high scores for confident errors, low scores for correct-but-uncertain samples) and are particularly fragile under distribution drift. The fundamental issue is that **output-layer signals are insufficient for judging the reliability of weak labels**.

**Goal**: Given a fixed architecture and training algorithm, identify the subset within the weak annotation pool that "truly makes the student stronger" and unify the formalization of how label reliability is determined.

**Key Insight**: The authors observe that prior work (Kadavath et al. 2022; Kuhn et al. 2023) found that **intermediate representations themselves** encode separable signals of "whether the answer is correct," which are often flattened by the decoding layer. Therefore, one should train a discriminator on hidden states rather than relying on decoded probabilities.

**Core Idea**: Use a small MLP $\tau$ to predict "whether this weak label is actually correct" directly from the weak teacher's hidden states. Perform SFT/GRPO only on high-trust samples and treat the resulting student as the next-generation teacher in a "Weak-to-Strong Chain."

## Method

### Overall Architecture
The framework, named **Learning to Trust (L2T)**, requires two datasets: a labeled source set $\mathcal{D}_{\ell}=\{(x_i,y_i)\}$ and an unlabeled target set $\mathcal{D}_u=\{x_j\}$, which do not need to be from the same distribution. The process consists of four steps:

1. The weak teacher $\pi_{\mathcal{W}}$ runs a forward pass on $\mathcal{D}_u$ to produce weak labels $\hat{y}=\pi_{\mathcal{W}}(x)$, while caching the hidden states $g_{\pi_{\mathcal{W}}}(x,\hat{y})$ of the final layer and final generated token.
2. On $\mathcal{D}_{\ell}$, a Neural Trust Function (NTF) $\tau$ is trained as a binary classifier using pairs of $(g_{\pi_{\mathcal{W}}}(x,\hat{y}),\,\text{is\_correct})$ based on whether the prediction matches the ground truth.
3. $\tau$ assigns a trust score $t=\tau(g_{\pi_{\mathcal{W}}}(x,\hat{y}))$ to each sample in $\mathcal{D}_u$, and a high-trust subset $\tilde{\mathcal{D}}_u$ is selected (e.g., top-$n$).
4. The strong student $\pi_{\mathcal{S}}$ is trained using SFT or GRPO on the weak labels in $\tilde{\mathcal{D}}_u$. This process does not require ground truth for $\mathcal{D}_u$.

The chained version treats the current student as the next-generation teacher, repeating the four steps to iteratively amplify gains.

### Key Designs

1. **Neural Trust Function (NTF) Based on Hidden States**:
    - **Function**: Maps the weak teacher's internal representation $g_{\pi_{\mathcal{W}}}(x,\hat{y})\in\mathbb{R}^d$ to a trust score $\tau(\cdot) \in [0,1]$, estimating the probability that the weak label is true.
    - **Mechanism**: The input consists of the hidden vector from the last layer and final generated token (which aggregates the prefix and intermediate reasoning via attention). $\tau$ is a residual MLP composed of RMSNorm-SwiGLU blocks (with dropout/stochastic depth), followed by an RMSNorm and linear head to produce logits, converted to probabilities via sigmoid. The loss is class-reweighted BCE to handle label imbalance. Training samples are automatically constructed on $\mathcal{D}_{\ell}$ using exact match (MCQA/Math) or best-move match (Chess).
    - **Design Motivation**: Output-layer confidence is systematically miscalibrated on hard problems; intermediate layers capture signals of "whether I likely answered correctly" much earlier. Moving the discriminator to the hidden space avoids confident-but-wrong pitfalls. The pipeline's overhead is dominated by the teacher's forward pass, as $\tau$ is a negligible MLP: $C_{\text{total}}=O(\bar{C}_{\text{teacher}}(|\mathcal{D}_{\ell}|+|\mathcal{D}_u|)+C_{\text{NTF}}(e|\mathcal{D}_{\ell}|+|\mathcal{D}_u|))$.

2. **Zero-Shot Deployment under In-domain Distribution Drift**:
    - **Function**: Relaxes the strong assumption that labels must exist for the target domain. $\tau$ is trained on the source distribution but deployed for zero-shot scoring on a target domain with the same task interface but different data distribution.
    - **Mechanism**: Generalization scenarios are categorized into three tiers: ID (held-out from the same benchmark), OOD$_{\text{dist}}$ (same task interface, different distribution, e.g., MMLU $\to$ ARC-Easy), and OOD$_{\text{domain}}$ (different task interface, e.g., MCQA $\to$ Chess). All "zero-shot transfer" claims refer to OOD$_{\text{dist}}$. Table 1 shows NTF achieves AUCs of 0.83–0.92 and purity of 0.69–0.98 in ID/OOD$_{\text{dist}}$ settings.
    - **Design Motivation**: Realistic label distributions are unbalanced—large labeled sets like MMLU/MATH are accessible, while target domains like AIME are scarce. Allowing $\tau$ to be trained once on a source domain and deployed on unlabeled target domains amortizes costs.

3. **Weak-to-Strong Chain**:
    - **Function**: Uses the student $\pi_{\mathcal{S}}^{(1)}$ trained via L2T as the next-generation teacher $\pi_{\mathcal{W}}^{(2)}$, and trains an even larger student $\pi_{\mathcal{S}}^{(2)}$ using the same NTF trust filtering.
    - **Mechanism**: Each student generation's accuracy on the target domain increases monotonically (referred to as snowballing) because they are trained on high-purity weak labels. When a new student becomes a teacher, the purity of its produced weak labels is higher, increasing the available sample volume and average accuracy even when using the same $\tau$.
    - **Design Motivation**: While single-generation L2T approaches GT performance, there is room for improvement as student scales increase. The chain structure amplifies benefits without new components and follows the same protocol, facilitating scalability.

### Loss & Training
- **NTF Training**: Class-reweighted BCE + AdamW (with weight decay). Evaluated using AUC / ECE / Brier / Purity (proportion of truly correct labels in the top-trust subset).
- **Strong Student Training**: MCQA tasks use LoRA-SFT to fit weak labels on top-$n$ high-trust samples. Quantitative reasoning uses GRPO for RL on high-trust rollouts. Recovery is defined as $\text{Recovery}=\frac{\text{Baseline}-\text{Base}}{\text{GT}-\text{Base}}\times 100\%$, measuring the "recovery ratio relative to GT training."

## Key Experimental Results

### Main Results
World Knowledge (Average accuracy across 5 MCQA benchmarks; Recovery% in parentheses):

| Teacher $\to$ Student | Naive | I-Confidence | ICl+I-Conf | Reward Model | **NTF (Ours)** | Ground Truth |
|---|---|---|---|---|---|---|
| OLMo2-1B $\to$ OLMo2-7B | 69.3 (48.3) | 69.2 (47.1) | 72.0 (79.3) | 68.8 (42.5) | **73.7 (98.9)** | 73.8 |
| OLMo2-1B $\to$ OLMo2-13B | 74.7 (12.2) | 75.1 (17.6) | 77.9 (55.4) | 78.4 (62.2) | **80.9 (95.9)** | 81.2 |
| Qwen3-0.6B $\to$ Qwen3-1.7B | 74.0 (86.0) | 74.3 (91.2) | 74.4 (93.0) | 71.7 (45.6) | **75.0 (103.5)** | 74.8 |
| Qwen3-0.6B $\to$ Qwen3-14B | 86.0 (86.8) | 85.7 (82.9) | 86.5 (93.4) | 86.1 (88.2) | **87.1 (101.3)** | 87.0 |

Across 8 settings, NTF was statistically indistinguishable from GT in 5 cases (near-lossless) and significantly better than GT in 1 case (super-recovery), consistently outperforming all baselines.

### Ablation Study
Calibration metrics of NTF across domains (Table 1; Qwen3-0.6B for World Knowledge/Games, Qwen3-1.7B / Gemma3-1B for Quantitative Reasoning):

| Domain | AUC ↑ | ECE ↓ | Brier ↓ | Purity ↑ |
|---|---|---|---|---|
| World Knowledge | 0.92 | 0.03 | 0.07 | 0.98 |
| Quantitative Reasoning (Omni) | 0.83 | 0.11 | 0.13 | 0.69 |
| Quantitative Reasoning (MATH) | 0.84 | 0.14 | 0.17 | 0.95 |
| Strategy Games | 0.91 | 0.02 | 0.11 | 0.95 |

### Key Findings
- **Gains are not just from "filtering error labels"**: The authors attribute performance to three mechanisms: preserving samples that induce an implicit easy-first curriculum; occasionally "correcting" suboptimal labels in GT (observed in MATH); and ensuring gradient directions of filtered samples are better aligned.
- **NTF remains effective for extremely weak teachers**: Qwen3-1.7B has <5% accuracy on AIME, yet with NTF, it achieves near-lossless GT recovery, proving the trust function can identify rare reliable samples in low-purity pools.
- **OOD$_{\text{domain}}$ (different task interfaces) leads to significant degradation**, indicating "trust" is coupled with task interfaces and output spaces. Cross-interface transfer remains an open problem.

## Highlights & Insights
- **Problem Redefinition**: Shifts W2S focus from "loss/algorithm design" to "data selection." The trust function serves as a unifying umbrella for entropy, agreement, self-assessment, and reward models.
- **Negligible Computational Overhead**: NTF is a tiny MLP using hidden states already calculated; compared to external reward models, it offers lower deployment costs and better performance.
- **Chain Amplification**: Chained W2S effectively treats data selection as iterative self-training, "purifying" weak supervision in a self-play style, providing a sustainable bootstrap path for superalignment.

## Limitations & Future Work
- **Reliance on Source Labels**: While target GT is not needed, "labeled source domains with matching task interfaces" are required, which may not be directly available in extreme superalignment scenarios.
- **Cross-Interface (OOD$_{\text{domain}}$) Failure**: The trust function is tightly coupled with task interfaces. Transferring to different tasks (e.g., MCQA $\to$ Math) causes degradation.
- **Scale Verification**: Evaluation is limited to medium-scale models (OLMo2 / Qwen3 1B–14B). Whether near-lossless performance holds at the 70B+ scale requires further validation.
- **Limits of Chained Gains**: Though "snowballing" is demonstrated, analysis of the collapse point is missing—how many generations until the chain becomes unstable?

## Related Work & Insights
- **vs Burns et al. 2023 (Original W2S)**: The latter focuses on training objectives (e.g., confidence loss); Ours keeps loss/architecture intact and filters data, closing the GT gap faster.
- **vs Internal/Verbalized Confidence**: Both measure teacher reliability, but heuristics use output signals; Ours proves hidden states + a small discriminator are more stable on hard problems and transfer zero-shot across benchmarks.
- **vs Reward Model Filtering**: RM is a general discriminator, but general reward signals do not map one-to-one to weak label correctness; NTF specifically models "correctness" for W2S.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Reforming W2S as data selection and using hidden state discriminators as a formalization is a significant shift in perspective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive coverage across three domains, two model families, multiple scales (1B–14B), and various baselines.
- **Writing Quality**: ⭐⭐⭐⭐ Clear formalization and rigorous generalization regime definitions.
- **Value**: ⭐⭐⭐⭐⭐ Provides an engineering-grade solution for near-lossless weak-to-strong generalization with direct relevance to superalignment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Lossless Vocabulary Reduction for Auto-Regressive Language Models](../../ICLR2026/llm_pretraining/lossless_vocabulary_reduction_for_auto-regressive_language_models.md)
- [\[ICML 2026\] Data Difficulty and the Generalization--Extrapolation Tradeoff in LLM Fine-Tuning](data_difficulty_and_the_generalization--extrapolation_tradeoff_in_llm_fine-tunin.md)
- [\[ICML 2026\] Tuning the Implicit Regularizer of Masked Diffusion Language Models: Enhancing Generalization via Insights from k-Parity](tuning_the_implicit_regularizer_of_masked_diffusion_language_models_enhancing_ge.md)
- [\[ICML 2026\] Focus and Dilution: The Multi-stage Learning Process of Attention](focus_and_dilution_the_multi-stage_learning_process_of_attention.md)
- [\[ICLR 2026\] Block-Sample MAC-Bayes Generalization Bounds](../../ICLR2026/llm_pretraining/block-sample_mac-bayes_generalization_bounds.md)

</div>

<!-- RELATED:END -->
