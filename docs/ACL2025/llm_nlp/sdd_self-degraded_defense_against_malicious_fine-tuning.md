---
title: >-
  [Paper Note] SDD: Self-Degraded Defense against Malicious Fine-tuning
description: >-
  [ACL 2025][LLM (Other)][LLM security] SDD achieves defense by training LLMs to generate high-quality but irrelevant benign responses to harmful instructions: when an attacker performs malicious fine-tuning, the model's general capability significantly degrades, rendering it unable to effectively execute malicious instructions.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "LLM security"
  - "malicious fine-tuning defense"
  - "safety alignment"
  - "self-degraded defense"
  - "open-source LLM security"
date: 2026-05-08
content_hash: 6af36948c107c7e2
---

# SDD: Self-Degraded Defense against Malicious Fine-tuning

**Conference**: ACL 2025  
**arXiv**: [2507.21182](https://arxiv.org/abs/2507.21182)  
**Code**: [GitHub](https://github.com/ZeroNLP/SDD)  
**Area**: Others  
**Keywords**: LLM security, malicious fine-tuning defense, safety alignment, self-degraded defense, open-source LLM security

## TL;DR

SDD achieves defense by training LLMs to generate high-quality but irrelevant benign responses to harmful instructions: when an attacker performs malicious fine-tuning, the model's general capability significantly degrades, rendering it unable to effectively execute malicious instructions.

## Background & Motivation

Open-source LLMs face a critical security challenge: **Malicious Fine-Tuning (MFT)** can easily bypass safety alignment mechanisms. Studies show that a mere 100 harmful Q&A pairs can "jailbreak" aligned models like Llama2, and even fine-tuning on benign data can inadvertently weaken safety guardrails. This poses a fundamental threat to the security ecosystem of open-source LLMs, as model publishers cannot control the fine-tuning behaviors of downstream users.

Existing defense methods (such as Vaccine, T-Vaccine, Booster, RepNoise, TAR, etc.) are primarily based on empirical observations rather than rigorous theoretical analysis:
- Vaccine-series methods attempt to counter harmful embedding drift.
- RepNoise disrupts the informational structure of harmful representations.
- TAR enhances safety mechanisms through adversarial training.

The key innovation of this paper lies in **relaxing the objective of safety alignment**: while traditional objectives aim to make models refuse harmful instructions, SDD's goal is merely to ensure that the model does not generate harmful responses. This is achieved by causing the model to lose its general capability after suffering MFT, thereby preventing it from executing any instructions (including malicious ones).

## Method

### Overall Architecture

The SDD framework consists of three steps:
1. Collect harmful instructions and high-quality benign responses.
2. Pair harmful instructions with randomly selected, irrelevant high-quality responses.
3. Apply SFT training on the paired data for LLMs.

When an attacker performs MFT on an SDD-protected model, the optimization process of MFT reduces the probability of the original responses (high-quality benign content), leading to a comprehensive degradation of the model's general capability.

### Key Designs

1. **Theoretical Foundation—Why MFT Disrupts Safety Alignment (Theorem 1)**:
   The paper simplifies LLMs as a combination of a feature selector $\Phi$ and a classifier $w$, dividing features into invariant features (which consistently aid prediction) and spurious features (which are unstably correlated). Theorem 1 proves that the accuracy degradation on safety alignment after MFT is primarily because near-optimal MFT models learn a large number of spurious features $n_s^*$, decreasing safety alignment accuracy.

2. **Relaxed Safety Objective (Theorem 2)**:
   The traditional safety objective is "refusing harmful instructions" (which is too strong and easily disrupted by MFT). The relaxed objective is "not generating harmful responses". Theorem 2 proves that under certain conditions (where the original model possesses more invariant features and fewer spurious features), the model's general capability decreases after MFT, i.e., $\xi_G(\tilde{f}) < \xi_G(\bar{f})$.

3. **Core Mechanism of Self-Degraded Defense**:
   The optimization objective of MFT is to maximize $p(y_c \succ y_o \mid x)$, making the harmful response $y_c$ preferred over the original response $y_o$. Through Bradley-Terry theoretical analysis (Eq. 4-7), this optimization process inevitably reduces $\pi^*(y_o \mid x)$—the probability of the original response.
   
   **The Ingenuity of SDD**: It sets $y_o$ as a high-quality benign response (e.g., steps for making coffee) instead of a refusal response. When MFT decreases the probability of these high-quality responses, the model's general generation capability is destroyed simultaneously.

4. **Dataset Construction**:

    - Collect harmful instructions from BeaverTails, covering 14 harmful categories with a total of 8K instances.
    - Collect high-quality responses from LIMA and Alpaca.
    - **Random Matching**: Randomly assign a high-quality response to each harmful instruction.
    - **Irrelevance Filtering**: Calculate semantic similarity of instruction-response pairs using SentenceBERT, re-sampling if it exceeds a threshold to ensure responses are irrelevant to the harmful instructions.
    - Final format: `<harmful instruction, irrelevant high-quality response>`

5. **Training Process**: Standard SFT training to minimize cross-entropy loss. SDD can be integrated into any stage of the LLM training pipeline (post-pretraining, post-SFT, or post-RLHF).

### Loss & Training

The training loss is the standard cross-entropy:

$$ L = -\sum \log p(y_{\text{irrelevant}} \mid x_{\text{harmful}}) $$

where $x_{\text{harmful}}$ is the harmful instruction, and $y_{\text{irrelevant}}$ is the irrelevant high-quality response. After training, the model will generate high-quality but irrelevant responses to harmful instructions.

## Key Experimental Results

### Main Results

Defense Capability Evaluation (Harmful rate on the LLM-Finetune-Safety benchmark):

| Method | Llama2-7b-chat post-MFT Harmful Rate |
|------|------------------------|
| Vanilla + MFT | High (Baseline) |
| Vaccine + MFT | Slightly decreased |
| T-Vaccine + MFT | Slightly decreased |
| TAR + MFT | Slightly decreased |
| **SDD + MFT** | **0%** |

General Capability Evaluation (MMLU / OpenBookQA):

| Model | MMLU | OpenBookQA |
|------|------|-----------|
| Llama2-7b-chat (Vanilla) | 46.35 | 33.40 |
| Llama2-7b-chat + SDD | 47.04 | 33.00 |
| SDD + BFT | 49.14 | 35.00 |
| SDD + MFT | 29.33 (36%↓) | 13.80 (59%↓) |

| Model | MMLU | OpenBookQA |
|------|------|-----------|
| Llama2-7b (Vanilla) | 38.87 | 31.40 |
| Llama2-7b + SDD | 45.78 | 31.80 |
| SDD + BFT | 45.93 | 32.60 |
| SDD + MFT | 25.79 (33%↓) | 13.40 (57%↓) |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| SDD training data volume | Effective with only 500 samples | Remains effective even if the attacker uses 20x more data |
| SDD_reject variant | Harmful rate comparable to SDD | Possesses explicit refusal capability at the same time |
| Different backbones | Phi-2(2.7B), GLM-3(6B) | Effective across models |

### Key Findings

1. **SDD maintains a 0% harmful rate after MFT**—significantly outperforming all baseline methods.
2. **SDD does not affect normal usage**—general capabilities are maintained or even slightly improved during direct inference and benign fine-tuning scenarios.
3. **Significant degradation of general capability after MFT** is the design goal (MMLU drops by 33-36%, OpenBookQA drops by 57-59%).
4. **SDD is highly defense-efficient**—requiring only 500 training samples, it remains effective even when the attacker uses 20x malicious data.
5. **SDD increases the cost of abuse**—attackers need to prepare a much larger scale of malicious data.

## Highlights & Insights

1. **Reverse-thinking defense philosophy**: Rather than seeking "correct model refusal", it aims to "make the model dumb after being attacked". This "self-destruct" strategy is highly novel in the security domain.
2. **Theoretical analysis driving methodology design**: From Theorem 1 (analyzing why MFT works) to Theorem 2 (proving general capabilities can degrade) to the SDD method, the logical chain is complete.
3. **Utilizing MFT's own optimization objective**: SDD ingeniously leverages the property that MFT must reduce the probability of original responses, converting this into the degradation of general capabilities.
4. **Compatibility with existing pipelines**: SDD is a simple SFT process that does not require complex adversarial training or meta-learning, allowing it to be directly integrated into any training stage.

## Limitations & Future Work

1. **Unnatural response patterns**: SDD makes the model generate irrelevant content for harmful instructions instead of explaining refusal like a human. Although the SDD_reject variant is provided, the response remains unnatural.
2. **General capability degradation is a "lose-lose" scenario**: While it prevents harmful outputs, the model also becomes incapable of providing helpful harmless responses—this might be unacceptable in certain scenarios.
3. **Simplification of theoretical assumptions**: Simplifying LLMs into a feature selector + classifier is too coarse, and the plausibility of Assumption 1 (linear extrapolation) is worth questioning.
4. **Adaptive attackers**: If attackers know the SDD mechanism, they might design targeted attack strategies (such as detecting general capability degradation before adjusting their fine-tuning strategy).
5. **Only two backbone models tested**: The main experiments only used Llama2-7b and 7b-chat, without validation on larger models or newer architectures.

## Related Work & Insights

- **Vaccine (Huang et al.)**: Counters harmful embedding drift by adding perturbations during the alignment phase, but fundamentally still pursues traditional safety objectives.
- **TAR**: Uses adversarial training and meta-learning to enhance LLM safety, which incurs higher computational costs.
- **RepNoise (Rosati et al.)**: Disrupts the informational structure of harmful representations, sharing similarities with SDD's philosophy (both doing "subtraction" rather than "addition").
- **DPO (Rafailov et al., 2023)**: SDD's theoretical analysis utilizes the Bradley-Terry framework of DPO to analyze the MFT optimization objective.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The "self-degraded" defense philosophy is highly novel, beautifully integrating theory with methodology design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple settings but with limited backbone models, lacking evaluations against adaptive attackers.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear, though some formulas could be simplified.
- Value: ⭐⭐⭐⭐ Holds direct practical significance for open-source LLM security, but the acceptability of the "lose-lose" strategy remains to be seen.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Self-Tuning: Instructing LLMs to Effectively Acquire New Knowledge through Self-Teaching](self-tuning_instructing_llms_to_effectively_acquire_new_knowledge_through_self-t.md)
- [\[ACL 2025\] HFT: Half Fine-Tuning for Large Language Models](hft_half_fine-tuning_for_large_language_models.md)
- [\[ACL 2025\] Efficient Ensemble for Fine-tuning Language Models on Multiple Datasets](efficient_ensemble_for_fine-tuning_language_models_on_multiple_datasets.md)
- [\[ACL 2025\] GORP: Continual Gradient Low-Rank Projection Fine-Tuning for LLMs](gorp_continual_gradient_projection.md)
- [\[NeurIPS 2025\] Triplets Better Than Pairs: Towards Stable and Effective Self-Play Fine-Tuning for LLMs](../../NeurIPS2025/llm_nlp/triplets_better_than_pairs_towards_stable_and_effective_self-play_fine-tuning_fo.md)

</div>

<!-- RELATED:END -->
