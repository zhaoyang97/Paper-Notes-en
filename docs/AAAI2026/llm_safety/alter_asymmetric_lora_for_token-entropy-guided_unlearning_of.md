---
title: >-
  [Paper Note] ALTER: Asymmetric LoRA for Token-Entropy-Guided Unlearning of LLMs
description: >-
  [AAAI 2026][LLM Safety][Machine Unlearning] This paper proposes ALTER, a framework that combines an asymmetric LoRA architecture with token-level Tsallis entropy guidance to achieve precise unlearning of target knowledge…
tags:
  - "AAAI 2026"
  - "LLM Safety"
  - "Machine Unlearning"
  - "Asymmetric LoRA"
  - "Token Entropy"
  - "Parameter Isolation"
  - "Knowledge Decoupling"
date: 2026-05-08
content_hash: 540fd691d4380354
---

# ALTER: Asymmetric LoRA for Token-Entropy-Guided Unlearning of LLMs

**Conference**: AAAI 2026
**arXiv**: [2603.01792v1](https://arxiv.org/abs/2603.01792v1)  
**Code**: [https://github.com/MastrOrigami/ALTER.git](https://github.com/MastrOrigami/ALTER.git)  
**Area**: Model Compression
**Keywords**: Machine Unlearning, Asymmetric LoRA, Token Entropy, Parameter Isolation, Knowledge Decoupling

## TL;DR

This paper proposes ALTER, a framework that combines an asymmetric LoRA architecture with token-level Tsallis entropy guidance to achieve precise unlearning of target knowledge in LLMs. A parameter isolation mechanism is employed to preserve the model's general capabilities, achieving state-of-the-art performance on three benchmarks: TOFU, WMDP, and MUSE.

## Background & Motivation

As LLMs scale in size and training data diversity, models inevitably encode sensitive information, private data, or illegal content. The "right to be forgotten" mandated by regulations such as the GDPR has made LLM unlearning a prominent research direction. However, existing unlearning methods face two core challenges:

1. **Knowledge entanglement**: Continual multi-domain training leads to highly coupled parameter spaces, causing "over-forgetting" during unlearning—inadvertently impairing capabilities in unrelated domains (Figure 1 illustrates the impact of corpus heterogeneity on FT/PEFT and the resulting disorder in LoRA parameter spaces).
2. **Computational cost**: Full-parameter fine-tuning of billion-scale SOTA models is prohibitively expensive.

Limitations of existing methods:
- **Prompt-based / auxiliary model methods**: Limited generalization and robustness.
- **Full-parameter fine-tuning (GA, etc.)**: Achieves unlearning at the cost of catastrophic degradation of general capability (MMLU drops to 23–26%).
- **Standard LoRA**: Although parameter-efficient, it still struggles to precisely isolate forgetting targets within a multi-domain coupled parameter space.

## Core Problem

How can parameter-efficient fine-tuning (PEFT) achieve precise unlearning within a multi-domain coupled parameter space while preserving overall model performance? Specifically, this requires simultaneously addressing: (1) parameter isolation among unlearning sub-tasks; (2) decoupling between unlearning tasks and retention tasks; and (3) fine-grained token-level unlearning (to avoid destroying grammatical structure through sentence-level forgetting).

## Method

### Overall Architecture

ALTER is a lightweight two-stage unlearning framework:

- **Stage I (Token Entropy Capture)**: A shared matrix A within LoRA learns high-entropy tokens (structural language elements).
- **Stage II (Asymmetric Unlearning)**: An asymmetric LoRA architecture achieves token-level unlearning within target sub-domains through parameter isolation.

The overall weight update is formulated as:
$$\mathbf{W} = \mathbf{W}_0 + \Delta\mathbf{W} = \mathbf{W}_0 + \left(\mathbf{B}_r + \sum_{d=1}^{N} \omega_{f_d} \cdot \mathbf{B}_{f_d}\right)\mathbf{A}$$

where $\mathbf{A}$ is the shared matrix capturing task-agnostic structural knowledge, $\mathbf{B}_{f_d}$ are domain-specific expert matrices for each forgetting sub-domain, and $\mathbf{B}_r$ is the retention expert matrix.

### Key Designs

1. **Asymmetric LoRA Parameter Isolation (Observation I)**:

   Drawing on the finding from HydraLoRA that the shared matrix A typically captures general knowledge while individual B matrices adapt to domain-specific knowledge, ALTER applies this property to unlearning. The complex heterogeneous unlearning problem is decomposed into local optimization tasks over individual sub-domain datasets. Each forgetting expert $\mathbf{B}_{f_d}$ corresponds to a sub-domain $d$ and is initialized from clustering centroids; the retention expert $\mathbf{B}_r$ is initialized from the feature distribution of the retention set.

2. **Token Entropy Guidance Mechanism (Observation II)**:

   The paper observes that token-level entropy exhibits a robust bimodal distribution: high-entropy tokens (e.g., "however," "therefore") are primarily structural language elements, while low-entropy tokens carry knowledge-intensive content (e.g., entity nouns). This distribution remains stable during PEFT (>87% of high-entropy tokens remain uncertain; >92% of low-entropy tokens remain certain). Accordingly, ALTER replaces Shannon entropy with Tsallis entropy for hierarchical modeling:
   $$S_q(x_t) = \frac{1}{q-1}\left(1 - \sum_{i=1}^{V} p_{t,i}^q\right), \quad q > 0$$
   The deformation parameter $q$ provides dual control: $q<1$ enhances the structural invariance of high-entropy tokens in matrix A; $q>1$ disrupts cross-domain associations of low-entropy tokens to enable targeted forgetting.

3. **Entropy-Aware Adaptive Gating (MoE Routing)**:

   An entropy-based adaptive gating mechanism is introduced:
   $$g_d(x_t) = \text{softmax}(W_g^T \cdot S_q(x_t) / \tau)$$
   The routing temperature $\tau$ is dynamically adjusted: for high-entropy tokens ($S_q > 1.2$), $\tau = 0.8$ activates multiple experts to enhance structural robustness; for low-entropy tokens ($S_q \leq 1.2$), $\tau = 0.01$ enforces single-expert precise routing.

4. **Differentiated Inference Paths**:

   During inference, different computation paths are triggered based on a token entropy threshold. High-entropy tokens use multi-expert fusion (aggregating A and the top-3 $\mathbf{B}_{f_d}$ matrices) to preserve structural integrity; low-entropy tokens activate a single-expert bypass (only the highest-weight $\mathbf{B}_{i^*}$), avoiding redundant computation.

### Loss & Training

ALTER employs a hierarchical cascaded loss extended into a three-level optimization objective:

$$\min_{\omega_{f_d}, \omega_r} \beta \sum_{d=1}^{N} \mathbb{E}_{(q,a)\sim\mathcal{D}_{f_d}}[\mathcal{L}_{\text{IHL}}] + \gamma \mathbb{E}_{(q,a)\sim\mathcal{D}_r}[l_r]$$

- **Inverse Hinge Loss $\mathcal{L}_{\text{IHL}}$**: Inspired by classical hinge loss, this reverses the optimization direction to suppress target prediction probabilities while elevating sub-optimal token probabilities on low-entropy tokens.
- **Retention Loss $l_r$**: Reinforces core model capabilities.
- **Strict Gradient Isolation**: Each forgetting expert $\mathbf{B}_{f_d}$ is updated only through gradients from its corresponding sub-domain; $\mathbf{B}_r$ is updated only through retention gradients; the shared matrix $\mathbf{A}$ is updated only through high-entropy token gradients.

Training configuration: $\eta_B = 10^{-3}$, $\eta_A = 10^{-5}$ (the learning rate of A is much smaller than that of B), $\beta = \gamma = 1.0$, $\lambda = 0.01$, batch size = 4, epochs = 3.

## Key Experimental Results

| Dataset / Metric | Metric | ALTER (Ours) | Prev. SOTA | Gain / Advantage |
|---|---|---|---|---|
| WMDP-Bio (Llama3-8B) ↓ | Accuracy | 24.4% | 25.7% (AsymLoRA) | Closer to random chance (25%) |
| WMDP-Cyber (Llama3-8B) ↓ | Accuracy | 25.6% | 28.8% (AsymLoRA) | −3.2% |
| MMLU (Llama3-8B) ↑ | Accuracy | 57.8% | 57.2% (ELM) | +0.6% |
| Flu-mean (Llama3-8B) ↑ | Fluency Mean | 3.46 | 3.07 (ELM) | +0.39 |
| Flu-var (Llama3-8B) ↓ | Fluency Variance | 1.17 | 1.42 (LoRA/AsymLoRA) | −0.25 |
| WMDP-Bio (Zephyr-7B) ↓ | Accuracy | 24.4% | 27.1% (AsymLoRA) | −2.7% |
| MMLU (Zephyr-7B) ↑ | Accuracy | 56.4% | 57.8% (RMU/NPO_KL) | Slightly lower but more thorough forgetting |
| HarryPotter ASG ↓ | Similarity Gap | 1.3 | 1.9 (A-LoRA) | −0.6 |
| HarryPotter MMLU ↑ | Accuracy | 44.6% | 44.6% (ELM) | On par |
| HarryPotter Flu ↑ | Fluency | 3.3 | 3.1 (KL) | +0.2 |

Core advantage: Methods such as GA/RL can reduce WMDP accuracy to ~25%, but cause catastrophic MMLU degradation to 23–26%. ALTER achieves equivalent unlearning while retaining >90% model utility (compared to 47.8–83.6% for baselines).

### Ablation Study

- **Sequential Unlearning**: On TOFU, the forget set is progressively expanded from 1% to 10%. ALTER maintains stable performance near the original model level, while baselines exhibit progressive degradation (severe utility loss for GA/GD, moderate degradation for NPO).
- **Time Efficiency**: ALTER reduces training time by 86.1%–87.1% compared to non-AsymLoRA approaches. AsymLoRA serves as the unit-time baseline (1.0×), while ALTER requires only 1.25× to achieve further performance gains.
- **LoRA Rank**: All variants uniformly use rank = 8 to balance effectiveness and efficiency.

## Highlights & Insights

1. **Fine-grained token-level unlearning**: This work is the first to propose a knowledge localization method based on token entropy, distinguishing structural tokens (high-entropy) from knowledge-intensive tokens (low-entropy) to achieve surgical knowledge removal.
2. **Novel application of asymmetric architecture**: The A/B matrix separation property of HydraLoRA is ingeniously repurposed for unlearning, establishing dual parameter isolation between forgetting sub-tasks and between forgetting and retention.
3. **Tsallis entropy as a replacement for Shannon entropy**: Accounting for the non-extensive nature of LLMs, the deformation parameter $q$ enables differentiated treatment of high- and low-entropy tokens.
4. **High parameter efficiency**: The unlearning process is decoupled from the billions of LLM parameters, achieving SOTA results with only a small number of trainable parameters.
5. **Comprehensive validation across three benchmarks**: Superior performance is demonstrated on entity unlearning (TOFU), hazardous knowledge unlearning (WMDP), and copyright unlearning (MUSE).

## Limitations & Future Work

1. **Fixed entropy threshold**: The boundary between high and low entropy ($S_q = 1.2$) appears to be manually set; adaptive learning of this threshold warrants investigation.
2. **Determination of sub-domain count N**: The number of forgetting sub-domains must be predetermined via clustering; adaptability to unknown domain distributions remains to be explored.
3. **Evaluation limitations**: Fluency assessment relies on GPT-4o scoring, which does not fully align with human judgment.
4. **Insufficient robustness verification**: The paper does not discuss resistance to extraction attacks on the unlearned model (i.e., whether forgotten knowledge can be recovered after jailbreaking).
5. **Validation limited to 7B–8B models**: Performance and parameter isolation effectiveness on larger-scale models (70B+) remain unknown.

## Related Work & Insights

- **vs. gradient-based methods (GA/GD, etc.)**: These methods achieve unlearning through destructive parameter updates but cause catastrophic capability degradation; ALTER avoids this through parameter isolation.
- **vs. NPO family**: NPO employs blunt regularization, yielding inferior unlearning quality and utility retention compared to ALTER.
- **vs. RMU/ELM**: These methods achieve unlearning via steering vectors or concept erasure, reducing WMDP scores but producing poor fluency and introducing entanglement errors.
- **vs. standard LoRA**: Standard LoRA lacks sufficient rigidity in multi-domain coupled settings; ALTER overcomes this through its asymmetric architecture and entropy guidance.
- **vs. HydraLoRA**: ALTER innovatively transfers the asymmetric architecture of HydraLoRA from multi-task fine-tuning to the unlearning domain.

**Broader implications:**

1. **Generalizability of token-granularity knowledge management**: The separation of high- and low-entropy tokens is not limited to unlearning and could theoretically extend to knowledge editing and model compression.
2. **Tsallis entropy in NLP**: This provides NLP with a tool beyond Shannon entropy, particularly suited for sequential data with long-range dependencies.
3. **Relevance to safety alignment**: The unlearning framework can complement alignment techniques—first align, then unlearn residual harmful knowledge.
4. **Insights for LoRA architecture design**: The knowledge separation property of A/B matrices merits further exploration and exploitation in other settings.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Combining asymmetric LoRA with token entropy guidance for unlearning is a novel research direction, though individual components (AsymLoRA, entropy guidance, MoE routing) each have precursors.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three benchmarks, multiple backbone models, ablation studies, and efficiency analysis are fairly comprehensive, but robustness evaluation under adversarial attacks is absent.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clearly articulated and the logical chain from observations to method is complete, though some formulations are dense.
- **Value**: ⭐⭐⭐⭐ Practically significant for safe LLM deployment; parameter-efficient with strong results, offering a new paradigm for unlearning research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Modeling LLM Unlearning as an Asymmetric Two-Task Learning Problem](../../ACL2026/llm_safety/modeling_llm_unlearning_as_an_asymmetric_two-task_learning_problem.md)
- [\[ACL 2026\] Representation-Guided Parameter-Efficient LLM Unlearning](../../ACL2026/llm_safety/representation-guided_parameter-efficient_llm_unlearning.md)
- [\[ACL 2026\] Red-Bandit: Test-Time Adaptation for LLM Red-Teaming via Bandit-Guided LoRA Experts](../../ACL2026/llm_safety/red-bandit_test-time_adaptation_for_llm_red-teaming_via_bandit-guided_lora_exper.md)
- [\[ACL 2026\] Maximizing Local Entropy Where It Matters: Prefix-Aware Localized LLM Unlearning](../../ACL2026/llm_safety/maximizing_local_entropy_where_it_matters_prefix-aware_localized_llm_unlearning.md)
- [\[AAAI 2026\] WaterMod: Modular Token-Rank Partitioning for Probability-Balanced LLM Watermarking](watermod_modular_token-rank_partitioning_for_probability-balanced_llm_watermarki.md)

</div>

<!-- RELATED:END -->
