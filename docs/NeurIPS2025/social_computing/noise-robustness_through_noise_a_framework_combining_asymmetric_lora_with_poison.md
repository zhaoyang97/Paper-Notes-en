---
title: >-
  [Paper Note] Noise-Robustness Through Noise: A Framework Combining Asymmetric LoRA with Poisoning MoE
description: >-
  [NeurIPS 2025][Social Computing][LoRA] This paper proposes LoPE, which designates a dedicated "poisoning expert" within an asymmetric LoRA architecture to absorb injected noise during training; at inference time…
tags:
  - "NeurIPS 2025"
  - "Social Computing"
  - "LoRA"
  - "Noise Robustness"
  - "Mixture-of-Experts"
  - "Parameter-Efficient Fine-Tuning"
  - "Data Denoising"
date: 2026-05-08
content_hash: a5663ba885ba4a8d
---

# Noise-Robustness Through Noise: A Framework Combining Asymmetric LoRA with Poisoning MoE

**Conference**: NeurIPS 2025
**arXiv**: [2505.23868](https://arxiv.org/abs/2505.23868)  
**Code**: None  
**Area**: Social Computing
**Keywords**: LoRA, Noise Robustness, Mixture-of-Experts, Parameter-Efficient Fine-Tuning, Data Denoising

## TL;DR

This paper proposes LoPE, which designates a dedicated "poisoning expert" within an asymmetric LoRA architecture to absorb injected noise during training; at inference time, this expert is masked so that only the clean experts contribute to the output — achieving noise robustness through noise itself, entirely without data cleaning.

## Background & Motivation

Parameter-efficient fine-tuning (PEFT) methods are susceptible to noisy data when adapting pre-trained language models to downstream tasks. Existing denoising approaches fall into two categories: (1) pre-training data cleaning (which relies on human intervention or prior assumptions, incurring high cost and being limited to specific noise types); and (2) architectural modifications during training for denoising (which avoids explicit data cleaning but still requires noise discrimination and is prone to error accumulation).

The core insight is that **noise injection** is far less costly and more easily automated than noise identification and handling. However, naively adding noise to training data seems counterproductive — clean and noisy samples would affect all model parameters simultaneously, making it impossible to exploit noise patterns effectively. The key observation is: if noise-injection-related patterns can be **isolated into a dedicated module** that is subsequently **masked at inference time**, the robustness benefits of noisy data can be retained without the associated negative effects. The asymmetric LoRA + MoE architecture naturally supports this kind of functional specialization.

## Method

### Overall Architecture

LoPE builds upon an asymmetric LoRA architecture (a shared matrix $A$ combined with multiple independent matrices $B_i$), designating one $B$ matrix as the "poisoning expert" $B_D$. The framework proceeds in three steps: Stage I trains the poisoning expert on mixed noise to learn noise patterns; Stage II freezes the poisoning expert and fine-tunes the clean experts on clean knowledge; and at inference time, the poisoning expert is masked so that only the clean experts contribute to the output.

### Key Designs

1. **HyNoIse (Hybrid Noise Injection)**: Combines discrete noise (character-level: word order shuffling, noisy character insertion, character deletion) and continuous noise (embedding-level: additive uniform noise $N \sim \mathcal{U}(-1,1)$ applied at valid token positions, controlled by noise ratio $\alpha$). The two-level noise covers multi-dimensional perturbations including character, token, label, and structural anomalies, injected in equal proportions for uniform coverage across noise types.

2. **Two-Stage Fine-Tuning**:

    - **Stage I (Poisoning Expert Specialization)**: The clean experts $B_i$ are frozen; only the shared matrix $A$ and the poisoning expert $B_D$ are trained using HyNoIse-augmented data. $A$ learns general knowledge while $B_D$ specializes in noise-handling patterns.
    - **Stage II (Dynamic Compensation Expert Collaboration)**: $B_D$ parameters are frozen (but still participate in the forward pass); $A$ and the clean experts $B_i$ are fine-tuned on original (noise-free) data. A DyCompEnSate mechanism is introduced: an expert dependency matrix $\Theta$ is constructed by computing the cosine similarity $\theta_{iD}$ between each clean expert's output and the poisoning expert's output. After masking the poisoning expert, dynamic compensation weights $(1 + \theta_{iD})$ are applied to bridge the dependency gap, with a normalization factor $\beta$ to maintain output stability.

3. **Inference-Time Masking Strategy**: The poisoning expert $B_D$ is directly masked, excluding its noise-contaminated knowledge. A gating router dynamically assigns weights to the clean experts; their weighted average is combined with the shared $A$ to complete the low-rank transformation. The DyCompEnSate compensation weights ensure that the output distribution after masking remains consistent with that observed during training.

### Loss & Training

- Base model: LLaMA2-7B
- Fine-tuning data: 10% of Alpaca-52K
- Noise settings: Orig (original data) and Nois (5% discrete noise injection)
- HyNoIse ratio $\alpha = 5\%$; default configuration of 3 clean experts + 1 poisoning expert; rank = 4
- Evaluation benchmarks: MMLU (57 subtasks), GSM8K, PIQA, SIQA, ARC-e

## Key Experimental Results

### Main Results

| Method | MMLU | PIQA | SIQA | GSM8K | ARC-e | Note |
|--------|------|------|------|-------|-------|------|
| HydraLoRA(r=4)† | 43.08 | 74.92 | 47.29 | 11.83 | 55.80 | Baseline asymmetric LoRA |
| LoPE(r=4)† | **44.42** | **76.28** | **49.03** | **13.72** | **56.84** | Noise robust |
| LoRA(r=4)† | 40.45 | 71.45 | 43.17 | 11.02 | 49.03 | Standard LoRA |
| LoPE(r=8)† | **44.82** | **76.83** | **49.90** | **14.31** | **58.02** | Higher rank |

On noisy data, LoPE(r=4) achieves an average improvement of 1.34% on MMLU and 4.89% on ARC-e over HydraLoRA.

### Ablation Study

| Configuration | Avg. PIQA+SIQA (5% noise) | Note |
|---------------|--------------------------|------|
| No noise injection | 59.89 | LoPE without HyNoIse |
| Continuous noise only | 60.71 | Embedding-level noise |
| Discrete noise only | 61.95 | Character-level noise |
| Mixed noise (HyNoIse) | **62.66** | Complementary combination |

Cross-noise-type experiments using NCI noise in the training set and WOS noise in HyNoIse show that LoPE still outperforms HydraLoRA, demonstrating that robustness does not depend on consistency between noise types.

### Key Findings

- **Clear advantage under noisy conditions**: The performance gap between LoPE and HydraLoRA is small on Orig data (where noise is minimal), but LoPE consistently outperforms on Nois data.
- **Discrete noise outperforms continuous noise**: Discrete noise directly manipulates natural language text and aligns more easily with the semantic space in Stage II.
- **Stability across noise levels**: LoPE maintains relatively stable performance at noise levels of 3.5%, 5%, and 8%, while conventional methods degrade significantly.
- **Necessity of DyCompEnSate**: Directly masking the poisoning expert disrupts inter-expert dependencies learned during training; the compensation mechanism is indispensable.
- Time complexity remains $O(n^2)$, consistent with other PEFT methods, introducing no additional computational overhead.

## Highlights & Insights

- **Counter-intuitive "fight noise with noise" paradigm**: In direct contrast to conventional denoising approaches (cleaning/identifying noise), LoPE actively injects noise to enhance robustness — a conceptually bold and novel contribution.
- **Functional specialization in asymmetric LoRA**: The approach cleverly leverages the design properties of shared $A$ (general knowledge) and independent $B_i$ (differentiated knowledge) to naturally embed the noise-absorbing module within an existing architecture.
- **Entirely data-cleaning-free**: This lowers the barrier to practical deployment, as real-world data is pervasively noisy yet cleaning is costly.
- The expert dependency compensation idea in DyCompEnSate is applicable to any scenario requiring dynamic activation or deactivation of MoE experts.

## Limitations & Future Work

- The number of poisoning experts is fixed at one, which may be insufficient under extremely high noise levels or when multiple noise types coexist.
- Validation is limited to LLaMA2-7B; generalization to larger models or other LLM architectures has not been explored.
- Domain discrepancies between fine-tuning data (Alpaca) and evaluation benchmarks (MMLU, GSM8K, etc.) may introduce confounding variables.
- No direct comparison is made with dedicated denoising methods (e.g., MICL, LeCoRE, LLMClean) under the same experimental settings.
- The uniform distribution for continuous noise may not be optimal; Gaussian or structured noise distributions warrant further exploration.

## Related Work & Insights

- The asymmetric LoRA design draws from HydraLoRA (Tian et al.); LoPE further exploits its functional differentiation among experts.
- The "poisoning" concept has potential connections to adversarial training and model immunization, though the implementation is fundamentally different (isolation rather than adversarial optimization).
- The work offers insights for noisy label learning (NLL) and data selection/curriculum learning: rather than cleaning noise, models can be designed to isolate it internally.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The "fight noise with noise" paradigm is highly innovative; the poisoning expert design is elegant
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-task, multi-configuration validation with thorough ablation, but lacks direct comparison with dedicated denoising methods
- **Writing Quality**: ⭐⭐⭐⭐ Pipeline diagrams are clear; the three-stage workflow is described in an organized manner
- **Value**: ⭐⭐⭐⭐ Introduces a new paradigm for noise robustness in PEFT with strong practical applicability

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Auto-Search and Refinement: An Automated Framework for Gender Bias Mitigation in LLMs](auto-search_and_refinement_an_automated_framework_for_gender_bias_mitigation_in_.md)
- [\[AAAI 2026\] Fact2Fiction: Targeted Poisoning Attack to Agentic Fact-checking System](../../AAAI2026/social_computing/fact2fiction_targeted_poisoning_attack_to_agentic_fact-check.md)
- [\[ACL 2026\] Investigating Counterfactual Unfairness in LLMs towards Identities through Humor](../../ACL2026/social_computing/investigating_counterfactual_unfairness_in_llms_towards_identities_through_humor.md)
- [\[ICLR 2026\] GRADIEND: Feature Learning within Neural Networks Exemplified through Biases](../../ICLR2026/social_computing/gradiend_feature_learning_within_neural_networks_exemplified_through_biases.md)
- [\[AAAI 2026\] Bias Association Discovery Framework for Open-Ended LLM Generations](../../AAAI2026/social_computing/bias_association_discovery_framework_for_open-ended_llm_generations.md)

</div>

<!-- RELATED:END -->
