---
title: >-
  [Paper Note] COFT: Counterfactual-Conformal Decoding for Fair Chain-of-Thought Reasoning in Large Language Models
description: >-
  [ICML 2026][LLM Safety][Counterfactual Fairness] COFT achieves token-level counterfactual fairness guarantees on frozen LLMs in a training-free and gradient-free manner by constructing counterfactual masked branches duri…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "Counterfactual Fairness"
  - "Conformal Prediction"
  - "CoT Debiasing"
  - "Decoding-time Intervention"
  - "Training-free Debiasing"
date: 2026-05-08
content_hash: 9db02ee1e7a0db78
---

# COFT: Counterfactual-Conformal Decoding for Fair Chain-of-Thought Reasoning in Large Language Models

**Conference**: ICML 2026  
**arXiv**: [2605.30641](https://arxiv.org/abs/2605.30641)  
**Code**: None  
**Area**: LLM Safety/Fairness  
**Keywords**: Counterfactual Fairness, Conformal Prediction, CoT Debiasing, Decoding-time Intervention, Training-free Debiasing  

## TL;DR

COFT achieves token-level counterfactual fairness guarantees on frozen LLMs in a training-free and gradient-free manner by constructing counterfactual masked branches during decoding and fusing their logits, followed by filtering tokens through dual-branch split conformal prediction. This method reduces bias metrics by 30–55% (median 38%) with negligible loss in task performance.

## Background & Motivation

**Background**: Large Language Models (LLMs) expose and amplify social biases found in training data token-by-token during the Chain-of-Thought (CoT) generation process. Even if the final answer appears neutral, the reasoning trajectory may contain harmful stereotypical associations.

**Limitations of Prior Work**: Existing debiasing schemes have their respective limitations. Data cleaning and fine-tuning require retraining and may degrade general capabilities; auxiliary classifier-guided methods (e.g., DExperts, GeDi) depend on external models and inherit their blind spots; representation space debiasing (e.g., INLP) performs global linear projections, failing to adapt to specific prompt semantics and potentially removing legitimate content.

**Key Challenge**: The aforementioned methods fail to satisfy two critical properties simultaneously: (1) **Step-wise statistical guarantees**—the inability to ensure that a selected token remains stable after sensitive attribute replacement at each decoding step; (2) **Local counterfactual equivalence**—fairness goals are typically defined at the aggregate level rather than through per-token operations.

**Goal**: Design a decoding-time framework that achieves three properties: per-token counterfactual invariance, gradient-free/model-agnostic operation (suitable for frozen weights), and auditable step-wise marginal guarantees.

**Key Insight**: For each prompt, construct both factual and masked branches. Locate and eliminate biases driven by sensitive attributes by contrasting the differences in their logit distributions. Then, utilize the distribution-free guarantees of Conformal Prediction to filter unsafe tokens.

**Core Idea**: A three-stage pipeline consisting of counterfactual masking + logit convex interpolation fusion + dual-branch conformal filtering to jointly realize training-free per-token counterfactual fairness decoding.

## Method

### Overall Architecture

COFT is a pure inference-time framework acting on frozen causal language models. For a given prompt $p$, COFT executes three stages at each decoding step: (1) Replace sensitive segments in the prompt with neutral sentinel tokens to generate a masked prompt $\tilde{p} = M(p)$; (2) Perform forward inference on both the original and masked prompts separately, applying convex interpolation fusion to the two sets of logits to attenuate attribute-driven bias; (3) Filter tokens using dual-branch split conformal prediction thresholds calibrated offline, sampling only from the candidate set supported with high probability by both branches. The entire process requires only one additional cached forward pass and requires no training, gradients, or external classifiers.

### Key Designs

1. **Counterfactual Masking**:

    - **Function**: Generate a "de-sensitized" version of the original prompt that is structurally aligned to serve as the counterfactual branch.
    - **Mechanism**: Define a deterministic mask operator $M$ that replaces each sensitive segment $s \in S$ (e.g., gender or race identifiers) in the prompt with a neutral sentinel token `[MASK]`. A key design is **maintaining token count invariance**: if a sensitive segment is split into $k$ tokens by the tokenizer, it is replaced by $k$ copies of the sentinel. This ensures strict alignment of absolute positions between the two branches, making the bit-wise comparison of $z_t^F \leftrightarrow z_t^{CF}$ effective.
    - **Design Motivation**: Deleting sensitive segments disrupts grammar and attention geometry; replacing them with another identity injects new attributes; only masking preserves structure while severing direct lexical links to sensitive attributes.

2. **Counterfactual Logit Fusion**:

    - **Function**: Attenuate token probability biases driven by sensitive attributes in the logit space.
    - **Mechanism**: Define per-token attribute sensitivity as $\Delta_t = z_t^F - z_t^{CF}$, and generate fused logits $\hat{z}_t = (1-\lambda) z_t^F + \lambda z_t^{CF}$, where $\lambda \in [0,1]$ controls debiasing intensity. This is equivalent to $\hat{\pi}_t(v) \propto (\pi_t^F(v))^{1-\lambda} (\pi_t^{CF}(v))^{\lambda}$, which is a normalized geometric mixture of the two branch distributions. $\lambda$ is selected via the inflection point of a Pareto curve on the validation set (typically $\lambda \approx 0.6$).
    - **Design Motivation**: Performing fusion before filtering removes spurious amplification directions early, allowing subsequent conformal filtering to operate on aligned high-probability regions, which reduces false rejections and overly conservative thresholds.

3. **Dual-Branch Split-Conformal Filtering**:

    - **Function**: Construct a statistically certified candidate token set for each decoding step, providing distribution-free marginal coverage guarantees.
    - **Mechanism**: Define a dual-branch non-conformity score $s_t(v) = 1 - \min\{\hat{\pi}_t(v), \pi_t^{CF}(v)\}$. A token $v$ yields a small score only if it has sufficiently high probability in both the fused and masked distributions. Scores for all ground-truth next-tokens are calculated offline on a calibration set, and the $(1-\alpha)$ quantile $q_t$ is taken as the threshold. During online decoding, the candidate set is $C_t = \{v : \min\{\hat{\pi}_t(v), \pi_t^{CF}(v)\} \geq \tau_t\}$ (where $\tau_t = 1 - q_t$). Sampling is then performed from the conditional distribution of $\hat{\pi}_t$ restricted to $C_t$; if $C_t = \emptyset$, it falls back to $\arg\max$.
    - **Design Motivation**: Single-branch conformal prediction cannot guarantee counterfactual stability. The dual-branch approach requires tokens to be supported by "both worlds" simultaneously, directly operationalizing counterfactual equivalence.

## Key Experimental Results

### Main Results: Bias Metrics

| Dataset | Metric | Vanilla | SDD | DExperts | DT-CD | COFT | Gain (vs DT-CD) |
|--------|------|---------|-----|----------|-------|------|----------------|
| StereoSet (LLaMA-13B) | Bias↓ | 0.41 | 0.36 | 0.33 | 0.31 | **0.26** | -16% |
| CrowS-Pairs (LLaMA-13B) | Acc↑ | 58.7 | 60.1 | 61.0 | 61.3 | **63.5** | +2.2 |
| BBQ (LLaMA-13B) | Bias Rate↓ | 0.27 | 0.22 | 0.20 | 0.19 | **0.14** | -26% |
| BOLD (LLaMA-13B) | Toxicity↓ | 0.123 | 0.105 | 0.099 | 0.094 | **0.079** | -16% |
| Utrecht (LLaMA-13B) | DP Gap↓ | 0.184 | 0.153 | 0.149 | 0.141 | **0.118** | -16% |
| COMPAS (LLaMA-13B) | Bias Gap↓ | 0.161 | 0.147 | 0.141 | 0.136 | **0.119** | -12% |
| BBQ (Mistral-7B-Inst) | Bias Rate↓ | 0.24 | 0.20 | 0.18 | 0.17 | **0.12** | -29% |
| Utrecht (Mistral-7B-Inst) | DP Gap↓ | 0.173 | 0.146 | 0.141 | 0.136 | **0.112** | -18% |

### Ablation Study

| Configuration | BiasAvg↓ | UtilityAvg↑ | Description |
|------|----------|-------------|------|
| COFT (Full) | **0.129** | 68.0 | All three stages active |
| w/o Fusion (CP only) | 0.171 | 68.2 | Bias metrics increase by 32% without logit fusion |
| Single-branch CP (factual) | 0.158 | 68.1 | Cannot guarantee counterfactual stability |
| Fusion only (no CP) | 0.149 | 67.9 | Residual bias due to lack of statistical certification |

### Key Findings

- **Logit Fusion Contributes Most**: Removing fusion alone causes BiasAvg to rise from 0.129 to 0.171 (+33%), making it the most significant of the three components as it mechanically attenuates attribute-driven log-odds bias at the logit source.
- **Dual-Branch vs. Single-Branch CP**: Dual-branch CP reduces bias by an additional 18% compared to single-branch (0.158→0.129), validating the necessity of requiring tokens to be supported by high probabilities in both worlds.
- **Negligible Task Performance Loss**: COFT shows a gap of $\leq 0.2$ points compared to the Vanilla model on GSM8K, StrategyQA, ARC-easy, and PIQA, with almost no difference in PPL or MAUVE.
- **Controllable Efficiency Overhead**: Approximately 10.2% additional throughput overhead (equivalent to one cached forward pass), with peak VRAM increasing by $\leq 0.8$ GB.
- **Sensitivity of $\lambda$ and $\alpha$**: The bias-utility Pareto curve is stable within the $\lambda \in [0.4, 0.8]$ range, with the default set at the inflection point $\lambda \approx 0.6$; $\alpha = 0.10$ is the optimal risk level for conformal filtering.

## Highlights & Insights

- **Three-Stage Decoupled Design**: The pipeline of masking → fusion → filtering allows each component to be independently analyzed and replaced. Fusion first compresses the logit difference space before handing it to conformal filtering; their joint effect far exceeds separate use. This "denoise then certify" paradigm is transferable to any scenario requiring constraints during decoding.
- **Innovative Application of Conformal Prediction in Fairness**: Extends distribution-free statistical guarantees from traditional "confidence set" scenarios to "counterfactual stability certification." By designing a dual-branch score, fairness constraints are transformed into a standard quantile calibration problem, offering methodological generality.
- **Completely Training-Free Practical Advantage**: Requires only one additional cached forward pass ($\leq 11\%$ overhead). It is applicable to any frozen LLM checkpoint without requiring weight access, auxiliary classifiers, or fine-tuning, making it highly practical for API-only deployment scenarios.

## Limitations & Future Work

- **Sensitive Segment Detection Relies on External Tools**: COFT controls the use of identified sensitive segments during decoding but is not a universal implicit bias detector; unrecognized proxy terms may still escape.
- **Guarantees are Marginal, Not Conditional**: Conformal prediction provides marginal rather than input-conditional guarantees, which may fail under severe distribution shifts.
- **Sequence-Level Guarantees Require Additional Handling**: Current step-wise guarantees do not directly extend to the entire reasoning chain; joint upper bounds or rollout score calibration are needed for end-to-end control.
- **$\lambda$ Selection Requires a Validation Set**: A clean validation split is needed for Pareto inflection point selection, which may require re-tuning when deploying in new domains.

## Related Work & Insights

- **Counterfactual Fairness** (Kusner et al. 2017) provides the core theoretical framework, which COFT operationalizes as per-token local equivalence; **Conformal Prediction** (Vovk et al. 2005) provides distribution-free guarantee tools, which COFT innovatively adapts to the dual-branch scenario of autoregressive decoding.
- Compared to inference-time methods like DExperts/GeDi, COFT does not require external classifiers and provides statistical guarantees. Compared to representation debiasing methods like INLP, COFT is prompt-level adaptive rather than a fixed global projection.
- **Insight**: Similar "counterfactual masking + conformal filtering" paradigms can be extended to other trustworthy AI goals (e.g., privacy protection, factuality constraints) by defining different mask operators and non-conformity scores to implement various safety properties.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] BadThink: Triggered Overthinking Attacks on Chain-of-Thought Reasoning in Large Language Models](../../AAAI2026/llm_safety/badthink_triggered_overthinking_attacks_on_chain-of-thought_reasoning_in_large_l.md)
- [\[ICML 2026\] dgMARK: Decoding-Guided Watermarking for Diffusion Language Models](dgmark_decoding-guided_watermarking_for_diffusion_language_models.md)
- [\[ICML 2026\] Internalizing Safety Understanding in Large Reasoning Models via Verification](internalizing_safety_understanding_in_large_reasoning_models_via_verification.md)
- [\[ICML 2026\] Inducing Overthink: Hierarchical Genetic Algorithm-based DoS Attack on Black-Box Large Language Reasoning Models](inducing_overthink_hierarchical_genetic_algorithm-based_dos_attack_on_black-box_.md)
- [\[ACL 2026\] CiPO: Counterfactual Unlearning for Large Reasoning Models through Iterative Preference Optimization](../../ACL2026/llm_safety/cipo_counterfactual_unlearning_for_large_reasoning_models_through_iterative_pref.md)

</div>

<!-- RELATED:END -->
