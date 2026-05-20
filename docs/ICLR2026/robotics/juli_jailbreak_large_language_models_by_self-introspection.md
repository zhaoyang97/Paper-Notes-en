---
title: >-
  [Paper Note] JULI: Jailbreak Large Language Models by Self-Introspection
description: >-
  [ICLR 2026][Robotics][jailbreak] This paper reveals that top-k token log probabilities returned by aligned LLM APIs still contain harmful knowledge leakage…
tags:
  - "ICLR 2026"
  - "Robotics"
  - "jailbreak"
  - "logit bias"
  - "API attack"
  - "token log probability"
  - "BiasNet"
date: 2026-05-08
content_hash: 5488ebf755c85f8e
---

# JULI: Jailbreak Large Language Models by Self-Introspection

**Conference**: ICLR 2026
**arXiv**: [2505.11790](https://arxiv.org/abs/2505.11790)  
**Code**: None  
**Area**: Robotics
**Keywords**: jailbreak, logit bias, API attack, token log probability, BiasNet

## TL;DR
This paper reveals that top-k token log probabilities returned by aligned LLM APIs still contain harmful knowledge leakage, and proposes JULI—a BiasNet plugin with fewer than 1% of the target model's parameters—that manipulates logit bias to successfully jailbreak Gemini-2.5-Pro (Harmful Info Score 4.19/5) under API settings restricted to top-5 token probabilities, achieving approximately 140× speedup over LINT while doubling harmfulness scores.

## Background & Motivation
**Background**: LLM jailbreak attacks are broadly categorized into white-box attacks requiring model weights and black-box attacks operating solely through APIs. API-based attacks are particularly challenging due to the absence of gradient access, full logits, and control over the generation process.

**Limitations of Prior Work**: (a) White-box methods such as GCG require full gradient access and are inapplicable to commercial APIs; (b) LINT, the current API-based attack, requires access to top-500 tokens (whereas most APIs expose only top-5/20), incurs 99.7 seconds of inference time, and achieves a harmfulness score of only 2.25/5; (c) Weak-to-Strong and Emulated Disalignment require access to model weights from both pre- and post-alignment versions.

**Key Challenge**: Although alignment training is intended to suppress the expression of harmful knowledge, it remains an open question whether top-k token probabilities returned by LLM APIs still leak harmful information.

**Goal**: Can mainstream commercial LLMs be efficiently jailbroken using only a small number of token probabilities (e.g., top-5) available through standard APIs?

**Key Insight**: Statistical analysis reveals that >85% of harmful response tokens appear within the top-5 probability candidates—indicating that alignment suppresses their sampling probability rather than erasing the underlying knowledge.

**Core Idea**: A lightweight BiasNet (<1% of the target model's parameters) is trained on only 100 harmful examples to learn logit biases that elevate the sampling probability of harmful tokens.

## Method

### Overall Architecture
BiasNet $F_\theta$ takes the log probability output $\log p_\alpha(x_n)$ from the target LLM and computes a logit bias $B = F_\theta(\log p_\alpha(x_n))$. The corrected probability is then given by $\tilde{p}_\alpha(x_n) = p_\alpha(x_n) + B$.

### Key Designs

1. **Token Leakage Discovery**: Statistical analysis across multiple aligned LLMs shows that >85% of harmful response tokens appear within the top-5 predicted probabilities, demonstrating that alignment training does not erase harmful knowledge but merely reduces its probability rank. The core insight is that alignment performs *probability suppression* rather than *knowledge erasure*.

2. **BiasNet Architecture**: With fewer than 1% of the target model's parameters (~$10^7$), BiasNet consists of three components:

    - Projection Layer 1: token space → hidden space (white-box: pseudo-inverse of the LLM head; black-box: random orthogonal matrix)
    - Intermediate transformation layer (learnable)
    - Projection Layer 2: hidden space → token space (white-box: LLM head; black-box: random orthogonal matrix)
    - Output logit bias $B = F_\theta(\log p_\alpha(x_n))$, corrected log-probability $\log \tilde{p} = \log p + B$

3. **Padding Mechanism (API Setting)**: When the API returns only top-k token probabilities, non-top-k tokens are assigned a padding value (the probability of the $k$-th token minus a fixed offset of 10), enabling BiasNet to operate on incomplete probability vectors.

4. **Training**: Only 100 harmful question–answer pairs from LLM-LAT are used, trained for 15 epochs with batch size 1 and AdamW at lr=$10^{-5}$, at extremely low cost.

## Key Experimental Results

### Open-Source Model Attacks (White-Box Setting, AdvBench)

| Target Model | JULI Harmful Score | Best Baseline | Baseline Method | JULI Inference Time |
|---|---|---|---|---|
| Llama3-8B-Instruct | **3.44** | 3.02 | ED | 0.71s |
| Llama2-7B-Chat | **3.38** | 2.89 | ED | 0.71s |
| Llama3-3B-Instruct | **3.52** | 3.15 | ED | 0.65s |
| Qwen2-1.5B-Instruct | **3.61** | 3.28 | ED | 0.58s |
| Llama3-8B-CB (Circuit Breaker defense) | **2.95** | 1.85 | GCG | 0.71s |
| Llama2-7B-DeepAlign (DeepAlign defense) | **3.21** | 2.45 | GCG | 0.71s |

### API Attacks (Black-Box Setting, top-5 API)

| Target Model | JULI Harmful Info Score | FLIP | Naive | Base |
|---|---|---|---|---|
| Gemini-2.5-Pro | **4.19** | 2.09 | 1.21 | 0.06 |
| Gemini-2.5-Flash | **1.74** | 1.33 | 1.29 | 0.02 |

### Key Findings
- Top-5 token probabilities from aligned LLMs are sufficient to recover harmful outputs—alignment suppresses probabilities rather than erasing knowledge.
- A plugin with fewer than 1% of the target model's parameters, trained on only 100 examples, suffices to break state-of-the-art defenses (Circuit Breaker and DeepAlign).
- JULI is 140× faster than LINT (0.71s vs. 99.7s) and achieves approximately 2× higher harmfulness (3.44 vs. 2.25).
- The newly proposed Harmful Info Score correlates more strongly with human judgments than conventional BERT Score and Harmful Score metrics.

## Highlights & Insights
- **The "knowledge leakage" vs. "knowledge erasure" distinction**: Consistent with the "shallow alignment" finding in *Erase or Hide*, harmful knowledge remains fully intact within aligned models and is only probabilistically suppressed. JULI demonstrates that this suppression can be trivially reversed by an external plugin, which has fundamental implications for alignment research—suggesting that all current RLHF/DPO-based safety training amounts to a superficial fix.
- **A red flag for API security**: Real-world LLM APIs (e.g., the Gemini API) that return top-k probabilities inadvertently expose an attack surface. API designers must reassess the security risks of exposing log probabilities.
- **Methodological contribution of Harmful Info Score**: The newly proposed evaluation metric focuses on the informativeness and quality of harmful responses rather than surface-level "harmfulness" labels, achieving higher correlation with human judgments and serving as a more rigorous standard for jailbreak evaluation.

## Ablation Study

| Analysis Dimension | Finding |
|---|---|
| Top-k hit rate | >85% of harmful tokens appear in top-5; >95% in top-10 |
| Training data size | Saturation is reached with only 100 samples; additional data yields negligible gains |
| Projection layer design | White-box reuse of the LLM head outperforms random initialization, yet black-box random orthogonal matrices also function effectively |
| Robustness against defenses | Successfully breaks both Circuit Breaker and DeepAlign |
| Hard subset | Remains effective on the hardest 5% of AdvBench, where baseline methods nearly fail |
| Harmful Info Score | The proposed metric correlates more strongly with human judgments than BERT Score and Harmful Score |

## Limitations & Future Work
- BiasNet requires a small amount of harmful training data (100 examples), precluding fully zero-knowledge attacks.
- Defensive countermeasures are not thoroughly discussed—restricting the number of returned tokens or injecting noise into probabilities are obvious mitigations.
- API providers could defend against JULI by not returning log probabilities, at the cost of sacrificing legitimate use cases.
- Black-box attacks have been validated only on the Gemini API; further testing on the OpenAI API is needed.
- The padding mechanism in BiasNet may introduce bias under certain token distributions.

## Related Work & Insights
- **vs. GCG (Zou et al.)**: GCG requires full gradient access (white-box), whereas JULI operates with only top-5 probabilities—a qualitative gap in practical applicability.
- **vs. Weak-to-Strong (Zhao et al.)**: WTS requires pre-trained base model weights; JULI operates entirely through the API.
- **vs. LINT (Zhang et al.)**: LINT requires top-500 tokens and 99.7 seconds of inference; JULI requires only top-5 and 0.71 seconds—a 140× speedup.
- **vs. Emulated Disalignment**: ED requires weights from both pre- and post-alignment model versions; JULI is fully black-box.
- **Insight—alignment as probability suppression rather than knowledge erasure**: The most profound contribution is demonstrating that harmful knowledge remains fully intact after alignment—the top-5 distribution alone contains sufficient information to recover complete harmful outputs.

## Rating
- Novelty: ⭐⭐⭐⭐ First practical jailbreak using only top-5 API probabilities; BiasNet is a conceptually novel design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple models (including closed-source) × multiple settings × multiple evaluation metrics × state-of-the-art defenses.
- Writing Quality: ⭐⭐⭐⭐ Clear presentation; Harmful Info Score constitutes a methodological contribution.
- Value: ⭐⭐⭐⭐ Direct warning for API security design—whether log probabilities should be exposed warrants re-evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Sysformer: Safeguarding Frozen Large Language Models with Adaptive System Prompts](sysformer_safeguarding_frozen_large_language_models_with_adaptive_system_prompts.md)
- [\[ACL 2026\] Reasoning Hijacking: The Fragility of Reasoning Alignment in Large Language Models](../../ACL2026/robotics/reasoning_hijacking_the_fragility_of_reasoning_alignment_in_large_language_model.md)
- [\[ACL 2026\] GRASPrune: Global Gating for Budgeted Structured Pruning of Large Language Models](../../ACL2026/robotics/grasprune_global_gating_for_budgeted_structured_pruning_of_large_language_models.md)
- [\[NeurIPS 2025\] Uncovering Strategic Egoism Behaviors in Large Language Models](../../NeurIPS2025/robotics/uncovering_strategic_egoism_behaviors_in_large_language_models.md)
- [\[ACL 2026\] DeCoVec: Building Decoding Space based Task Vector for Large Language Models via In-Context Learning](../../ACL2026/robotics/decovec_building_decoding_space_based_task_vector_for_large_language_models_via_.md)

</div>

<!-- RELATED:END -->
