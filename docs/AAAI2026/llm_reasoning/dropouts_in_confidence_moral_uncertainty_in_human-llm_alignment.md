---
title: >-
  [Paper Note] Dropouts in Confidence: Moral Uncertainty in Human-LLM Alignment
description: >-
  [AAAI 2026][LLM Reasoning][Moral Reasoning] This paper systematically investigates decision-making uncertainty across 32 open-source LLMs in moral dilemma scenarios (trolley problem variants)…
tags:
  - "AAAI 2026"
  - "LLM Reasoning"
  - "Moral Reasoning"
  - "LLM Alignment"
  - "Uncertainty"
  - "Dropout"
  - "Moral Machine"
date: 2026-05-08
content_hash: a39ba4223902fd61
---

# Dropouts in Confidence: Moral Uncertainty in Human-LLM Alignment

**Conference**: AAAI 2026
**arXiv**: [2511.13290](https://arxiv.org/abs/2511.13290)  
**Code**: [GitHub](https://github.com/jeakwon/MoralUncertainty)  
**Area**: LLM Reasoning
**Keywords**: Moral Reasoning, LLM Alignment, Uncertainty, Dropout, Moral Machine

## TL;DR
This paper systematically investigates decision-making uncertainty across 32 open-source LLMs in moral dilemma scenarios (trolley problem variants), finding that uncertainty is primarily driven by model architecture rather than moral dimension. Introducing attention dropout at inference time significantly increases mutual information and improves human-LLM moral alignment, suggesting that reducing overconfidence in moral scenarios can enhance consistency with human preferences.

## Background & Motivation

**Background**: LLMs are increasingly deployed in ethically sensitive decision-making contexts, yet while humans exhibit significant uncertainty and hesitation when facing moral dilemmas, LLMs tend to produce overconfident responses.

**Limitations of Prior Work**: LLM moral decisions are excessively decisive—high-confidence answers are generated even in ambiguous ethical scenarios. This overconfidence distorts alignment with human preferences and amplifies cognitive biases. Prior work has identified systematic AI preferences for inaction and stronger altruistic behavior.

**Core Problem**: (a) How can decision-making uncertainty in LLMs facing moral dilemmas be quantified? (b) Does this uncertainty stem from model differences or moral dimension differences? (c) Can regulating uncertainty improve human-LLM moral alignment?

**Key Insight**: Binary entropy is adopted as the mathematical measure of moral decision uncertainty, decomposed into three components—total entropy, conditional entropy, and mutual information—while inference-time dropout is used to artificially introduce stochasticity and observe its effect on alignment.

**Core Idea**: LLM overconfidence in moral dilemmas is a contributing cause of misalignment. By introducing dropout in attention layers to increase model "hesitation," the decision distribution can be made to more closely resemble human uncertainty patterns, thereby improving moral alignment.

## Method

### Overall Architecture
Building on the Moral Machine experimental framework (an autonomous driving variant of the classic trolley problem), binary decision probabilities are collected from 32 open-source LLMs across 9 moral dimensions (utilitarianism, age, gender, fitness, legality, etc.), followed by:
1. Uncertainty quantification via binary entropy decomposition
2. Inference-time dropout to regulate uncertainty
3. Measurement of changes in human-LLM alignment

### Key Designs

1. **Binary Decision & Uncertainty Quantification**:

    - Function: Moral dilemmas are cast as binary choices; the probability $p(c|x)$ is extracted from model logits.
    - Confidence is defined as $\Delta p^2 = (2p-1)^2$; uncertainty is measured by binary entropy $\mathbb{H}(p)$.
    - Total entropy is decomposed into three components: (a) Total Entropy TE = $\mathbb{H}(\mathbb{E}[p])$; (b) Conditional Entropy CE = $\mathbb{E}[\mathbb{H}(p)]$; (c) Mutual Information MI = TE − CE.
    - Design Motivation: TE reflects overall group-level decision uncertainty, CE reflects intrinsic per-scenario hesitation, and MI reflects the model's ability to discriminate across different scenarios.

2. **Inference-Time Attention Dropout**:

    - Function: Dropout (rate $r \in \{0.05, 0.1\}$) is applied after the softmax in the attention layer at inference time, injecting randomness into attention weights.
    - Formula: $\text{Attention}(Q,K,V) = \text{dropout}(\sigma(\frac{QK^T}{\sqrt{d_k}} + M), r) V$
    - Design Motivation: The randomness introduced by dropout simulates human uncertainty in moral dilemmas—making the model less "certain" of its own judgments and producing a more dispersed decision distribution closer to human behavior.

3. **Human-LLM Alignment Metric**:

    - Human preference vector $\vec{\delta}_h$ is derived from the original Moral Machine experiment data (AMCE analysis).
    - LLM preference vector $\vec{\delta}_m$ is aggregated from responses to 10K randomly sampled scenarios.
    - Alignment score = $L_2$ distance $\|\vec{\delta}_h - \vec{\delta}_m\|_2$; $\Delta L_2 < 0$ indicates improved alignment.

## Key Experimental Results

### Main Results: Sources of Uncertainty
- **Cross-model variance > cross-dimension variance**: Confidence differences across models within the same moral dimension far exceed differences across dimensions within the same model. This indicates that moral uncertainty is primarily determined by model architecture and training methodology rather than the nature of the moral question itself.
- Gemma-family models consistently exhibit high confidence; Llama-family models exhibit relatively lower confidence.

### Effect of Dropout on Uncertainty

| Dropout Rate | Total Entropy (TE) | Conditional Entropy (CE) | Mutual Information (MI) |
|---|---|---|---|
| 0.00 | Baseline | Baseline | Baseline |
| 0.05 | ↑ Significant (p<0.05) | ≈ Unchanged (ns) | ↑ Significant (p<0.05) |
| 0.10 | ↑↑ Significant (p<0.05) | ≈ Unchanged (ns) | ↑↑ Significant (p<0.05) |

### Effect of Dropout on Alignment (Selected Models, $\Delta L_2$)

| Model | Baseline $L_2$ | dropout=0.05 | dropout=0.10 |
|---|---|---|---|
| Llama-3.1-70B | 0.703 | 0.673 (−0.03) | **0.550 (−0.15)** |
| Llama-3.1-8B | 1.570 | 1.528 (−0.04) | 1.264 (−0.31) |
| Phi-4 | 0.989 | 0.946 (−0.04) | 0.790 (−0.20) |
| Qwen3-8B | 1.796 | 1.733 (−0.06) | **1.335 (−0.46)** |
| Qwen3-1.7B | 1.808 | 1.663 (−0.15) | 1.300 (−0.51) |

- $\Delta L_2 < 0$ indicates improved alignment (closer to human preferences); most models show significant alignment improvement after dropout.
- Qwen3-1.7B achieves the largest improvement (−0.51); Llama-3.1-70B achieves the best absolute alignment (0.550).

### Key Findings
- **Overconfidence is a contributing cause of misalignment**: High-confidence models (e.g., Gemma family) show larger gaps from human preferences across moral dimensions.
- **Uncertainty is primarily determined by model architecture**: Cross-model variance within the same moral dimension far exceeds cross-dimension variance within the same model.
- **Dropout increases "scenario sensitivity" (mutual information) rather than "intrinsic hesitation" (conditional entropy)**—models become more discriminative across scenarios rather than more confused.
- **The relationship between uncertainty and alignment is nonlinear**—some already well-aligned large models show slight alignment degradation after dropout.
- The Gemma family consistently exhibits high confidence while the Llama family exhibits relatively lower confidence; intra-family consistency exceeds cross-dimension variation.

## Highlights & Insights
- Applying information-theoretic tools (entropy decomposed into TE/CE/MI) to moral reasoning analysis constitutes a valuable methodological contribution. The finding that MI increases while CE remains unchanged is particularly noteworthy—dropout does not make models "more confused" but rather "more sensitive" to the specific details of moral scenarios, which more closely resembles human moral reasoning patterns.
- The counterintuitive finding that reducing confidence improves alignment carries significant practical implications, suggesting that future safety alignment methods may benefit from actively introducing uncertainty in morally sensitive scenarios rather than pursuing deterministic answers.

## Limitations & Future Work
- The study is confined to trolley problem variants from the Moral Machine, a highly simplified moral framework involving binary choices that cannot represent the more complex ethical situations encountered in practice.
- Dropout is a coarse mechanism for introducing uncertainty and may produce side effects on other model capabilities (e.g., reasoning accuracy), an impact that the paper does not evaluate.
- Human preference data originate from the 2018 Moral Machine experiment and may carry cultural and temporal biases.
- Only open-source models are evaluated; the behavior of closed-source models (e.g., GPT-4, Claude) remains unknown.

## Related Work & Insights
- **vs. Takemoto et al.'s Moral Machine LLM framework**: That work proposed alignment metrics; the present paper extends it by adding uncertainty analysis and dropout intervention, uncovering a causal relationship between uncertainty and alignment.
- **vs. Cheung et al. (omission bias study)**: Their finding that LLMs systematically prefer inaction may stem from fine-tuning practices; this paper offers a complementary explanation from an uncertainty perspective—overconfidence may obscure models' true preferences along the action-vs.-inaction dimension.
- **vs. traditional calibration methods (e.g., temperature scaling)**: Temperature scaling shifts the overall output distribution, whereas dropout operates at a finer granularity within the attention layers. Moreover, the observed effect manifests in MI rather than CE, indicating a distinct underlying mechanism.

## Rating
- Novelty: ⭐⭐⭐⭐ Applying information-theoretic uncertainty decomposition to moral reasoning alignment represents a novel perspective; the finding that dropout improves alignment is insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐ Large-scale analysis across 32 models × 9 dimensions with rigorous statistical testing.
- Writing Quality: ⭐⭐⭐⭐ The information-theoretic framework is clearly derived and supported by rich visualizations.
- Value: ⭐⭐⭐⭐ Offers a new direction for LLM moral alignment—actively introducing uncertainty rather than pursuing deterministic answers.
- Overall: A valuable exploration of moral alignment from an information-theoretic perspective; the finding that dropout improves alignment carries important practical implications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Chain-of-Thought as a Lens: Evaluating Structured Reasoning Alignment between Human Preferences and Large Language Models](../../ACL2026/llm_reasoning/chain-of-thought_as_a_lens_evaluating_structured_reasoning_alignment_between_hum.md)
- [\[ACL 2026\] Discovering a Shared Logical Subspace: Steering LLM Logical Reasoning via Alignment of Natural-Language and Symbolic Views](../../ACL2026/llm_reasoning/discovering_a_shared_logical_subspace_steering_llm_logical_reasoning_via_alignme.md)
- [\[ICLR 2026\] Annotation-Efficient Universal Honesty Alignment](../../ICLR2026/llm_reasoning/annotation-efficient_universal_honesty_alignment.md)
- [\[ACL 2026\] Taming Actor-Observer Asymmetry in Agents via Dialectical Alignment](../../ACL2026/llm_reasoning/taming_actor-observer_asymmetry_in_agents_via_dialectical_alignment.md)
- [\[AAAI 2026\] Graph of Verification: Structured Verification of LLM Reasoning with Directed Acyclic Graphs](graph_of_verification_structured_verification_of_llm_reasoning_with_directed_acy.md)

</div>

<!-- RELATED:END -->
