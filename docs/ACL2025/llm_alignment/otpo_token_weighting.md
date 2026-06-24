---
title: >-
  [Paper Note] Optimal Transport-Based Token Weighting for Enhanced Preference Optimization
description: >-
  [ACL 2025][LLM Alignment][DPO] OTPO utilizes Unbalanced Optimal Transport (UOT) to calculate semantic alignment weights between token representations of chosen and rejected responses, focusing preference optimization on critical distinguishing tokens instead of treating all tokens equally. It improves the LC WR of DPO from 48.14% to 55.84% on AlpacaEval 2.0, while unifying DPO, SimPO, SamPO, and LDDPO as special cases of token weighting.
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "DPO"
  - "optimal transport"
  - "token weighting"
  - "preference optimization"
  - "length bias"
date: 2026-05-08
content_hash: 182676136adfe618
---

# Optimal Transport-Based Token Weighting for Enhanced Preference Optimization

**Conference**: ACL 2025  
**arXiv**: [2505.18720](https://arxiv.org/abs/2505.18720)  
**Code**: [https://github.com/Mimasss2/OTPO](https://github.com/Mimasss2/OTPO)  
**Area**: LLM Alignment / Preference Optimization  
**Keywords**: DPO, optimal transport, token weighting, preference optimization, length bias

## TL;DR

OTPO utilizes Unbalanced Optimal Transport (UOT) to calculate semantic alignment weights between token representations of chosen and rejected responses, focusing preference optimization on critical distinguishing tokens instead of treating all tokens equally. It improves the LC WR of DPO from 48.14% to 55.84% on AlpacaEval 2.0, while unifying DPO, SimPO, SamPO, and LDDPO as special cases of token weighting.

## Background & Motivation

**Token Equality Issue in DPO**: Standard DPO assigns equal weights to all tokens in a sequence when calculating log probabilities. However, humans focus more on critical semantic components (such as argument quality and informational accuracy) when evaluating preferences, rather than filler words or formatting tokens.

**Length Bias as a Byproduct of Equal Weighting**: Longer responses have more tokens contributing to the loss, leading to a systematic bias in DPO toward choosing longer responses—even when the content quality is equivalent. This has been confirmed by several studies (e.g., SimPO, SamPO).

**Lack of a Unified Framework in Extant Corrections**: Prior methods provide heuristic solutions—such as length normalization in SimPO, sampling probability weighting in SamPO, and length-difference correction in LDDPO—but lack a unified theoretical perspective and do not exploit token-level semantic information.

**Absence of Semantic Alignment**: Although semantic correspondences exist between tokens of chosen and rejected responses (e.g., different paragraphs addressing the same prompt), existing methods completely ignore this cross-sequence token-level semantic structure.

**Natural Fit of Optimal Transport**: Optimal Transport (OT) is a mathematical framework designed to solve optimal matching between two distributions, making it a natural tool to establish semantic correspondences between chosen and rejected tokens and subsequently derive weights.

**Core Innovation**: This work uses the marginal distributions of the transport plan from UOT as token weights. Tokens semantically matched to important tokens in the opposing sequence receive high weights, whereas irrelevant tokens receive low weights.

## Method

### Overall Architecture

Based on standard DPO, a token weight computation module is introduced: (1) extracting semantic representations for each token of the chosen and rejected sequences using the hidden representations of the model's final layer; (2) computing the Euclidean distance between token pairs to construct a cost matrix $C$; (3) solving the UOT problem to obtain the transport plan; (4) deriving weights for the chosen/rejected tokens from the row/column marginal distributions of the transport plan; (5) replacing the uniform log-likelihood in DPO with the weighted log-likelihood.

### Key Designs

**1. Cost Matrix Construction**

- **Function**: Calculate semantic distances for each token pair between the chosen ($m$ tokens) and rejected ($n$ tokens) sequences.
- **Mechanism**: Extract the final-layer hidden states and compute the Euclidean distances to obtain an $m \times n$ cost matrix.
- **Design Motivation**: The final-layer representations contain the richest semantic information; Euclidean distance is simple, efficient, and offers high discriminative power in high-dimensional spaces.

**2. Unbalanced Optimal Transport Solving**

- **Function**: Find the optimal semantic matching scheme between the chosen and rejected tokens.
- **Mechanism**: Solve the transport problem configured with entropy-regularized and KL marginal relaxation, allowing marginal distributions to deviate from uniform distributions.
- **Design Motivation**: UOT is used instead of standard OT because chosen and rejected sequences usually differ in length; UOT allows partial transport (without requiring total mass conservation), naturally handling length discrepancies.

**3. Token Weight Derivation**

- **Function**: Extract the importance weight of each token from the transport plan.
- **Mechanism**: The weight of chosen token $i$ is calculated as the marginal sum of the $i$-th row of the transport plan, and similarly for the rejected tokens. These weights are normalized and then applied to the DPO log probability calculations.
- **Design Motivation**: Tokens in the transport plan that are heavily "transported" to opposing tokens are semantically most relevant to preference differences, representing critical distinguishing points between the chosen and rejected sequences.

**4. Unified Framework Interpretation**

- **Function**: Demonstrate that DPO, SimPO, SamPO, and LDDPO are special cases of OTPO under specific weight degenerations.
- **Mechanism**: DPO corresponds to a uniform weight ($1/m$); SimPO to length-normalized weight; SamPO to sampling-probability-based weight; LDDPO to length-difference-modulated weight. OTPO represents the most generalized context-aware weighting.
- **Design Motivation**: A unified perspective not only enhances theoretical understanding but also demonstrates that the heuristic corrections of prior methods can be "automatically discovered" through the optimal solution of OT.

### Loss & Training

Weighted DPO loss, where weights are dynamically computed at each step by the UOT solver. The UOT module does not propagate gradients during training (the weights are treated as constants).

## Key Experimental Results

### Main Results

| Model | Method | AlpacaEval2 LC WR | MT-Bench |
|------|------|-------------------|----------|
| Llama-3-8B | DPO | 48.14% | 7.65 |
| Llama-3-8B | SimPO | 52.67% | 7.72 |
| Llama-3-8B | SamPO | 51.43% | 7.68 |
| Llama-3-8B | LDDPO | 53.21% | 7.70 |
| Llama-3-8B | **OTPO** | **55.84%** | **7.81** |
| Gain | OTPO vs DPO | **+7.70%** | **+0.16** |

### Ablation Study

| Ablation Item | AlpacaEval2 LC WR | Impact |
|--------|-------------------|------|
| UOT $\rightarrow$ Standard OT | 53.12% | Strict mass conservation impairs adaptability to length discrepancies |
| Euclidean distance $\rightarrow$ Cosine distance | 54.91% | Comparable but slightly lower performance |
| Final layer $\rightarrow$ Intermediate layer | 53.78% | Insufficient semantic discriminative power in shallow representations |
| Uniform weight (=DPO) | 48.14% | Degenerates to the baseline |
| Fixed weights (no updates) | 52.30% | Clear advantage of dynamic weights |

### Key Findings

- OTPO improves LC WR by 7.7% over DPO on AlpacaEval 2.0, outperforming the best heuristic method, LDDPO, by 2.6%.
- UOT is better suited for preference optimization than standard OT—partial transport between sequences of unequal lengths is physically more reasonable than forced full matching.
- Token weight visualization reveals that OTPO automatically learns to focus on tokens containing key arguments and factual information, while ignoring conjunctions and formatting tokens.
- The weight distribution stabilizes as training progresses, indicating that OT uncovers stable semantic structures.
- The framework is orthogonal to base preference optimization methods and theoretically can be integrated with other methods such as KTO and IPO.

## Highlights & Insights

- **Theoretical Elegance of Optimal Transport and Preference Optimization**: OT naturally models the optimal matching between two discrete distributions, corresponding to the semantic alignment between chosen and rejected tokens. This is not an ad-hoc combination but a perfect alignment of problem structure and mathematical tools.
- **Explanatory Power of the Unified Framework**: Unifying several heuristic corrections as special cases of token weights offers both theoretical insights and practical guidance. Future preference optimization methods can be directly explored within this weight design space.
- **Plug-and-Play**: OTPO only requires integrating a UOT solving step into the DPO training loop, maintaining a manageable computational overhead.

## Limitations & Future Work

- The UOT solver (Sinkhorn iterations) increases the computational cost per training step, which could become a bottleneck for extremely long sequences.
- The cost matrix is based on the hidden states of the active model; when the representation quality is low in the early stages of training, the weights may be inaccurate.
- Validation is restricted to the UltraFeedback dataset, offering limited data diversity.
- Integration with RLHF/PPO remains unexplored; it is uncertain whether token weighting can also enhance online preference optimization.
- The interpretability analysis of weight visualization is not yet deep, lacking quantitative evaluations of semantic alignment quality.

## Related Work & Insights

- **vs SimPO**: SimPO utilizes length normalization, which is a special case of OTPO weights and fails to capture token-level semantic variations.
- **vs SamPO**: SamPO utilizes sampling probability weighting, which considers the generation difficulty of tokens but ignores cross-sequence semantic correspondences.
- **vs LDDPO**: LDDPO explicitly models length differences; however, it remains a sequence-level adjustment rather than token-level.
- **vs TDPO**: TDPO also implements token-level DPO but defines weights heuristically using rules, whereas OTPO automatically discovers optimal weights using OT.
- **Insights**: Applications of OT in NLP are increasingly prominent (e.g., document matching, cross-lingual alignment), and preference optimization represents a new, successful application scenario.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Theoretical elegance and unified perspective of using OT for token weighting.
- Experimental Thoroughness: ⭐⭐⭐⭐ Significant performance gains and thorough ablations.
- Writing Quality: ⭐⭐⭐⭐ Tight integration of theoretical derivation and empirical results.
- Value: ⭐⭐⭐⭐⭐ Unified framework + strong empirical results + plug-and-play capability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Probability-Consistent Preference Optimization for Enhanced LLM Reasoning](probability-consistent_preference_optimization_for_enhanced_llm_reasoning.md)
- [\[ACL 2025\] T-REG: Preference Optimization with Token-Level Reward Regularization](t-reg_preference_optimization_with_token-level_reward_regularization.md)
- [\[ACL 2025\] Reverse Preference Optimization for Complex Instruction Following](reverse_preference_optimization_for_complex_instruction_following.md)
- [\[ICML 2025\] TGDPO: Harnessing Token-Level Reward Guidance for Enhancing Direct Preference Optimization](../../ICML2025/llm_alignment/tgdpo_harnessing_token-level_reward_guidance_for_enhancing_direct_preference_opt.md)
- [\[ACL 2025\] SDPO: Segment-Level Direct Preference Optimization for Social Agents](sdpo_segment-level_direct_preference_optimization_for_social_agents.md)

</div>

<!-- RELATED:END -->
