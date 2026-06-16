---
title: >-
  [Paper Note] Conf-Gen: Conformal Uncertainty Quantification for Generative Models
description: >-
  [ICML 2026][Image Generation][Generative Model] The Conf-Gen framework is proposed to extend Conformal Risk Control (CRC) to generative tasks. Using parameterized selection functions and admissibility functions, it provides formal uncertainty guarantees for tasks such as LLM QA, image generation, dialogue systems, and AI agents, while relaxing theoretical assumption
tags:
  - ICML 2026
  - Image Generation
  - Generative Model
date: 2026-05-08
content_hash: dc1b8594cc98756c
---
# Conf-Gen: Conformal Uncertainty Quantification for Generative Models

**Conference**: ICML 2026  
**arXiv**: [2605.28920](https://arxiv.org/abs/2605.28920)  
**Code**: https://github.com/layer6ai-labs/conf-gen (Available)  
**Area**: Uncertainty Quantification / Generative Models  
**Keywords**: Conformal Prediction, Uncertainty Quantification, Generative Models, Conformal Risk Control, Selection Functions  

## TL;DR
The Conf-Gen framework is proposed to extend Conformal Risk Control (CRC) to generative tasks. Using parameterized selection functions and admissibility functions, it provides formal uncertainty guarantees for tasks such as LLM QA, image generation, dialogue systems, and AI agents, while relaxing theoretical assumptions like the monotonicity of CRC.

## Background & Motivation

**Background**: Conformal Prediction (CP) and its extension, Conformal Risk Control (CRC), are mainstream frameworks for uncertainty quantification in supervised learning, providing distribution-free coverage guarantees for prediction sets. However, major breakthroughs in AI are currently driven by unsupervised generative models (LLMs, diffusion models, agent systems, etc.), which do not directly fit into traditional CP/CRC frameworks.

**Limitations of Prior Work**: Existing efforts to apply conformal methods to generative models are largely task-specific. For example, Quach et al. (2024) and Kladny et al. (2025) apply CRC to LLM QA, but they perform multiple independent filtering rounds (e.g., likelihood filtering, deduplication filtering) on generated answer sets. Each step requires independent calibration on different data subsets, resulting in large, impractical conformal sets. Furthermore, CRC requires utility functions to be monotonic almost everywhere, which does not hold in many generative tasks.

**Key Challenge**: The CRC theoretical framework only supports set outputs, requires callable utility functions, and necessitates strict monotonicity assumptions. In contrast, generative task outputs can be complex structures like sequences, admissibility may be defined by human evaluation (non-callable), and monotonicity might only hold in the sense of conditional expectation.

**Goal**: To design a unified framework that generalizes conformal methods to various generative tasks while relaxing CRC theoretical assumptions for broader applicability.

**Key Insight**: Several limitations in CRC (outputs must be sets, utility functions must be callable, monotonicity must hold almost everywhere) are not inherent requirements for theoretical proofs and can be systematically relaxed.

**Core Idea**: Introducing a parameterized selection function family $\mathbf{C}_\lambda$ to process inputs and generated sequences. Combined with an admissibility function $A$, the CRC calibration mechanism is generalized to "finding the minimum $\lambda$ such that the average admissibility on the calibration set meets the target," providing formal guarantees for arbitrary generative tasks.

## Method

### Overall Architecture
Conf-Gen addresses the problem of providing distribution-free formal guarantees, similar to conformal prediction, for unsupervised generative models like LLMs, diffusion models, and agents. It abstracts tasks into the processing of triplets $\mathbf{G} = (X, \mathbf{Y}, Y_{\text{GT}})$—where $X$ is the conditional input, $\mathbf{Y} = (Y_1, \dots, Y_T)$ is a sequence of candidate outputs, and $Y_{\text{GT}}$ is the optional ground truth. The process consists of two steps: first, finding the minimum parameter $\hat{\lambda}$ on a labeled calibration set such that the average output processed by the selection function $\mathbf{C}_{\hat{\lambda}}$ is "good enough" (reaches the admissibility target); during inference, the same $\hat{\lambda}$ is used to process new inputs. The resulting output structure (set or sequence) naturally possesses a formal lower bound on admissibility $\mathbb{E}[A^{(n+1)}(\hat{\lambda})] \geq \gamma$.

### Key Designs

**1. Parameterized Selection Function Family $\mathbf{C}_\lambda$: Allowing Sequence Outputs Instead of Just Sets**

The original selection function $\mathcal{C}_\lambda$ in CRC can only output sets. However, answers in generative tasks are naturally sequences (e.g., multi-turn responses, agent trajectories), where set representation loses order information and is difficult to truncate. Conf-Gen redefines the selection function as $\mathbf{C}_\lambda(x, \mathbf{y})$, which accepts the generated sequence $\mathbf{y}$ and allows the output to be a set, sequence, or single element. Larger $\lambda$ values result in more conservative outputs. A typical instance is truncation based on cumulative scores: $\mathbf{C}_\lambda(x, \mathbf{y}) = \mathbf{y}_{:\tau(x,\mathbf{y},\lambda)}$, where the stopping time $\tau(x,\mathbf{y},\lambda) = \inf\{t : \texttt{accum}(S_1^\uparrow, \dots, S_t^\uparrow) > \lambda\} \wedge |\mathbf{y}|$ indicates accumulation along sorted scores until $\lambda$ is exceeded. A key property is that the selection function has a **finite image** as a function of $\lambda$ (due to finite candidate sequence lengths and truncation points). Thus, even if $\Lambda$ is a continuous infinite set, calibration only requires enumerating finite outputs, enabling efficient searching for $\hat{\lambda}$—and facilitating the use of human evaluation for admissibility.

**2. Admissibility Functions and Instance-level Decomposition: Changing Selection Functions Without Re-labeling**

The admissibility function $A(x, \mathbf{C}_\lambda(x, \mathbf{y}), y_{\text{GT}}) \in [0, \infty]$ measures output quality; higher is better. However, it is often defined by human evaluation or LLM judges, which are expensive to call. Re-evaluating the entire output for every $\mathbf{C}_\lambda$ would cause calibration costs to explode. Conf-Gen decomposes global admissibility to the instance level: $A(x, \mathbf{y}, y_{\text{GT}}) = \texttt{agg}(A_1', \dots, A_T')$, where $A_t' = A'(x, y_t, y_{\text{GT}})$ evaluates only the quality of a single generated element $y_t$. The aggregation operator $\texttt{agg}$ can be max (at least one good answer) or min (all answers are good). This way, the calibration set only requires $\sum_{i=1}^n T_i$ evaluations of $A'$, and the evaluation results are independent of specific $\mathbf{C}_\lambda$ choices. Changing a selection function (e.g., from truncation to filtering) does not require re-collecting labels, as the evaluation and selection strategy are decoupled, significantly reducing calibration costs.

**3. $\gamma$-sensible Condition: Relaxing CRC Monotonicity to Conditional Expectation Monotonicity**

To ensure the guarantee $\mathbb{E}[A^{(n+1)}(\hat{\lambda})] \geq \gamma$, CRC requires the utility function $U^{(i)}(\lambda)$ to be monotonic almost everywhere with respect to $\lambda$. This often fails in generative tasks. For instance, in image de-memorization, theoretically increasing the prompt modification should decrease training data "memorization," but in practice, more changes can occasionally lead to more memorization; thus $\lambda \mapsto A(\lambda)$ is not monotonic for every sample. Conf-Gen relaxes the assumption to being **$\gamma$-sensible**, requiring only conditional expectation monotonicity: $\lambda \mapsto \mathbb{E}[A^{(n+1)}(\lambda) \mid \lambda', \lambda'']$ must be monotonically increasing. It also relaxes the right-continuity assumption. The cost is that the upper bound changes from exact to a more general form $\mathbb{E}[A^{(n+1)}(\hat{\lambda})] \leq \gamma + \frac{a_{\max}}{n+1} + \mathbb{E}[H]$, but the lower bound guarantee remains valid. This relaxation allows Conf-Gen to cover generative tasks where individual cases are non-monotonic but the overall trend remains reasonable.

## Key Experimental Results

### Main Results

The paper validates Conf-Gen effectiveness across 5 tasks:

| Task | Dataset | Model | Evaluation Method | Guarantee Content |
|------|---------|-------|-------------------|-------------------|
| Open-domain QA | TriviaQA | LLaMA-13B | Automatic (LLM judge) | Output sequence contains correct answer |
| Image De-memorization | Webster (2023) | Stable Diffusion v1.5 | Human Eval (10 people) | Generated image does not memorize training data |
| Conversational AI | ClariQ | LLM | Binary labels | Sufficient clarification questions asked |
| Agent Web Task | WebVoyager | LMM Agent | LLM judge | Trajectory contains successful solution |
| Random Forest | Click Prediction | RF (100 trees) | Accuracy | Subset contains ≥k correct trees |

| Task | Conf-Gen Gain | Specific Performance |
|------|---------------|----------------------|
| QA (vs Quach et al.) | Shorter output sequences | Shorter lengths at most $\gamma$ values; fewer LLM calls (due to partial generation) |
| Agent Task | Significant success rate boost | Single attempt ~60% → >65% guaranteed with <2 average attempts; ~80% with more attempts |
| Random Forest | Effective pruning | Number of selected trees stays below 59 ($2k-1$) across wide $\gamma$ range while guaranteeing majority vote |

### Key Findings
- Compared to Quach et al. (2024), Conf-Gen avoids data waste and set explosion from multi-round independent calibration, producing more compact conformal sets on TriviaQA.
- In image de-memorization, $\lambda \mapsto A(\lambda)$ is not monotonic almost everywhere, yet the $\gamma$-sensible condition holds in practice, satisfying the conformal guarantee.
- In dialogue tasks, required clarification turns increase as $\gamma$ grows but do not degenerate to the maximum number of rounds always, showing that Conf-Gen outputs are non-trivial.

## Highlights & Insights
- **Finite Image Trick Enables Human Calibration**: The selection function effectively has a finite image as a function of $\lambda$. Thus, even when admissibility is defined by human labels (non-callable), calibration is possible by evaluating only a finite number of outputs. This elegantly bypasses the barrier of human labeling for continuous $\lambda$.
- **Strong Framework Unification**: Previous independent methods like conformal factuality, conformal summarization, and conformal agent error attribution can be recovered as special cases of Conf-Gen. Table 1 summarizes compatible combinations of selection/aggregation functions.
- **Partial Generation Saves Inference Cost**: The stopping strategy based on cumulative scores allows stopping after generating $\tau$ outputs, avoiding waste in computing the remaining $T - \tau$ elements.

## Limitations & Future Work
- Calibration datasets still need to cover diverse input scenarios; robustness to distribution shift was not explored in depth.
- While the $\gamma$-sensible condition is weaker than CRC assumptions, it remains difficult to verify in some scenarios.
- The conformal guarantee is in the sense of marginal expectation ($\mathbb{E}[A^{(n+1)}(\hat{\lambda})] \geq \gamma$) and does not provide instance-specific guarantees for single test samples.
- Future directions include optimizing score function designs for various tasks and exploring more application scenarios for Conf-Gen.

## Related Work & Insights
- **vs Quach et al. (2024) / Kladny et al. (2025)**: These perform multi-round independent filtering on LLM answer sets, requiring independent calibration per round and leading to large sets. Conf-Gen uses a single $\lambda$ for unified calibration, yielding more compact sets.
- **vs Mohri & Hashimoto (2024) Conformal Factuality**: Only addresses factuality filtering at the claim level ($\texttt{agg} = \min$), which is a special case of Conf-Gen (Table 1, row 3).
- **vs Feng et al. (2026) Conformal Agent Error Attribution**: Only handles scenarios where continuous subsequences contain the first error, which also fits into the Conf-Gen framework.

## Rating
- Novelty: ⭐⭐⭐⭐ A unified framework generalizing CRC to generative tasks with substantial innovation in relaxing theoretical assumptions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 5 diverse tasks including LLM, image, dialogue, agents, and traditional ML, including human evaluation.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear exposition; the derivation chain from CP → CRC → Conf-Gen is rigorous, with a running example throughout.
- Value: ⭐⭐⭐⭐ Provides a unified conformal framework for uncertainty quantification in generative models with broad application prospects.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Conformal Reliability: A New Evaluation Metric for Conditional Generation](conformal_reliability_a_new_evaluation_metric_for_conditional_generation.md)
- [\[ICLR 2026\] Unsupervised Conformal Inference: Bootstrapping and Alignment to Control LLM Uncertainty](../../ICLR2026/image_generation/unsupervised_conformal_inference_bootstrapping_and_alignment_to_control_llm_unce.md)
- [\[ACL 2025\] D-GEN: Automatic Distractor Generation and Evaluation for Reliable Assessment of Generative Models](../../ACL2025/image_generation/d-gen_automatic_distractor_generation_and_evaluation_for_reliable_assessment_of_.md)
- [\[ICML 2026\] Generative Visual Code Mobile World Models](generative_visual_code_mobile_world_models.md)
- [\[ICML 2026\] Threshold-Guided Optimization for Visual Generative Models](threshold-guided_optimization_for_visual_generative_models.md)

</div>

<!-- RELATED:END -->
