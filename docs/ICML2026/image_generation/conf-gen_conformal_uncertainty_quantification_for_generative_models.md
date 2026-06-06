---
title: >-
  [Paper Note] Conf-Gen: Conformal Uncertainty Quantification for Generative Models
description: >-
  [ICML 2026][Image Generation][Conformal Prediction] The paper proposes the Conf-Gen framework, extending Conformal Risk Control (CRC) to generative tasks. By utilizing parameterized selection functions and admissibility…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Conformal Prediction"
  - "Uncertainty Quantification"
  - "Generative Models"
  - "Conformal Risk Control"
  - "Selection Function"
date: 2026-05-08
content_hash: 66ea0f1d681cc762
---

# Conf-Gen: Conformal Uncertainty Quantification for Generative Models

**Conference**: ICML 2026  
**arXiv**: [2605.28920](https://arxiv.org/abs/2605.28920)  
**Code**: https://github.com/layer6ai-labs/conf-gen (Available)  
**Area**: Uncertainty Quantification / Generative Models  
**Keywords**: Conformal Prediction, Uncertainty Quantification, Generative Models, Conformal Risk Control, Selection Function  

## TL;DR
The paper proposes the Conf-Gen framework, extending Conformal Risk Control (CRC) to generative tasks. By utilizing parameterized selection functions and admissibility functions, it provides formal uncertainty guarantees for tasks such as LLM QA, image generation, dialogue systems, and AI Agents, while relaxing theoretical assumptions like monotonicity required by CRC.

## Background & Motivation

**Background**: Conformal Prediction (CP) and its extension Conformal Risk Control (CRC) are the dominant frameworks for uncertainty quantification in supervised learning, providing distribution-free coverage guarantees for prediction sets. However, major AI breakthroughs are currently driven by unsupervised generative models (LLMs, diffusion models, Agent systems, etc.), which do not directly fit into traditional CP/CRC frameworks.

**Limitations of Prior Work**: Existing works applying conformal methods to generative models are largely task-specific. For example, Quach et al. (2024) and Kladny et al. (2025) use CRC for LLM QA but perform multiple independent filtering rounds (e.g., likelihood or deduplication filtering) on the generated answer set. Each step requires independent calibration on different data subsets, leading to excessively large and impractical conformal sets. Furthermore, CRC requires utility functions to be almost everywhere monotonically increasing, which does not hold in many generative tasks.

**Key Challenge**: The CRC theoretical framework only supports set outputs, requires callable utility functions, and necessitates strict monotonicity assumptions. In contrast, generative outputs can be complex structures like sequences, admissibility may be defined by human evaluation (non-callable), and monotonicity might only hold in terms of conditional expectation.

**Goal**: Design a unified framework to generalize conformal methods to various generative tasks while relaxing CRC's theoretical assumptions for broader applicability.

**Key Insight**: The authors observe that several constraints in CRC (outputs must be sets, utility functions must be callable, monotonicity must hold almost everywhere) are not essential requirements for theoretical proofs and can be systematically relaxed.

**Core Idea**: Introduce a parameterized selection function $\mathbf{C}_\lambda$ to handle inputs and generated sequences. Combined with an admissibility function $A$, the CRC calibration mechanism is generalized to "finding the minimum $\lambda$ such that the average admissibility on the calibration set meets the threshold," thereby providing formal guarantees for arbitrary generative tasks.

## Method

### Overall Architecture
The input to Conf-Gen is a triplet $\mathbf{G} = (X, \mathbf{Y}, Y_{\text{GT}})$, where $X$ is the conditional input (e.g., a question), $\mathbf{Y} = (Y_1, \dots, Y_T)$ is a series of outputs from a generative model (e.g., multiple answers), and $Y_{\text{GT}}$ is the optional ground truth. The framework consists of three core components: a family of selection functions $\mathbf{C}_\lambda$, an admissibility function $A$, and a calibration dataset $\mathbf{D}_{:n}$. During the calibration phase, the minimum $\hat{\lambda}$ is found on labeled data such that the average admissibility exceeds the threshold $\frac{n+1}{n}\gamma$. During the inference phase, $\mathbf{C}_{\hat{\lambda}}$ processes new inputs, and the output structure (set or sequence) obtains a formal lower bound guarantee for admissibility: $\mathbb{E}[A^{(n+1)}(\hat{\lambda})] \geq \gamma$.

### Key Designs

1.  **Parameterized Selection Function Family $\mathbf{C}_\lambda$**:

    - **Function**: Maps input $X$ and generated sequence $\mathbf{Y}$ to an output space $\mathcal{S}$ (which can be a set, sequence, or single element); larger $\lambda$ indicates more conservative output.
    - **Mechanism**: Defines a score-based selection strategy, e.g., $\mathbf{C}_\lambda(x, \mathbf{y}) = \mathbf{y}_{:\tau(x,\mathbf{y},\lambda)}$, where $\tau(x,\mathbf{y},\lambda) = \inf\{t : \texttt{accum}(S_1^\uparrow, \dots, S_t^\uparrow) > \lambda\} \wedge |\mathbf{y}|$ is a stopping time based on accumulated scores. Since the selection function as a function of $\lambda$ has only finitely many images, efficient calibration is possible even if $\Lambda$ is an infinite set.
    - **Design Motivation**: Unlike $\mathcal{C}_\lambda$ in CRC which only outputs sets, $\mathbf{C}_\lambda$ allows sequence or single-element outputs and takes the generated sequence $\mathbf{Y}$ as an extra input, naturally suiting generative tasks. The finite image property enables calibration even when admissibility is defined by human evaluation (non-callable).

2.  **Admissibility Function and Instance-level Decomposition**:

    - **Function**: Measures the quality of the selection function output, $A(x, \mathbf{C}_\lambda(x, \mathbf{y}), y_{\text{GT}}) \in [0, \infty]$, where larger values indicate higher quality.
    - **Mechanism**: Decomposes global admissibility as $A(x, \mathbf{y}, y_{\text{GT}}) = \texttt{agg}(A_1', \dots, A_T')$, where $A_t' = A'(x, y_t, y_{\text{GT}})$ is the instance-level admissibility of a single generated element, and $\texttt{agg}$ can be max (at least one good answer) or min (all answers are good). This way, it only requires evaluating $A'$ for $\sum_{i=1}^n T_i$ times, rather than re-evaluating for every variation of $\mathbf{C}_\lambda$.
    - **Design Motivation**: Decoupling admissibility evaluation from the selection function means that changing the selection function does not require re-collecting labels, significantly reducing calibration costs.

3.  **$\gamma$-sensible Condition and Relaxed Monotonicity Assumption**:

    - **Function**: Provides sufficient conditions for the conformal guarantee $\mathbb{E}[A^{(n+1)}(\hat{\lambda})] \geq \gamma$.
    - **Mechanism**: CRC requires the utility function $U^{(i)}(\lambda)$ to be monotonically increasing in $\lambda$ almost everywhere. Conf-Gen relaxes this to conditional expectation monotonicity, i.e., $\lambda \mapsto \mathbb{E}[A^{(n+1)}(\lambda) \mid \lambda', \lambda'']$ being monotonically increasing. It also relaxes the right-continuity assumption and provides a more general upper bound $\mathbb{E}[A^{(n+1)}(\hat{\lambda})] \leq \gamma + \frac{a_{\max}}{n+1} + \mathbb{E}[H]$.
    - **Design Motivation**: In tasks like image de-memorization, images with more prompt modifications might occasionally become "more memorized," violating almost-everywhere monotonicity, yet conditional expectation monotonicity remains reasonable.

## Key Experimental Results

### Main Results

The paper validates the effectiveness of Conf-Gen across 5 tasks:

| Task | Dataset | Model | Evaluation Method | Guarantee Content |
|------|--------|------|----------|----------|
| Open-domain QA | TriviaQA | LLaMA-13B | Auto (LLM judge) | Output sequence contains correct answer |
| Non-memorized Image Gen | Webster (2023) | Stable Diffusion v1.5 | Human Eval (10 people) | Generated image does not memorize training data |
| Conversational AI | ClariQ | LLM | Binary Label | Sufficient clarification questions asked |
| Agent Web Tasks | WebVoyager | LMM Agent | LLM judge | Trajectory sequence contains success plan |
| Random Forest | Click Prediction | RF (100 trees) | Accuracy | Subset contains $\geq k$ correct trees |

| Task | Conf-Gen Advantage | Specific Performance |
|------|--------------|----------|
| QA (vs Quach et al.) | Shorter output sequences | Shorter sequence lengths across most $\gamma$ values, with fewer LLM calls (due to partial generation) |
| Agent Tasks | Significant success rate boost | Single attempt ~60% $\rightarrow$ average <2 attempts guarantees >65%; allowing more attempts reaches ~80% |
| Random Forest | Effective pruning | Number of selected trees stays below 59 ($2k-1$) across a wide range of $\gamma$, ensuring majority vote correctness |

### Key Findings
- Compared to Quach et al. (2024), Conf-Gen avoids data waste and set expansion caused by multi-round independent calibration, producing more compact conformal sets on TriviaQA.
- In image de-memorization tasks, $\lambda \mapsto A(\lambda)$ is not monotonic almost everywhere, but the $\gamma$-sensible condition holds in practice, and conformal guarantees are still met.
- In dialogue tasks, as $\gamma$ increases, the required number of clarification rounds increases but does not degenerate into always choosing the maximum number of rounds, indicating that Conf-Gen produces non-trivial outputs.

## Highlights & Insights
- **Finite image trick makes human evaluation calibration feasible**: Since the selection function as a function of $\lambda$ has only finitely many images, calibration only requires evaluating a finite number of outputs, even if admissibility is defined by human labels (non-callable). This design elegantly bypasses the hurdle of being unable to perform human labeling for continuous $\lambda$.
- **Strong framework unification**: Prior independent methods such as conformal factuality, conformal summarization, and conformal Agent error attribution can all be recovered as special cases of Conf-Gen. Table 1 clearly summarizes compatible combinations of various selection and aggregation functions.
- **Partial generation saves inference costs**: The stopping strategy based on accumulated scores allows stopping after generating only $\tau$ outputs during inference, avoiding the computational waste of generating the remaining $T - \tau$ elements.

## Limitations & Future Work
- Calibration datasets still need to cover sufficiently diverse input scenarios; robustness to distribution shift has not been explored in depth.
- While the conditional expectation monotonicity of $\gamma$-sensible is weaker than CRC's assumptions, it remains difficult to verify in some scenarios.
- The conformal guarantee is in the sense of marginal expectation ($\mathbb{E}[A^{(n+1)}(\hat{\lambda})] \geq \gamma$) and does not provide instance-specific guarantees for single test samples.
- Future directions include optimizing score function designs for various tasks and exploring more application scenarios for Conf-Gen.

## Related Work & Insights
- **vs Quach et al. (2024) / Kladny et al. (2025)**: These perform multi-round independent screening on LLM answer sets, requiring independent calibration for each round, which leads to oversized conformal sets. Conf-Gen uses a single $\lambda$ for unified calibration, yielding more compact sets.
- **vs Mohri & Hashimoto (2024) Conformal Factuality**: Only handles claim-level factuality screening ($\texttt{agg} = \min$), which is a special case of Conf-Gen (Table 1, row 3).
- **vs Feng et al. (2026) Conformal Agent Error Attribution**: Only handles scenarios where continuous subsequences contain the first error, which can similarly be subsumed under the Conf-Gen framework.

## Rating
- Novelty: ⭐⭐⭐⭐ A unified framework generalizing CRC systems to generative tasks with substantial innovation in relaxing theoretical assumptions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 different tasks covering LLM/Image/Dialogue/Agent/Traditional ML, including human evaluation.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear exposition with a rigorous derivation chain from CP $\rightarrow$ CRC $\rightarrow$ Conf-Gen; a running example is used throughout.
- Value: ⭐⭐⭐⭐ Provides a unified conformal framework for uncertainty quantification in generative models with broad application prospects.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Unsupervised Conformal Inference: Bootstrapping and Alignment to Control LLM Uncertainty](../../ICLR2026/image_generation/unsupervised_conformal_inference_bootstrapping_and_alignment_to_control_llm_unce.md)
- [\[ICML 2026\] Conformal Reliability: A New Evaluation Metric for Conditional Generation](conformal_reliability_a_new_evaluation_metric_for_conditional_generation.md)
- [\[ICML 2026\] Generative Visual Code Mobile World Models](generative_visual_code_mobile_world_models.md)
- [\[ICML 2026\] Threshold-Guided Optimization for Visual Generative Models](threshold-guided_optimization_for_visual_generative_models.md)
- [\[ICML 2026\] RAIGen: Rare Attribute Identification in Text-to-Image Generative Models](raigen_rare_attribute_identification_in_text-to-image_generative_models.md)

</div>

<!-- RELATED:END -->
