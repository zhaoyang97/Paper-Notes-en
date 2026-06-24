---
title: >-
  [Paper Note] Can DPO Learn Diverse Human Values? A Theoretical Scaling Law
description: >-
  [NeurIPS 2025][LLM Alignment][DPO] This paper establishes a theoretical generalization framework for DPO under diverse human value settings. By analyzing the dynamic trajectory of reward margins after a finite number of gradient steps, it proves that the number of samples required per value must grow logarithmically with the number of value categories $K$ (i.e., $Q = \Theta(\log K)$) to maintain generalization performance, thereby revealing the statistical cost of aligning wi…
tags:
  - "NeurIPS 2025"
  - "LLM Alignment"
  - "DPO"
  - "value diversity"
  - "scaling law"
  - "generalization error"
  - "reward margin"
  - "preference learning theory"
date: 2026-05-08
content_hash: cf2caed6e589f9c3
---

# Can DPO Learn Diverse Human Values? A Theoretical Scaling Law

**Conference**: NeurIPS 2025
**arXiv**: [2408.03459](https://arxiv.org/abs/2408.03459)  
**Code**: [https://github.com/shawn-im/dpo-diverse](https://github.com/shawn-im/dpo-diverse)  
**Area**: LLM Alignment Theory / DPO
**Keywords**: DPO, value diversity, scaling law, generalization error, reward margin, preference learning theory

## TL;DR
This paper establishes a theoretical generalization framework for DPO under diverse human value settings. By analyzing the dynamic trajectory of reward margins after a finite number of gradient steps, it proves that the number of samples required per value must grow logarithmically with the number of value categories $K$ (i.e., $Q = \Theta(\log K)$) to maintain generalization performance, thereby revealing the statistical cost of aligning with diverse societal values.

## Background & Motivation

**Background**: DPO has become one of the standard methods for LLM alignment, widely adopted by GPT-4, Claude, Llama, and others. Most theoretical analyses assume that preference data is homogeneous and drawn from a unified reward distribution.

**Limitations of Prior Work**: Real-world society consists of diverse values—different cultures, personalities, political stances, and moral beliefs give rise to fundamentally distinct preferences. Current DPO practice typically mixes these diverse preferences into a single training dataset, yet a theoretical understanding is lacking: how does value diversity affect generalization performance, and how much data is needed to align $K$ distinct values?

**Key Challenge**: Intuitively, more values make learning harder, but what is the precise statistical relationship? Existing generalization theories either assume models are trained to near-optimality (overparameterized regime) or are independent of the training process, neither of which matches the practical setting of LLM fine-tuning that runs for only a few epochs.

**Goal**: To provide, for the first time, rigorous generalization guarantees and a scaling law for DPO under a finite-gradient-step, multi-value clustering setting.

**Key Insight**: The paper leverages the linear representation hypothesis—distinct human values are represented along approximately orthogonal directions in the LLM embedding space. Preference data is modeled as a mixture of $K$ pairs of Gaussian clusters, where each pair corresponds to the aligned/misaligned samples for one value.

**Core Idea**: By tracking the gradient-flow dynamics of the reward margin (the log-likelihood difference between preferred and non-preferred responses) for each sample during DPO training, the paper derives a precise scaling of generalization error with respect to $K$ and per-class sample count $Q$: $\mathcal{R}(\mathcal{P}) \leq 2KQ^2 e^{-Q/45}$.

## Method

### Overall Architecture
The theoretical analysis proceeds in three steps: (1) constructing a structured model of the preference distribution ($K$ pairs of orthogonal/approximately orthogonal Gaussian clusters); (2) deriving the training dynamics of reward margins under gradient flow (Lemma 4.1); and (3) using these dynamic bounds to establish a training guarantee (Theorem 4.2) and a generalization guarantee (Theorem 4.3).

### Key Designs

1. **Structured Preference Distribution**:

    - Function: Models the preference data arising from diverse values as a clustering structure in the embedding space.
    - Mechanism: Each value $i$ corresponds to a pair of clusters $C_{i,+}$ (aligned) and $C_{i,-}$ (misaligned), distributed as $\mathcal{N}(\pm c_i + b, v^2 I_d)$. Here $c_i$ is the unit direction vector for that value, $b$ is a shared component across all values (with norm $l_b$), and the $c_i$ vectors for different values are approximately orthogonal.
    - Design Motivation: Based on the linear representation hypothesis (Park et al., 2023)—concepts in LLMs are encoded along linear directions, and causally separable concepts are encoded along orthogonal directions. Figure 3 validates this assumption using the Anthropic Persona dataset.

2. **Reward Margin Dynamics Analysis**:

    - Function: Tracks how the reward margin of each sample evolves during DPO training.
    - Mechanism: Lemma 4.1 gives the gradient-flow dynamics of reward margins: $\tau \dot{r}_j = \frac{1}{N} \sum_{i=1}^{N} \beta^2 \sigma(-r_i) (\mathbf{y}_{w,j} - \mathbf{y}_{l,j})^\top (\mathbf{y}_{w,i} - \mathbf{y}_{l,i}) \Sigma_{ij}$. Two factors govern inter-sample influence: (1) a preference sharing factor (whether samples share the same preferred/rejected tokens), and (2) embedding correlation $\Sigma_{ij}$.
    - Design Motivation: Solving the ODE of the gradient flow rather than performing asymptotic analysis enables a precise characterization of performance after a finite number of training steps.

3. **Training Guarantee + Generalization Guarantee**:

    - **Theorem 4.2 (Training Reward Guarantee)**: Under specific conditions ($Z \leq \frac{1}{4}l_b^2$, $d \leq 5Q$, $v \leq \frac{1}{32\sqrt{Q}}$), with high probability all training samples achieve positive reward margins after a finite number of steps, meaning the model correctly distinguishes all training preference pairs. At the end of training, $\frac{\log 3}{40} \leq r(t) \leq \log 3$.
    - **Theorem 4.3 (Generalization Error)**: $\mathcal{R}(\mathcal{P}) \leq 2KQ^2 e^{-Q/45}$, indicating that the generalization error decreases exponentially with $Q$ (per-class sample count) but grows linearly with $K$ (number of value categories). To maintain a fixed generalization error, $Q = \Theta(\log K)$.

### Loss & Training
- Standard DPO loss (Equation 1).
- Analysis is based on gradient flow (a continuous-time approximation of gradient descent).
- Theoretical derivations target training of the unembedding layer (last-layer), with extensions to multi-token generation (Section 4.3).
- Experiments are conducted using Llama-3.1-8B, Mistral-7B-v0.3, and Qwen3-8B-Base with $\beta=0.01$ on 4×A100 GPUs.

## Key Experimental Results

### Theory vs. Experiment (Llama-3.1-8B, Last-Layer DPO)

| $K$ (# Values) | Training Reward Margin Growth Rate | Test Reward Margin Growth Rate |
|---|---|---|
| 1 | Fastest | Fastest |
| 2 | Fast | Fast |
| 4 | Moderate | Moderate |
| 8 | Slow | Slow |
| 16 | Slowest | Slowest |

Figure 5 perfectly validates the theoretical prediction: as $K$ increases, the reward margin growth rate decreases monotonically.

### Cross-Model Validation (Full Fine-Tuning)

| Model | $R^2$ (Linear Fit: $K$ vs. Test Error) |
|---|---|
| Llama-3.1-8B | 0.97 |
| Mistral-7B-v0.3 | 0.95 |
| Qwen3-8B-Base | 0.99 |

The theoretically predicted scaling trend is highly consistent under full-parameter fine-tuning as well.

### Key Findings
- **Scaling Law: $Q = \Theta(\log K)$**: When $K=10$, more than 875 samples per value are required to approach zero generalization error. This quantifies the statistical cost of aligning with a diverse society.
- **Orthogonal Structure in Embedding Space**: The Anthropic Persona data indeed exhibits near-orthogonal value directions in the Llama-3.1-8B embedding space (cosine similarity across values ≈ 0 after removing the shared component), validating the theoretical assumptions.
- **Extensible to the GPO Framework**: The theoretical framework generalizes to other preference optimization methods such as IPO ($f(r_i) = (r_i - 1)^2$) and SLiC ($f(r_i) = \max(0, 1-r_i)$).
- **Explains Known DPO Failure Modes**: The upper bound $r_U = \log 3$ in Theorem 4.2 explains why preference pairs in which the reference model assigns a probability to the rejected response more than $3^{1/\beta}$ times higher than the preferred response cannot be flipped during DPO training.

## Highlights & Insights
- **First Finite-Step DPO Generalization Theory**: Unlike classical generalization theory that assumes convergence or independence from the training process, this paper precisely tracks reward margin trajectories over a finite number of gradient steps—more faithfully reflecting the practical LLM fine-tuning regime of 2–3 epochs.
- **Practical Significance of $\Theta(\log K)$ Scaling**: Aligning $K=100$ values requires approximately 1.5× more per-class samples than $K=10$. This logarithmic growth rate implies that the data requirements for diverse alignment, while increasing, remain manageable—provided each value is represented by a sufficient number of samples.
- **Theory-to-Practice Bridge**: The results offer principled guidance for preference dataset design: simply scaling up total data volume cannot be assumed to solve multi-value generalization; the per-group sample count for each value must also grow with the total number of values.

## Limitations & Future Work
- Only in-distribution (ID) generalization is analyzed; out-of-distribution (OOD) settings are not considered.
- Theoretical guarantees are strongest for last-layer training; full fine-tuning is empirically validated but lacks rigorous theoretical backing.
- The mixture-of-Gaussians with orthogonal directions assumption, while empirically supported, may not hold for all value types (e.g., highly correlated value pairs).
- Appendix C extends the analysis to $\delta$-approximately orthogonal clusters, but at the cost of wider bounds.

## Related Work & Insights
- **vs. Shirali et al. (2025)**: That work identifies limitations of DPO on heterogeneous data but does not provide a scaling law. This paper delivers the precise $\Theta(\log K)$ scaling.
- **vs. RLCF (2507.18624)**: RLCF addresses reward signal quality issues via instruction-specific checklists; this paper reveals from a theoretical perspective that even with perfect reward signals, value diversity itself incurs a statistical cost.
- **vs. PAL / Projection Optimization**: These works propose concrete methods for handling heterogeneous preferences; this paper provides the theoretical foundation explaining why such methods are necessary.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First finite-step DPO generalization framework with a scaling law; introduces NTK/gradient flow analysis into preference learning theory.
- Experimental Thoroughness: ⭐⭐⭐⭐ Theoretical predictions are validated across 3 models, though experiments primarily serve to verify theory rather than demonstrate practical applications.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are rigorous but the appendix is extremely lengthy; main conclusions are clearly presented (the scaling curve in Figure 4 is intuitive).
- Value: ⭐⭐⭐⭐⭐ Provides a theoretical foundation for diverse value alignment; the $\Theta(\log K)$ scaling offers direct guidance for dataset design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Preference Learning with Lie Detectors can Induce Honesty or Evasion](preference_learning_with_lie_detectors_can_induce_honesty_or_evasion.md)
- [\[ACL 2026\] BACH-V: Bridging Abstract and Concrete Human-Values in Large Language Models](../../ACL2026/llm_alignment/bach-v_bridging_abstract_and_concrete_human-values_in_large_language_models.md)
- [\[NeurIPS 2025\] Strategyproof Reinforcement Learning from Human Feedback](strategyproof_reinforcement_learning_from_human_feedback.md)
- [\[ICML 2026\] Large Language Models Should Learn Personalized Rather Than Aggregated Human Preferences](../../ICML2026/llm_alignment/large_language_models_should_learn_personalized_rather_than_aggregated_human_pre.md)
- [\[NeurIPS 2025\] Capturing Individual Human Preferences with Reward Features](capturing_individual_human_preferences_with_reward_features.md)

</div>

<!-- RELATED:END -->
