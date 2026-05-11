---
title: >-
  [Paper Note] The Hot Mess of AI: How Does Misalignment Scale With Model Intelligence and Task Complexity?
description: >-
  [ICLR 2026][bias-variance decomposition] This paper decomposes AI model errors into bias (systematic misalignment) and variance (incoherent behavior), finding that longer reasoning leads to greater incoherence…
tags:
  - "ICLR 2026"
  - "bias-variance decomposition"
  - "AI incoherence"
  - "reasoning length"
  - "model scale"
  - "AI alignment"
date: 2026-05-08
content_hash: 6f805942d9d02bb5
---

# The Hot Mess of AI: How Does Misalignment Scale With Model Intelligence and Task Complexity?

**Conference**: ICLR 2026
**arXiv**: [2601.23045](https://arxiv.org/abs/2601.23045)
**Code**: Available
**Area**: Other / AI Safety
**Keywords**: bias-variance decomposition, AI incoherence, reasoning length, model scale, AI alignment

## TL;DR
This paper decomposes AI model errors into bias (systematic misalignment) and variance (incoherent behavior), finding that longer reasoning leads to greater incoherence, and that larger models become more incoherent on difficult tasks. This suggests that future superintelligent AI is more likely to exhibit unpredictable, "industrial accident"-style failures than to coherently pursue wrong objectives.

## Background & Motivation
A central concern in AI alignment is that models may coherently pursue incorrect goals (misalignment). In practice, however, AI failures are often random and incoherent—resembling a "hot mess" rather than a shrewd adversary. The key question is: as AI capability and task complexity increase, will failures look more like systematic pursuit of wrong objectives (bias-dominated) or unpredictable chaotic behavior (variance-dominated)?

The "hot mess theory of intelligence" (Sohl-Dickstein, 2023) posits that as agents become more intelligent, their behavior tends to become more incoherent and less describable by a single objective. If this holds for AI, it would fundamentally reshape the likelihood and focus of misalignment risks. This paper quantifies this question via the decomposition $\text{Error} = \text{Bias}^2 + \text{Variance}$, systematically validating it across multiple tasks and models.

## Method

### Bias-Variance Decomposition Framework
The core mechanism involves sampling the same question at least 30 times to estimate the distribution of model outputs, then decomposing total error into bias and variance components.

**KL decomposition** (Equation 1): For input $x$, model $f_\varepsilon$ produces a probability distribution, and target $y$ is one-hot encoded:

$$\mathbb{E}_\varepsilon[\text{CE}(y, f_\varepsilon)] = D_{KL}(y \| \bar{f}) + \mathbb{E}_\varepsilon[D_{KL}(\bar{f} \| f_\varepsilon)]$$

where the first term is KL-Bias² (divergence between the average prediction and the true target), and the second term is KL-Variance (inconsistency across individual predictions).

**Distinction from classical literature**: Traditional bias-variance decomposition takes expectations over training randomness (different random seeds). Here, expectations are taken over test-time randomness (sampling and few-shot context variation), since the analysis concerns fixed models rather than re-trained ones.

### Incoherence Definition
$$\text{Incoherence}(Q, f_\varepsilon) = \frac{\sum_i \text{Variance}(q_i, f_\varepsilon)}{\sum_i \text{Error}(q_i, f_\varepsilon)} \in [0, 1]$$

- $0$ = fully coherent (consistently correct or incorrect; pure bias)
- $1$ = fully random (pure variance)
- Key advantage: incoherence remains comparable across models of different capability even as overall error rates decrease.

### Experimental Design
1. **Multiple-choice tasks**: GPQA (scientific reasoning) and MMLU (general knowledge). Each question is sampled ≥30 times with different seeds and few-shot contexts.
2. **Agent coding**: SWE-Bench, using unit tests as binary indicators decomposed into bias and variance.
3. **Safety evaluation**: Model Written Evals (MWE), covering both multiple-choice and open-ended formats. Open-ended responses are assessed using embedding variance.
4. **Synthetic setting**: Transformers of varying sizes are trained to simulate optimizer descent on ill-conditioned quadratic functions (condition number = 50), using decoding-based regression and teacher-forcing training.
5. **Human survey**: Disjoint groups of participants rank AI systems, humans, and organizations on both intelligence and coherence.

### Analytical Dimensions
- **Reasoning length analysis**: Questions are grouped by average reasoning token count to examine the incoherence–length relationship.
- **Controlling for natural variation**: Within the same question, responses are split into "short reasoning" and "long reasoning" groups by median token length.
- **Scale analysis**: The Qwen3 series (0.6B–32B) is used, with incoherence–scale relationships analyzed across difficulty-stratified question groups.

## Key Experimental Results

### Finding 1: Longer Reasoning → Greater Incoherence

| Setting | Trend | Notes |
|---------|-------|-------|
| GPQA (Sonnet 4 / o3-mini / o4-mini) | Longer reasoning → more incoherent | Consistent across all models |
| SWE-Bench (o3-mini / o4-mini) | More actions → more incoherent | Consistent in agent tasks |
| MWE safety questions | Embedding variance ↑ with length | Holds for open-ended format |
| Synthetic optimizer | More steps → higher variance | Validated in controlled setting |

The effect persists after controlling for task difficulty: when responses to each question are split by median reasoning length, the naturally longer group shows significantly higher incoherence despite minimal accuracy differences. The effect of natural reasoning variation is far stronger than that of reasoning budget manipulation.

### Finding 2: Effect of Model Scale Depends on Task Difficulty

| Question Difficulty | Incoherence Change (Qwen3 0.6B→32B) | Notes |
|--------------------|--------------------------------------|-------|
| Easy | ↓ (more coherent) | Increased capability reduces random errors |
| Medium | ≈ (stable) | Transitional regime |
| **Hard** | **↑ (more incoherent)** | Bias decreases faster than variance |

Key mechanism: both bias and variance decrease as model size grows, but the slope of bias reduction is similar across difficulty groups, whereas the slope of variance reduction is shallower for hard questions. On the most difficult questions, the variance slope is lower than the bias slope, making variance the binding constraint.

### Finding 3: Synthetic Optimizer Validation
Transformers of varying sizes are trained to simulate optimizer descent on quadratic functions:
- Training loss follows a clear power law.
- During autoregressive rollout, bias decreases much faster than variance as model size increases.
- This indicates that models learn the correct objective more readily (fast bias reduction) than they maintain long-horizon coherent behavior (slow variance reduction).

### Finding 4: Ensembling and Reasoning Budget
- **Ensembling**: Aggregating $E$ samples reduces variance at a rate of $1/E$, effectively lowering incoherence.
- **Larger reasoning budget**: Slightly reduces incoherence, but the effect is far weaker than that of natural reasoning length variation.
- The authors conjecture that improvements from reasoning budget may stem from better backtracking and error correction.

## Discussion & Insights

### Why Do More Capable Models Become More Incoherent?
1. **LLMs are dynamical systems, not optimizers**: Among all dynamical systems, those that precisely optimize a fixed loss constitute a set of measure zero. As capability and state space expand, constraining a model to behave as an optimizer becomes increasingly difficult.
2. **Variance accumulates along trajectories**: Unless active error-correction mechanisms (e.g., ensembling) are in place, variance grows with action sequence length. In real-world deployments, actions are typically irreversible and cannot be reset and corrected as in experimental settings.

### Further Decomposition of Bias
$\text{Bias} = \text{Bias}_{\text{mesa}} + \text{Bias}_{\text{spec}}$, where the former reflects divergence between model behavior and the training objective, and the latter reflects divergence between the training objective and the true goal (reward misspecification). In the tasks studied, $\text{Bias}_{\text{spec}}$ is negligible; however, in real deployments, $\text{Bias}_{\text{spec}}$ may dominate errors as capability increases. This underscores the importance of precise training objective specification.

### Implications for AI Safety
- If incoherence grows with capability and task complexity, failures of advanced future AI will more closely resemble "industrial accidents" than "malicious adversaries."
- This shifts the focus of AI safety from defending against coherent scheming toward preventing unpredictable accidents.
- It elevates the relative importance of research on reward hacking and goal misspecification.
- However, this does not imply that misalignment is unimportant—$\text{Bias}_{\text{spec}}$ may still dominate in practice.

## Highlights & Insights
- The paper introduces a novel quantitative framework for AI safety discussions (bias-variance decomposition), transforming the vague question of "how will AI fail" into a measurable one.
- The "hot mess theory" perspective is original and empirically supported—being smarter does not imply being more coherent.
- The synthetic optimizer experiment elegantly controls for confounds, directly validating the claim that learning the correct objective is easier than maintaining coherent long-horizon behavior.
- Consistency between the human survey results and the LLM experimental findings enhances cross-domain credibility.
- The work has substantive implications for AI governance: should society prepare for industrial accidents or adversarial attacks?

## Limitations & Future Work
- Bias is only defined relative to a known target—in open-ended tasks (e.g., generation, dialogue) where targets are ambiguous, the applicability of the decomposition is limited.
- While 30 samples are validated as sufficient, estimation in high-dimensional output spaces may still be noisy.
- Extrapolating from current frontier models to future superintelligent AI is risky, as future training methods may alter the bias-variance structure fundamentally.
- Variance in deployment can be mitigated through ensembling and repeated sampling, which partially limits the practical severity of the "industrial accident" conclusion.
- The paper does not deeply analyze the specific mechanisms underlying incoherence (the *why*); results are primarily descriptive.

## Related Work & Insights
- Complements the reasoning scaling law literature (Gema et al. 2025: inverse scaling)—not only does performance degrade, but errors become less consistent.
- Connects to evaluation variance literature (Biderman et al. 2024: high variance in evaluations).
- Self-consistency (Wang et al. 2023) can be reinterpreted as a mechanism for reducing incoherence.
- Forms an interesting contrast with the platonic representation hypothesis (Huh et al. 2024: convergence of representations)—representations may converge while behavior remains incoherent.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Both the problem formulation and methodology are highly original, opening a new analytical dimension.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-task evaluation, synthetic validation, and human surveys provide broad coverage.
- Writing Quality: ⭐⭐⭐⭐⭐ Engaging prose, apt metaphors, and excellent visualizations.
- Value: ⭐⭐⭐⭐⭐ Offers profound guidance for the direction of AI safety research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Intrinsic Barriers and Practical Pathways for Human-AI Alignment: An Agreement-Based Complexity Analysis](../../AAAI2026/others/intrinsic_barriers_and_practical_pathways_for_human-ai_alignment_an_agreement-ba.md)
- [\[AAAI 2026\] Bridging the Skills Gap: A Course Model for Modern Generative AI Education](../../AAAI2026/others/bridging_the_skills_gap_a_course_model_for_modern_generative_ai_education.md)
- [\[AAAI 2026\] How Wide and How Deep? Mitigating Over-Squashing of GNNs via Channel Capacity Constrained Estimation](../../AAAI2026/others/how_wide_and_how_deep_mitigating_over-squashing_of_gnns_via_channel_capacity_con.md)
- [\[AAAI 2026\] How to Marginalize in Causal Structure Learning?](../../AAAI2026/others/how_to_marginalize_in_causal_structure_learning.md)
- [\[ICLR 2026\] Speculative Actions: A Lossless Framework for Faster AI Agents](speculative_actions_faster_ai_agents.md)

</div>

<!-- RELATED:END -->
