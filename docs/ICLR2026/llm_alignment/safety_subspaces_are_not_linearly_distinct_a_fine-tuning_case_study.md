---
title: >-
  [Paper Note] Safety Subspaces are Not Linearly Distinct: A Fine-Tuning Case Study
description: >-
  [ICLR 2026][LLM Alignment][Safety alignment] This paper comprehensively validates a key finding across five open-source LLMs through four systematic experiments (parallel projection, orthogonal projection, subspace overlap, and activation space analysis): safety alignment behaviors are highly entangled with general learning in both the weight space and activation space. There is no linearly separable independent safety subspace, indicating that defense strategies based on sub…
tags:
  - "ICLR 2026"
  - "LLM Alignment"
  - "Safety alignment"
  - "subspace"
  - "fine-tuning attacks"
  - "linear separability"
  - "weight space"
date: 2026-05-08
content_hash: 837cfe574e1b0240
---

# Safety Subspaces are Not Linearly Distinct: A Fine-Tuning Case Study

**Conference**: ICLR 2026  
**arXiv**: [2505.14185](https://arxiv.org/abs/2505.14185)  
**Code**: [GitHub](https://github.com/CERT-Lab/safety-subspaces)  
**Area**: LLM Safety / Alignment  
**Keywords**: Safety alignment, subspace, fine-tuning attacks, linear separability, weight space

## TL;DR

This paper comprehensively validates a key finding across five open-source LLMs through four systematic experiments (parallel projection, orthogonal projection, subspace overlap, and activation space analysis): safety alignment behaviors are highly entangled with general learning in both the weight space and activation space. There is no linearly separable independent safety subspace, indicating that defense strategies based on subspace projection/filtering face fundamental limitations.

## Background & Motivation

**Background**: LLMs can reject harmful prompts after safety alignment (e.g., RLHF), but this safety is extremely fragile—continued fine-tuning even on benign data can undermine safety behaviors. Mixing a small number of malicious samples into the training set can subvert alignment. This exposes a deeper attack surface at the weight level beyond prompt injection: alignment degradation.

**Limitations of Prior Work**: Several studies (e.g., SafeLoRA, LDIFS) attempt to utilize "safety subspaces" to defend against fine-tuning attacks. The core hypothesis is that safety alignment information is concentrated in specific linear directions within the weight space, which can be extracted via SVD and protected during subsequent fine-tuning. However, this hypothesis has never been rigorously scrutinized.

**Key Challenge**: If safety information indeed resided in an independent linear subspace, one could orthogonalize harmful updates to the safety directions via simple projection, maintaining safety while preserving task performance. However, if safety and general learning are highly entangled (i.e., the same directions amplify both safety and harmful behaviors), projection-based defenses will fail to selectively suppress harmfulness without losing utility.

**Goal**: To systematically examine the fundamental hypothesis of whether "LLM safety behavior is concentrated in specific linear subspaces."

**Key Insight**: Instead of proposing a new defense method, the authors perform a rigorous empirical study. They construct candidate safety subspaces from two perspectives: alignment updates $\Delta_A$ (aligned - base) and safety-tuned updates $\Delta_S$ (safety-tuned - base), and then test their specificity through projection experiments.

**Core Idea**: Through four meticulously designed experiments, the authors prove that safety-related weight updates and activation patterns are not linearly separable from general learning, imposing fundamental constraints on subspace-based defense strategies.

## Method

### Overall Architecture

This paper does not propose a new method but seeks to directly answer a widely assumed but unverified question: Does the safety alignment behavior of LLMs truly concentrate on a few independent linear directions in the weight space? If so, orthogonalizing harmful updates to these directions would preserve both safety and performance; if not, all subspace-projection-based defenses are untenable. The authors designed four progressive experiments along a logical chain: first, checking if useful and harmful updates have different "expressivity" in candidate safety subspaces; second, seeing if orthogonalizing this subspace can selectively erase harmfulness; third, directly quantifying the geometric overlap between useful, harmful, and safety updates; and finally, descending into the activation space to see if internal representations of harmful and useful prompts are separable.

The two basic materials for the experiments are two types of candidate safety subspaces: extracted from alignment updates $\Delta_A$ (aligned model weights - base weights) and safety-tuned updates $\Delta_S$ (safety-tuned weights - base weights). These are tested against two types of task updates: useful updates $\Delta_T^{\text{Useful}}$ obtained by fine-tuning on MetaMathQA and harmful updates $\Delta_T^{\text{Harmful}}$ obtained by fine-tuning on a harmful subset of BeaverTails.

### Key Designs

**1. Parallel Projection: Does the subspace encode "safety" or "efficient learning"?**

The first step is to confirm whether the top directions of candidate safety subspaces are specific to safety behavior. The authors perform SVD on $\Delta_{A/S}$, take the top-k singular directions to construct a projection matrix $P_k$, and project the task updates into this subspace: $\tilde{\Delta}_T^j = P_k \Delta_T^j$. Behavior is measured using GSM8k accuracy (utility) and AdvBench harmfulness scores. If these were truly "safety directions," their ability to retain harmful updates should differ significantly from useful ones. Instead, the results show that energy is uniformly distributed, and the retention ratios are nearly identical. Notably, these top-k directions amplify both utility **and** harmfulness better than random projections. In other words, these directions characterize "efficient learning" (high parameter sensitivity) rather than "safety."

**2. Orthogonal Projection: Can removing the safety subspace kill only harmfulness?**

Since projecting into the space amplifies everything, the inverse is tested: in a mixed fine-tuning scenario (80% useful + 20% harmful), can orthogonalizing the component of the update aligned with candidate safety directions $\tilde{\Delta}_T = P_k^{\perp} \Delta_T$ selectively suppress harmfulness? This is the core mechanism of defenses like SafeLoRA and LDIFS. The experiment shows a negative result: utility and harmfulness drop synchronously without selectivity. Worse, after removing top-k directions, utility drops faster than with random projection, while harmfulness drops at a rate similar to random projection. Safety gains always come with a proportional loss in task performance.

**3. Mode Subspace Overlap (MSO): What is the geometric structure between the three updates?**

While the first two experiments infer properties indirectly, the third quantifies geometric relationships directly. The authors perform SVD on three types of updates—$\Delta_A$ (alignment), $\Delta_T^{\text{Harmful}}$ (harmful), and $\Delta_T^{\text{Useful}}$ (useful). For a given energy retention ratio $\eta$, they take the minimum top-k directions to cover $\eta$ of the Frobenius energy, obtaining orthogonal bases $Q_V, Q_W$. The overlap matrix $S = Q_V^{\top} Q_W$ is used to calculate the overlap:

$$\mathrm{MSO}(\mathbf{V}, \mathbf{W}; \eta) = \frac{\|S\|_F^2}{\min(k_V, k_W)}, \quad 0 \le \mathrm{MSO} \le 1$$

MSO is 0 if subspaces are orthogonal and 1 if they span the same space. An expected overlap from random subspaces $\max(k_V, k_W)/d$ serves as the baseline. Intuitively, "safety directions" should appear in the shared directions of **alignment and harmful updates** because their effects on safety are opposite. However, the results show that the strongest overlap is between **useful $\leftrightarrow$ harmful updates**, not alignment $\leftrightarrow$ harmful. Useful and harmful updates share the most primary directions, forming a "general learning subspace" that is expressive for tasks but agnostic to safety.

**4. Activation Space Analysis: Is there separability at the representation level?**

Finally, the perspective shifts from weight updates to internal representations to check if intermediate activations for harmful vs. useful prompts occupy different regions. If they were separable, defenses might still work at the activation level even if weights are entangled. Consistent with the first three experiments, harmful and useful prompt activations overlap significantly, and no safety-specific linear direction can be found in the activation space.

### Loss & Training

Fine-tuning uses standard training: useful data from a 20K subset of MetaMathQA, harmful data from a 4K unsafe subset of BeaverTails, and mixed data (20% harmful + 80% useful). Safety tuning uses BeaverTails entries where `is_safe=True`. Harmfulness is evaluated by GPT-4o-mini scoring AdvBench outputs on a scale of 1-5.

## Key Experimental Results

### Parallel Projection Experiment (Qwen-2.5 1.5B)

| Method | SVD 0.01 | 0.25 | 0.50 | 0.75 | 0.99 | Full FT |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Top-K (Utility↑) | 0.50 | 0.53 | 0.55 | 0.57 | 0.58 | 0.61 |
| Random (Utility↑) | 0.49 | 0.50 | 0.53 | 0.53 | 0.56 | 0.61 |
| Top-K (Harm↓) | 1.62 | 1.80 | 1.92 | 1.90 | 1.97 | 2.09 |
| Random (Harm↓) | 1.56 | 1.65 | 1.74 | 1.83 | 1.95 | 2.09 |

### Orthogonal Projection Experiment (Mixed FT Qwen-2.5 1.5B)

| Method | SVD 0.01 | 0.25 | 0.50 | 0.75 | 0.99 | Full FT |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Top-K (Utility↑) | 0.50 | 0.53 | 0.55 | 0.57 | 0.58 | 0.60 |
| Top-K (Harm↓) | 1.58 | 1.65 | 1.80 | 1.91 | 1.92 | 2.16 |

### Ablation Study

| Configuration | Description |
| :--- | :--- |
| Alignment Subspace $\Delta_A$ | Amplifies both useful and harmful behaviors; no safety specificity. |
| Safety Subspace $\Delta_S$ | Similarly amplifies both behaviors; no selectivity. |
| Random-K Control | Randomly selecting k singular vectors results in weaker behavioral impact than Top-K. |
| Random Control | SVD of a random matrix shows the weakest behavioral impact. |

### Key Findings
- **Core Negative Conclusion**: Across five LLMs (Llama 3.2 1B, Llama 2 7B, Qwen-2.5 1B/3B/7B), it is consistently observed that no linearly separable safety subspace exists.
- Top-k alignment directions amplify both utility and harmfulness—they are "high-impact learning directions" rather than "safety directions."
- Orthogonal projection cannot selectively remove harmfulness: utility drops faster than harmfulness when top-k directions are removed.
- MSO analysis: The overlap between harmful and safety updates is not higher than that between useful and safety updates, refuting the hypothesis of a special geometric relationship.

## Highlights & Insights
- The **progressive experimental design** is highly rigorous—corroborating the same conclusion from projection effects, orthogonal removal, geometric overlap, and activation representation.
- The discovery that "top-k directions amplify all behaviors" provides a crucial understanding: the primary directions found by alignment/safety training are not dedicated to safety but are general-purpose learning directions with high parameter sensitivity.
- The use of control experiments (Random-K, Random) ensures that findings are not artifacts of the projection process itself but stem from the lack of safety specificity in the subspaces.

## Limitations & Future Work
- The paper only refutes the separability of **linear** subspaces; it does not rule out whether non-linear methods (e.g., manifold learning, kernel methods) could achieve separation.
- Harmfulness evaluation relies on GPT-4o-mini scoring, which carries the evaluator's own biases and instability.
- Safety-tuning data and harmful-tuning data come from different splits of the same dataset (BeaverTails), which may introduce methodological dependencies.
- It remains unexplored whether different geometric structures might emerge in much larger models (e.g., 70B, 405B).
- The paper primarily presents negative results without proposing an alternative defense mechanism.

## Related Work & Insights
- **vs. SafeLoRA**: SafeLoRA assumes safety information is concentrated in specific directions of LoRA updates. This paper's conclusion directly challenges the foundation of that assumption.
- **vs. LDIFS**: LDIFS defends by identifying "safety directions" and restricting fine-tuning updates. This paper proves these "safety directions" are also "useful directions," meaning restricting them incurs a proportional loss in task performance.
- **vs. Refusal Direction Research**: Some works found that removing specific directions can eliminate refusal behavior. This paper's findings are consistent but go deeper—eliminating these directions simultaneously removes useful behaviors because both share the same high-impact directions.

## Rating
- Novelty: ⭐⭐⭐⭐ Rigorously tested a widely used but unverified assumption.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 models, 4 experimental perspectives, multiple controls, and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, tight logic, and accurate conclusions.
- Value: ⭐⭐⭐⭐⭐ Fundamentally challenges the subspace defense paradigm in LLM safety, with significant impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Anchored Supervised Fine-Tuning](anchored_supervised_fine-tuning.md)
- [\[ACL 2026\] Why Supervised Fine-Tuning Fails to Learn: A Systematic Study of Incomplete Learning in Large Language Models](../../ACL2026/llm_alignment/why_supervised_fine-tuning_fails_to_learn_a_systematic_study_of_incomplete_learn.md)
- [\[ICML 2026\] Safety Anchor: Defending Harmful Fine-tuning via Geometric Bottlenecks](../../ICML2026/llm_alignment/safety_anchor_defending_harmful_fine-tuning_via_geometric_bottlenecks.md)
- [\[ICLR 2026\] TS²: Sparsemax+ for Training and Softmax for Testing for Accurate and Diverse LLM Fine-tuning](ts2_training_with_sparsemax_testing_with_softmax_for_accurate_and_diverse_llm_fi.md)
- [\[ICLR 2026\] Pretrain Value, Not Reward: Decoupled Value Policy Optimization](pretrain_value_not_reward_decoupled_value_policy_optimization.md)

</div>

<!-- RELATED:END -->
