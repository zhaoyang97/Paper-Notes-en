---
title: >-
  [Paper Note] Landscape of Thoughts: Visualizing the Reasoning Process of Large Language Models
description: >-
  [ICLR 2026][Model Compression][LLM reasoning visualization] This paper proposes Landscape of Thoughts (LoT), the first tool to visualize LLM reasoning trajectories as two-dimensional terrain maps. By encoding intermediat…
tags:
  - "ICLR 2026"
  - "Model Compression"
  - "LLM reasoning visualization"
  - "reasoning trajectory analysis"
  - "t-SNE"
  - "test-time scaling"
  - "lightweight verifier"
date: 2026-05-08
content_hash: 9606f5ff3d2e374b
---

# Landscape of Thoughts: Visualizing the Reasoning Process of Large Language Models

**Conference**: ICLR 2026
**arXiv**: [2503.22165](https://arxiv.org/abs/2503.22165)  
**Code**: [GitHub](https://github.com/tmlr-group/landscape-of-thoughts)  
**Area**: Model Compression
**Keywords**: LLM reasoning visualization, reasoning trajectory analysis, t-SNE, test-time scaling, lightweight verifier

## TL;DR
This paper proposes Landscape of Thoughts (LoT), the first tool to visualize LLM reasoning trajectories as two-dimensional terrain maps. By encoding intermediate states via perplexity-based features and projecting them with t-SNE, LoT reveals reasoning behavior patterns and can be adapted as a lightweight verifier to improve reasoning accuracy and test-time scaling.

## Background & Motivation
Step-by-step reasoning in LLMs is widely applied in agentic settings, yet the reasoning behavior itself remains poorly understood. Existing analysis methods either rely on specific decoders/tasks or require manual inspection of individual reasoning trajectories—a process that neither scales (100 trajectories require ~50 minutes) nor supports dataset-level aggregation. This hinders model development, reasoning research, and safety monitoring.

The root cause is the absence of a general, automated, and scalable tool capable of analyzing LLM reasoning trajectories from the level of individual samples to entire datasets. The core idea of LoT is to represent each intermediate state during reasoning as a feature vector encoding "distances" to each candidate answer, then project these vectors into 2D space via t-SNE to form a "thought landscape" that intuitively depicts reasoning convergence patterns.

## Method

### Overall Architecture
LoT operates as a post-hoc analysis tool without interfering with the model's reasoning process. Given a multiple-choice dataset, after the LLM generates reasoning trajectories, LoT encodes textual states into numerical features and analyzes reasoning behavior through qualitative visualization (terrain maps) and quantitative metrics (consistency, uncertainty, and perplexity).

### Key Designs
1. **State Featurization**:

    - Function: Encode each intermediate state in the reasoning trajectory as a $k$-dimensional feature vector.
    - Mechanism: The LLM itself is used to estimate the distance from each state to each candidate answer. For state $s_i$, the perplexity with respect to each option $c_j$ is computed as $d(s_i, c_j) = \text{PPL}(c_j | s_i)$, and the normalized result forms feature $\bm{f}_i$.
    - Design Motivation: Perplexity naturally reflects model confidence toward a given answer, and token-length normalization ensures comparability across options of varying lengths.

2. **Terrain Map Visualization**:

    - Function: Project all state features and option landmarks into a 2D space.
    - Mechanism: A feature matrix $\bm{F} \in \mathbb{R}^{k \times (rn+k)}$ is constructed from all trajectory states and option landmark features, then projected via t-SNE into two dimensions. States are color-coded by correctness, and density maps display the distribution of states across reasoning stages.
    - Design Motivation: t-SNE excels at preserving local neighborhood structure, enabling visualization of convergence trends in the distance-based feature space.

3. **Quantitative Metric System**:

    - Consistency: Whether the optimal choice at an intermediate state matches that of the final state, $\text{Consistency}(s_i) = \mathbb{1}(\arg\min \bm{f}_i = \arg\min \bm{f}_n)$.
    - Uncertainty: The entropy of the feature vector, reflecting model confidence at intermediate steps.
    - Perplexity: Thought-level perplexity measuring model confidence in its generated reasoning steps.

### Lightweight Verifier
Based on observed differences in convergence speed and consistency, a random forest classifier $g$ is trained to predict trajectory correctness. The input consists of state features and consistency metrics; the output is a correct/incorrect label. Weighted majority voting replaces simple voting to select the final answer.

## Key Experimental Results

### Main Results

| Model / Method | AQuA (Acc%) | MMLU | CommonsensQA | StrategyQA |
|---|---|---|---|---|
| Llama-1B (CoT, no verifier) | 15.8 | - | - | - |
| Llama-3B (CoT, no verifier) | 42.0 | - | - | - |
| Llama-70B (CoT, no verifier) | 84.4 | 80.2 | 75.8 | 64.8 |
| + Verifier (10 trajectories) | consistent gain | consistent gain | consistent gain | consistent gain |
| Verifier (50 trajectories) | >65% | - | - | - |

### Ablation Study / Transferability

| Train → Test | ΔAcc | Notes |
|---|---|---|
| AQuA → StrategyQA | +4.5% | Positive cross-dataset transfer |
| 70B → 3B | +5.5% | Positive cross-scale transfer |
| 1B → 70B | positive | Small-model training transfers to large models |

### Key Findings
- Larger models exhibit faster convergence, higher consistency, and lower uncertainty and perplexity in their reasoning trajectories.
- Incorrect trajectories converge to wrong answers earlier than correct trajectories converge to correct ones, enabling early error detection.
- Intermediate-state consistency is generally low, revealing the instability inherent in the reasoning process.
- The verifier with 50 trajectories significantly outperforms baseline voting (>65% vs. ~30%), demonstrating strong test-time scaling.

## Highlights & Insights
- The approach of reformulating reasoning behavior analysis as a visualization problem is novel, analogous to the contribution of t-SNE to high-dimensional data analysis.
- The state featurization design is elegant: perplexity serves as a bridge connecting the text space to a numerical feature space.
- The lightweight verifier does not rely on pretrained language models; a random forest suffices to effectively distinguish correct from incorrect trajectories.
- The possibility of cross-model and cross-dataset transfer opens a direction toward general-purpose reasoning monitoring.

## Limitations & Future Work
- The framework is restricted to multiple-choice formats; open-ended tasks require new featurization schemes.
- Likelihood estimation depends on open-source LLMs and is inapplicable to closed-source models.
- Cross-dataset transfer is not always positive, and the transferability of features warrants further improvement.
- t-SNE projection may discard part of the structural information.

## Related Work & Insights
- **vs. Manual text inspection**: LoT provides automated and scalable analysis, avoiding subjective bias.
- **vs. Metric-only analysis**: Combining qualitative terrain maps with quantitative indicators reveals patterns invisible to either approach alone.
- **vs. LLM-based verifiers**: LoT is lightweight and fast, requiring no additional language model.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First reasoning trajectory visualization tool; a pioneering perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across multiple models, methods, and datasets, but lacks direct comparison with LLM-based verifiers.
- Writing Quality: ⭐⭐⭐⭐⭐ Polished figures and well-organized observations.
- Value: ⭐⭐⭐⭐ Practically valuable for reasoning research and safety monitoring; the verifier's practical utility is somewhat limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] InftyThink: Breaking the Length Limits of Long-Context Reasoning in Large Language Models](inftythink_breaking_the_length_limits_of_long-context_reasoning_in_large_languag.md)
- [\[ICLR 2026\] Scaling Reasoning Hop Exposes Weaknesses: Demystifying and Improving Hop Generalization in Large Language Models](scaling_reasoning_hop_exposes_weaknesses_demystifying_and_improving_hop_generali.md)
- [\[ICLR 2026\] BeyondBench: Contamination-Resistant Evaluation of Reasoning in Language Models](beyondbench_contamination-resistant_evaluation_of_reasoning_in_language_models.md)
- [\[ACL 2026\] SeLaR: Selective Latent Reasoning in Large Language Models](../../ACL2026/model_compression/selar_selective_latent_reasoning_in_large_language_models.md)
- [\[AAAI 2026\] Efficient Reasoning for Large Reasoning Language Models via Certainty-Guided Reflection Suppression](../../AAAI2026/model_compression/efficient_reasoning_for_large_reasoning_language_models_via_certainty-guided_ref.md)

</div>

<!-- RELATED:END -->
