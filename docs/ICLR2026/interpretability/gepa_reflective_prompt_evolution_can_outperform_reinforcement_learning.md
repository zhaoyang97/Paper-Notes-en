---
title: >-
  [Paper Note] GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning
description: >-
  [ICLR 2026 (Oral)][Interpretability][Prompt Optimization] This paper proposes GEPA (Genetic-Pareto), a prompt optimizer that diagnoses failure modes from a small number of execution trajectories via natural language refl…
tags:
  - "ICLR 2026 (Oral)"
  - "Interpretability"
  - "Prompt Optimization"
  - "Evolutionary Search"
  - "Natural Language Reflection"
  - "Pareto Front"
  - "GRPO Alternative"
date: 2026-05-08
content_hash: fdda59ce7940708a
---

# GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning

**Conference**: ICLR 2026 (Oral)
**arXiv**: [2507.19457](https://arxiv.org/abs/2507.19457)
**Code**: [https://github.com/gepa-ai/gepa](https://github.com/gepa-ai/gepa)
**Area**: Interpretability
**Keywords**: Prompt Optimization, Evolutionary Search, Natural Language Reflection, Pareto Front, GRPO Alternative

## TL;DR
This paper proposes GEPA (Genetic-Pareto), a prompt optimizer that diagnoses failure modes from a small number of execution trajectories via natural language reflection and iteratively refines prompts. GEPA outperforms GRPO by an average of 6% (up to 20%) across six tasks while using only 1/35 of the sampling budget.

## Background & Motivation
Large language models are increasingly adapted to downstream tasks via reinforcement learning methods such as GRPO. However, such approaches typically require thousands of rollouts and compress rich execution trajectories into sparse scalar reward signals, discarding a substantial amount of information.

Language itself is a highly interpretable medium that inherently contains far richer learning signals than scalar rewards. The reasoning chains, tool-call traces, and error messages produced during LLM inference embed diagnostic clues about "why a failure occurred," yet RL methods discard all of this, retaining only a single score.

**Key Challenge**: RL methods (e.g., GRPO) require a large number of rollouts yet exploit only scalar rewards, whereas natural language carries learning signals far richer than any scalar signal.

**Key Insight**: Given that LLMs can comprehend execution trajectories, why not allow an LLM to reflect directly on failure causes and propose improvements, thereby achieving efficient optimization with minimal sampling?

**Core Idea**: Prompt optimization is framed as an evolutionary search process augmented with reflection. An LLM reads complete execution trajectories to perform "gradient-equivalent" diagnosis and repair, while diversity is maintained through Pareto front selection.

## Method
GEPA is a general-purpose text-parameter optimization framework whose core loop consists of: Select → Execute → Reflect → Mutate → Accept.

### Overall Architecture
Given an AI system containing one or more LLM prompts, GEPA proceeds as follows:
1. Select a candidate prompt from the Pareto front.
2. Execute it on a mini-batch, capturing complete execution trajectories (reasoning traces, tool calls, outputs, and error messages).
3. An LLM reflector reads the trajectories and diagnoses failure causes in natural language.
4. Based on the diagnosis and lessons accumulated from all ancestors, an improved candidate prompt is generated.
5. If performance improves, the candidate is added to the pool and the Pareto front is updated.

### Key Designs
1. **Actionable Side Information (ASI)**: Diagnostic feedback returned by the evaluator; the central innovation of GEPA. It serves as the "gradient" in text optimization—conveying not only the score but also the reason for that score (e.g., error messages, performance profiles, reasoning logs). This enables GEPA to extract rich learning signals from very few samples.

2. **Pareto Front Selection**: A set of candidate prompts that are optimal on different task subsets is maintained. This prevents overfitting to a single metric and preserves diversity in the search space. Candidates are sampled from the Pareto front to ensure that variants excelling on particular sub-tasks are not discarded.

3. **System-aware Merge**: Two Pareto-optimal candidates, each strong on different tasks, are merged to complement each other's strengths. The LLM analyzes the reasons for each candidate's advantages and generates a new candidate that integrates both.

4. **Reflective Mutation**: Mutations are not random but directed by diagnosis. The LLM first reads failure trajectories to diagnose "why this prompt fails on this class of problems" and then modifies the prompt in a targeted manner. This is the fundamental reason GEPA is far more sample-efficient than RL.

### Loss & Training
GEPA uses no gradients or loss functions; acceptance of a new candidate is determined solely by improvement on the evaluation metric. Typical configurations require 100–500 evaluations, compared to 5,000–25,000+ rollouts for GRPO.
The acceptance criterion is configurable: the default accepts any metric improvement, but thresholds or statistical significance requirements can also be imposed.
The entire optimization process is fully gradient-free—no access to model weights is required, only API call capability.
This means GEPA can optimize any API-only model (GPT-5, Claude, Gemini), which is beyond the reach of RL-based methods.

## Key Experimental Results

### Main Results

| Task | Metric | GEPA | GRPO | MIPROv2 | Gain (vs GRPO) |
|------|--------|------|------|---------|----------------|
| Average over 6 tasks | Accuracy | — | — | — | +6% avg, up to +20% |
| AIME-2025 | Accuracy | — | — | — | +12% (vs MIPROv2) |
| GPT-4.1 Mini + AIME | Accuracy | 56.6% | — | 46.6% | +10 pp |
| DSPy MATH | Accuracy | 93% | — | 67% | — |
| ARC-AGI | Accuracy | 89% | — | 32% | — |

### Ablation Study

| Configuration | Key Metric | Note |
|---------------|-----------|------|
| Full GEPA | Best | Reflection + Pareto + Merge all enabled |
| w/o Reflection | Significant drop | Degenerates to random search |
| w/o Pareto Selection | Loss of diversity | Prone to local optima |
| w/o System-aware Merge | Moderate drop | Cannot integrate complementary sub-task strengths |

### Key Findings
- GEPA uses only 1/35 the rollouts of GRPO yet achieves 6% higher average performance.
- GEPA surpasses the leading prompt optimizer MIPROv2 by up to 12% on AIME-2025.
- The optimized prompts are human-readable and contain detailed problem-solving strategies.
- GEPA also demonstrates potential as an inference-time search strategy for code optimization.
- The framework has been integrated into major platforms including DSPy, MLflow, OpenAI Cookbook, Google ADK, and HuggingFace.

## Highlights & Insights
- Replacing scalar rewards with natural language reflection represents a profound reconceptualization of the RL paradigm—language itself constitutes the most informative gradient signal.
- The low sample requirement (100–500 evaluations) enables optimization of API-only models (GPT-5, Claude) without weight access.
- The generated prompts are interpretable "pre-computed reasoning plans" that can be directly inspected and understood.
- Pareto front maintenance is an elegant mechanism for avoiding overfitting.

## Limitations & Future Work
- Depends on high-quality reflection models (typically GPT-5-class), which incurs non-trivial cost.
- For tasks requiring large-scale weight updates (e.g., knowledge injection), prompt optimization faces inherent performance ceilings.
- The stochastic nature of the search process can lead to substantial variance across runs.
- Fair comparison with RL remains debatable, as the two approaches optimize different objects (prompts vs. weights).
- For very long prompts (thousands of tokens), the quality of reflection and mutation may degrade.
- The design of the evaluation metric critically affects outcomes; a poorly designed metric renders GEPA ineffective.
- For tasks such as safety alignment that require precise control over internal model representations, the limitations of prompt optimization are more pronounced.

## Related Work & Insights
- **vs GRPO/PPO**: GRPO optimizes model weights via policy gradients, requiring extensive rollouts; GEPA optimizes prompt text, substituting reflection for gradient computation.
- **vs MIPROv2**: Previously the strongest prompt optimizer; GEPA surpasses it by 10%+ on AIME and other tasks through ASI and Pareto-based search.
- **vs TextGrad**: TextGrad also employs textual feedback but simulates gradients; GEPA's evolutionary search combined with reflection is more efficient.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — The paradigm of replacing RL scalar rewards with natural language reflection is highly inspiring and fully merits an ICLR Oral designation.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Validated across six tasks with thorough comparisons against GRPO and MIPROv2.
- Writing Quality: ⭐⭐⭐⭐⭐ — Motivation is articulated with exceptional clarity; the method is intuitively accessible.
- Value: ⭐⭐⭐⭐⭐ — Already widely adopted in industry (Shopify, Databricks, OpenAI, etc.), with substantial real-world impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Evolution of Concepts in Language Model Pre-Training](evolution_of_concepts_in_language_model_pre-training.md)
- [\[NeurIPS 2025\] Beyond Accuracy: Dissecting Mathematical Reasoning for LLMs Under Reinforcement Learning](../../NeurIPS2025/interpretability/beyond_accuracy_dissecting_mathematical_reasoning_for_llms_u.md)
- [\[ICLR 2026\] Exploring Interpretability for Visual Prompt Tuning with Cross-layer Concepts](exploring_interpretability_for_visual_prompt_tuning_with_cross-layer_concepts.md)
- [\[ICLR 2026\] Auditing Cascading Risks in Multi-Agent Systems via Semantic–Geometric Co-evolution](auditing_cascading_risks_in_multi-agent_systems_via_semanti-geometric_co-evolut.md)
- [\[CVPR 2026\] Language Models Can Explain Visual Features via Steering](../../CVPR2026/interpretability/language_models_can_explain_visual_features_via_steering.md)

</div>

<!-- RELATED:END -->
