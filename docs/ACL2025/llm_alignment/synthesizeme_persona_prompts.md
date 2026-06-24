---
title: >-
  [Paper Note] SynthesizeMe! Inducing Persona-Guided Prompts for Personalized Reward Models in LLMs
description: >-
  [LLM Alignment] Introduces the SynthesizeMe method, which automatically reasons and synthesizes user personas from a limited number of pairwise preference interactions to construct interpretable and transferable personalized prompts, significantly improving personalized preference prediction accuracy on PersonalRewardBench.
tags:
  - "LLM Alignment"
date: 2026-05-08
content_hash: 494079628c15a020
---

# SynthesizeMe! Inducing Persona-Guided Prompts for Personalized Reward Models in LLMs

| Info | Content |
|------|------|
| Conference | ACL 2025 |
| arXiv | [2506.05598](https://arxiv.org/abs/2506.05598) |
| Code | - |
| Area | LLM Alignment |
| Keywords | Personalized Reward Models, Persona, Pluralistic Alignment, LLM-as-a-Judge, Preference Learning |

## TL;DR

Introduces the SynthesizeMe method, which automatically reasons and synthesizes user personas from a limited number of pairwise preference interactions to construct interpretable and transferable personalized prompts, significantly improving personalized preference prediction accuracy on PersonalRewardBench.

## Background & Motivation

**Core Problem:** How to construct personalized reward models to capture the pluralistic preferences of different users using only a small amount of pairwise preference feedback (5-15 pairs)?

**Limitations of Prior Work:** Mainstream LLM alignment approaches assume homogeneous preferences, whereas real-world user preferences are highly diverse due to factors like culture, values, and style. Existing personalization methods (e.g., Rewarded Soups, P-Soups) rely on predefined preference dimensions and fail to capture open-ended preference spaces.

**Key Challenge:** (1) **Data Sparsity** — each user has only a very small amount of preference data; (2) **Preference Attribution** — pairwise preferences are noisy/fuzzy observations of true user preferences, making it difficult to pinpoint the actual reasons behind user choices; (3) **Overfitting** — extremely limited data easily leads to overfitting to specific preference patterns.

## Method

### Overall Architecture

SynthesizeMe is a three-step pipeline method that takes a user's small set of pairwise preferences as input and outputs a personalized prompt in natural language (comprising a persona + informative exemplars):

1. **Bootstrap Reasoning** → 2. **Synthesize Persona** → 3. **Extract Informative Examples**

### Key Designs

**Step 1 — Bootstrap Reasoning:** Without any prior user information, the LLM is prompted to perform speculative reasoning using CoT for each preference, explaining which response the user might prefer and why. Only correctly reasoned samples are retained. Through $n=10$ random subset samplings and validation set filtering, the optimal set of reasoning is selected:

$$\mathop{\arg\max}_{i \in \{1,\dots,n\}} \text{Eval}(\text{Bootstrap}(\mathcal{D}_u^{\text{train}}, \varnothing)_i, \mathcal{D}_u^{\text{val}})$$

**Step 2 — Synthesize Persona:** Taking the validated reasoning as context, the LLM is guided to synthesize a user persona $\pi$. The persona-generation prompt $\Theta$ is optimized using DSPy MIPROv2 on the PRISM dataset, and the optimized $\Theta$ is found to transfer well to other datasets such as Chatbot Arena.

**Step 3 — Extract Informative Examples:** Using persona $\pi$ as context, a second round of bootstrapping is performed. Through $m=10$ trials, the exemplars that best represent user preferences are selected. These exemplars are combined with the persona to form the final personalized prompt.

### Core Advantages (Comparison with Existing Methods)

| Method | Unconstrained Preferences | Adaptation Mode | Personalization Mechanism |
|------|-----------|---------|-----------|
| Rewarded Soups | ✗ | Fine-tuning | Weight Interpolation |
| P-Soups | ✗ | Fine-tuning | Merging Reward Models |
| GPO | ✓ | Fine-tuning | Few-shot Group Embedding |
| VPL | ✓ | Fine-tuning | Latent User Embedding |
| PAL | ✓ | Fine-tuning | Prototypical Preference Groups |
| **SynthesizeMe** | **✓** | **In-Context** | **Bootstrap Reasoning + Persona** |

Key advantages of SynthesizeMe: (1) No need for predefined preference dimensions; (2) No fine-tuning required, fully in-context; (3) Generates interpretable natural language prompts; (4) Transferable across models.

## Experiments

### PersonalRewardBench Benchmark Construction

High-quality, highly controversial, and personalizable user preference data were filtered from Chatbot Arena (131 users) and PRISM (723 users). The benchmark was constructed through a three-stage filtering process (User Filtering → Personalization Filtering → Quality/Consensus Filtering).

### Main Results (Chatbot Arena, Llama 3.3 70B)

| Method | Accuracy |
|------|--------|
| Default LLM Judge | 56.69% |
| Memory | 57.57% |
| GPO | 58.10% |
| **SM: Personas + Demos** | **61.97%** |
| Bradley-Terry RM (Fine-tuned) | 71.48% |
| **FT RM + Personas** | **72.18%** |

### Ablation Study

| Configuration (Llama 70B, Chatbot Arena) | Accuracy |
|----------------------------------|--------|
| Just Demos | 61.97% |
| Just Personas | 53.70% |
| Personas + Demos | 61.97% |
| Personas + Distill Θ | — |
| Personas + Demos + Distill Θ | — |

### Key Findings

- **SynthesizeMe improves performance by up to 4.4% in the LLM-as-a-Judge setting**, achieving the best performance among all in-context methods.
- **Exemplars (Demos) are key to personalization**: configurations including demos win across all six settings.
- **Interaction history outperforms demographics**: SynthesizeMe outperforms the demographics baseline by 3.87% (Llama 70B, PRISM).
- **Learned personas match ground-truth user preferences**: The alignment rate between personas synthesized by the 70B model and the true preferences of PRISM users is significantly higher than that of random pairing (56.1% vs 47%, $p < 0.05$).
- **Each additional preference data point yields an accuracy boost of approximately 0.8%**, and as few as 5 context preferences can outperform non-personalized baselines.
- **Persona-generating prompts are transferable across models**: The optimized $\Theta$ on 70B is effective for 3B and 8B models.

## Highlights & Insights

- Proposes a fully in-context, fine-tuning-free personalized reward modeling scheme that generates interpretable and transferable natural language personas.
- Constructs PersonalRewardBench, systematically comparing multiple personalized reward models under the same benchmark for the first time.
- The persona-synthesizing prompt is transferable across datasets and model families, demonstrating high practical value.
- Cleverly leverages a validation set for reasoning quality filtering, effectively addressing the challenge of extreme data sparsity.

## Limitations & Future Work

- The performance gain on fine-tuned reward models is limited (falling within the confidence interval), making it mostly recommended for LLM-as-a-Judge scenarios.
- Persona synthesis relies heavily on the LLM's reasoning capabilities; the quality of personas synthesized by smaller models (3B) is significantly worse.
- The user scale in PersonalRewardBench is still limited (only 131 users in Chatbot Arena), which may not be fully representative.
- Dynamic persona updating mechanisms in multi-turn interactions have not yet been explored.

## Related Work & Insights

- **Personalized Reward Models:** GPO (Group Preference Optimization), VPL (Variational Preference Learning), and PAL (Pluralistic Alignment Framework) achieve personalization through embeddings or group learning.
- **LLM Personalization:** Includes content personalization (knowledge, opinions, values) and presentation personalization (style, format, verbosity).
- **Prompt Optimization:** The DSPy MIPROv2 optimizer is utilized to automatically rewrite persona-generating instructions.
- **Guided Profile Generation (GPG):** Most similar in concept but operates within a restricted preference space.

## Rating

| Dimension | Rating |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Overall Score | 8/10 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Dynamic Scaling of Unit Tests for Code Reward Modeling](dynamic_scaling_of_unit_tests_for_code_reward_modeling.md)
- [\[ACL 2026\] PERSA: Reinforcement Learning for Professor-Style Personalized Feedback with LLMs](../../ACL2026/llm_alignment/persa_reinforcement_learning_for_professor-style_personalized_feedback_with_llms.md)
- [\[ACL 2025\] Think&Cite: Improving Attributed Text Generation with Self-Guided Tree Search and Progress Reward Modeling](think_cite_attributed_text_gen.md)
- [\[ICLR 2026\] No Prompt Left Behind: Exploiting Zero-Variance Prompts in LLM Reinforcement Learning via Entropy-Guided Advantage Shaping](../../ICLR2026/llm_alignment/no_prompt_left_behind_exploiting_zero-variance_prompts_in_llm_reinforcement_lear.md)
- [\[ICLR 2026\] Sysformer: Safeguarding Frozen Large Language Models with Adaptive System Prompts](../../ICLR2026/llm_alignment/sysformer_safeguarding_frozen_large_language_models_with_adaptive_system_prompts.md)

</div>

<!-- RELATED:END -->
