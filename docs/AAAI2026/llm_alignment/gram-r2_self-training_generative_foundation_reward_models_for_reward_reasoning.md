---
title: >-
  [Paper Note] GRAM-R²: Self-Training Generative Foundation Reward Models for Reward Reasoning
description: >-
  [AAAI 2026][LLM Alignment][Reward modeling] This paper proposes GRAM-R², a generative foundation reward model that elicits reward reasoning capabilities on unlabeled data via self-training. The model simultaneously produ…
tags:
  - "AAAI 2026"
  - "LLM Alignment"
  - "Reward modeling"
  - "self-training"
  - "generative reward"
  - "preference reasoning"
  - "RLHF"
date: 2026-05-08
content_hash: a08f030bd96d6fd3
---

# GRAM-R²: Self-Training Generative Foundation Reward Models for Reward Reasoning

**Conference**: AAAI 2026
**arXiv**: [2509.02492](https://arxiv.org/abs/2509.02492)  
**Code**: N/A  
**Area**: LLM Alignment
**Keywords**: Reward modeling, self-training, generative reward, preference reasoning, RLHF

## TL;DR

This paper proposes GRAM-R², a generative foundation reward model that elicits reward reasoning capabilities on unlabeled data via self-training. The model simultaneously produces preference labels and reasoning rationales, consistently outperforming both discriminative and generative baselines across multiple downstream tasks including response ranking, task adaptation, and RLHF.

## Background & Motivation

**Background**: Reward modeling is central to LLM alignment, and the field has seen a clear trend toward generalist reward models that move beyond task-specific designs. Effective reward models must evaluate LLM outputs across diverse tasks and domains, ideally with interpretable justifications for their judgments.

**Limitations of Prior Work**: (1) The fundamental challenge in developing effective reward models lies in heavy reliance on large-scale annotated preference data, which is extremely costly and difficult to scale; (2) while pretraining can leverage abundant unlabeled data, existing pretraining approaches cannot endow reward models with explicit reasoning capabilities—models produce preference judgments without explaining why; (3) discriminative reward models output only scalar scores and lack interpretability, whereas generative reward models can produce rationales but require substantial training data with reasoning annotations.

**Key Challenge**: A fundamental tension exists between data scarcity and the demand for reasoning—the model must learn from limited (or zero) labeled data while still producing preference judgments supported by reasoning chains.

**Goal**: (1) Elicit reward reasoning capabilities from unlabeled data via self-training; (2) construct a generative reward model that serves as a foundation model; (3) minimize or eliminate dependence on annotated preference data.

**Key Insight**: The authors adopt a self-training paradigm in which the model generates pseudo-labels and reasoning rationales, then continues training on these self-generated data, forming an iteratively reinforcing cycle.

**Core Idea**: Self-training is used to bootstrap a generative reward model on unlabeled data, simultaneously generating preference labels and reward rationales, yielding a foundation reward model with inherent reasoning capabilities.

## Method

### Overall Architecture

The GRAM-R² training pipeline proceeds as follows: (1) starting from a pretrained LLM, a small amount of seed preference data initializes the reward reasoning capability; (2) on large-scale unlabeled data, the model itself generates preference labels and reasoning rationales; (3) high-quality self-generated data are filtered for further training; (4) multiple rounds of iteration progressively strengthen reasoning capability. Once trained, GRAM-R² can be directly applied to various downstream tasks—response ranking, reward fine-tuning, and RLHF—with no or minimal additional adaptation.

### Key Designs

1. **Self-Training for Reward Reasoning**:

    - **Function**: Bootstraps reward reasoning capabilities from unlabeled data.
    - **Mechanism**: Given a response pair (response A vs. response B), the model generates: (a) a preference label (A is better / B is better / tie); and (b) a reasoning rationale explaining the judgment. During self-training, the current model performs inference on unlabeled data, high-confidence outputs are filtered as pseudo training data, and the model is further trained on these data. Crucially, rationales and labels are generated jointly and used jointly for training.
    - **Design Motivation**: Self-training on labels alone is prone to confirmation bias, reinforcing the model's own errors. Requiring simultaneous rationale generation enriches the self-supervision signal—the model must not only judge correctness but also articulate reasons, compelling deeper reasoning.

2. **Quality Filtering and Curriculum Strategy**:

    - **Function**: Ensures the quality of self-generated pseudo training data and prevents noise accumulation.
    - **Mechanism**: Self-generated (label, rationale) pairs are filtered along multiple dimensions: (a) label confidence filtering—retaining only judgments in which the model is highly certain; (b) consistency checking—sampling the same response pair multiple times and retaining only consistent judgments; (c) rationale quality assessment—verifying logical coherence and consistency between the rationale and its conclusion. A curriculum strategy is adopted, applying stricter filtering in early rounds to prevent early-stage noise accumulation.
    - **Design Motivation**: The central risk of self-training is "garbage in, garbage out"—poor-quality pseudo-labels cause training to progressively diverge. Multi-dimensional filtering and the curriculum strategy serve as essential quality guarantees.

3. **Foundation Model Architecture and Multi-Task Adaptation**:

    - **Function**: Enables the model to serve as a generalizable reward foundation model.
    - **Mechanism**: GRAM-R² is built upon a pretrained LLM and outputs preference judgments and rationales in a generative manner rather than by regressing scalar scores. This generative design allows the model to naturally accommodate diverse downstream formats—ranking tasks require only comparing preference outputs across multiple responses, scoring tasks can extract scores from rationales, and RLHF can convert preference judgments into reward signals.
    - **Design Motivation**: Discriminative reward models that output scalar scores have limited flexibility and cannot provide reasoning rationales. The generative design unifies judgment and explanation, and is naturally suited to multi-task use.

### Loss & Training

Training uses the standard autoregressive language modeling objective:

$$\mathcal{L} = -\sum_t \log p_\theta(y_t | y_{<t}, x)$$

where $y$ is the sequence of (rationale + label) and $x$ is the input response pair. Self-training proceeds through multiple rounds of iteration; in each round, the current model generates pseudo-data on unlabeled examples, the data are filtered, and the retained examples are added to the training set.

## Key Experimental Results

### Main Results

Evaluation is conducted on three tasks: response ranking, task adaptation, and RLHF.

| Task | Metric | GRAM-R² | Discriminative Baseline | Generative Baseline | Notes |
|------|--------|---------|------------------------|---------------------|-------|
| Response Ranking | Ranking Accuracy | Best | Moderate | Second best | Consistently superior |
| Task Adaptation | Post-adaptation Accuracy | Best | -- | Second best | Few-shot adaptation |
| RLHF | Downstream LLM Quality | Best | Moderate | Second best | Better alignment |

### Ablation Study

| Configuration | Performance | Notes |
|--------------|-------------|-------|
| Full self-training | Best | Joint rationale + label self-training |
| Label-only self-training | Degraded | Lacks supervision signal from reasoning rationales |
| No quality filtering | Significantly degraded | Noisy pseudo-label accumulation |
| Single-round self-training | Inferior to multi-round | Multi-round iteration yields progressive improvement |
| No curriculum strategy | Degraded | Severe early-stage noise contamination |

### Key Findings

- The combination of self-training and rationale generation is central to GRAM-R²'s success—self-training on labels alone yields limited gains, while jointly generating rationales substantially enhances reasoning capability.
- Quality filtering is critical—self-training without filtering degrades model performance.
- As a foundation reward model, GRAM-R² achieves strong cross-task performance in zero-shot or few-shot settings, validating the effectiveness of the foundation model paradigm.
- The generative design offers clear advantages over discriminative approaches in terms of interpretability and flexibility.

## Highlights & Insights

- The conceptualization of **"reward reasoning"** is forward-looking—going beyond preference judgments to require explanations of the reasoning behind them, which is essential for building trustworthy AI alignment systems.
- The methodology of **eliciting reasoning via self-training** generalizes to other tasks requiring reasoning capabilities, enabling reasoning to be learned without large quantities of annotated reasoning chain data.
- The foundation model design allows GRAM-R² to be broadly reused, lowering the barrier to reward modeling.

## Limitations & Future Work

- The quality of self-training depends on the choice of seed data and the capability of the initial model.
- Multi-round self-training incurs substantial computational cost, as inference must be performed on large volumes of unlabeled data in each round.
- The true quality of self-generated rationales is difficult to assess objectively—it remains unclear whether the model is genuinely reasoning or producing post-hoc rationalizations.
- Integration with Constitutional AI could be explored to constrain rationale content using explicit principles.

## Related Work & Insights

- **vs. Traditional Discriminative Reward Models (e.g., Bradley-Terry)**: Discriminative models output only scores without explanation; GRAM-R² provides reasoning rationales, offering greater transparency.
- **vs. LLM-as-Judge**: LLM-as-Judge employs general-purpose LLMs as evaluators but lacks dedicated reward reasoning training. GRAM-R² explicitly trains for reasoning capability.
- **vs. Self-Training Methods (e.g., Self-Play)**: Self-training is common in policy optimization; this paper innovatively applies it to eliciting reasoning capabilities in reward models.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of self-training and reward reasoning is novel; the foundation reward model concept is forward-looking.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across three task dimensions with ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear; methodological exposition is logically rigorous.
- Value: ⭐⭐⭐⭐⭐ Represents an important methodological contribution to RLHF and AI alignment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ConsistRM: Improving Generative Reward Models via Consistency-Aware Self-Training](../../ACL2026/llm_alignment/consistrm_improving_generative_reward_models_via_consistency-aware_self-training.md)
- [\[ICML 2026\] Mitigating Reward Hacking in RLHF via Bayesian Non-negative Reward Modeling](../../ICML2026/llm_alignment/mitigating_reward_hacking_in_rlhf_via_bayesian_non-negative_reward_modeling.md)
- [\[AAAI 2026\] When Human Preferences Flip: An Instance-Dependent Robust Loss for RLHF](when_human_preferences_flip_an_instance-dependent_robust_loss_for_rlhf.md)
- [\[AAAI 2026\] DeCoRL: Decoupling Reasoning Chains via Parallel Sub-Step Generation and Cascaded Reinforcement for Interpretable and Scalable RLHF](decorl_decoupling_reasoning_chains_via_parallel_sub-step_gen.md)
- [\[CVPR 2026\] MapReduce LoRA: Advancing the Pareto Front in Multi-Preference Optimization for Generative Models](../../CVPR2026/llm_alignment/mapreduce_lora_advancing_the_pareto_front_in_multi-preference_optimization_for_g.md)

</div>

<!-- RELATED:END -->
