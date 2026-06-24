---
title: >-
  [Paper Note] BRIDGE: Bootstrapping Text to Control Time-Series Generation via Multi-Agent Iterative Optimization and Diffusion Modeling
description: >-
  [ICML 2025][Image Generation][Text-controlled time-series generation] This paper proposes the Bridge framework, which generates high-quality text-to-time-series paired data using an LLM multi-agent system and utilizes a hybrid prompt of semantic prototypes and textual descriptions to drive a diffusion model. It achieves cross-domain, instance-level text-controlled time-series generation (TC-TSG), ranking SOTA in 11 out of 12 datasets.
tags:
  - "ICML 2025"
  - "Image Generation"
  - "Text-controlled time-series generation"
  - "multi-agent"
  - "diffusion models"
  - "semantic prototype"
  - "cross-domain generalization"
date: 2026-05-08
content_hash: cb88152713455747
---

# BRIDGE: Bootstrapping Text to Control Time-Series Generation via Multi-Agent Iterative Optimization and Diffusion Modeling

**Conference**: ICML 2025  
**arXiv**: [2503.02445](https://arxiv.org/abs/2503.02445)  
**Code**: [Microsoft/TimeCraft](https://github.com/Microsoft/TimeCraft)  
**Area**: Time-Series Generation  
**Keywords**: Text-controlled time-series generation, multi-agent, diffusion models, semantic prototype, cross-domain generalization

## TL;DR

This paper proposes the Bridge framework, which generates high-quality text-to-time-series paired data using an LLM multi-agent system and utilizes a hybrid prompt of semantic prototypes and textual descriptions to drive a diffusion model. It achieves cross-domain, instance-level text-controlled time-series generation (TC-TSG), ranking SOTA in 11 out of 12 datasets.

## Background & Motivation

Time-series generation (TSG) is widely applied in financial simulation, medical data augmentation, and power stress testing. Most existing methods focus on **unconditional single-domain generation**, yet practical applications require satisfying specific constraints, such as generating ECGs that match a patient's profile or heart electrical signals representing specific disease conditions.

Limitations of existing cross-domain methods:
- **Domain-label-based conditional generation** (e.g., TimeVQVAE): Relies on explicit domain labels during training, cannot handle unseen domains, and suffers from low efficiency when the number of labels is massive.
- **Natural-language-based methods** (e.g., GenG): Provides only domain-level descriptions, lacking fine-grained instance-level control.

This work proposes using **natural language text** as control signals to guide TSG, but faces two core challenges:

**Scarcity of high-quality paired text-time-series data**: Existing texts mostly offer only high-level domain descriptions, lacking instance-level details such as trends and fluctuations. Simple rule-based generation (e.g., "upward", "downward") fails to provide substantial improvements.

**Modality gap between text and time series**: Text consists of discrete tokens, while time series are continuous signals. This granularity difference makes text too coarse to precisely capture domain features.

## Method

### Overall Architecture

Bridge consists of two tightly coupled stages:

**Stage 1: Text-Time-Series Data Preparation** (Multi-agent text generation and iterative optimization)  
**Stage 2: Text-Controlled Time-Series Generation** (Hybrid prompt-driven generation based on diffusion models)

### Key Designs

#### 1. Multi-Agent Text Data Preparation System

To address the data scarcity issue, a three-step iterative pipeline is designed:

**Step 1 - Text Template Collection**:
- Adopts a ReAct-style single-agent framework that interacts with external environments (Google, Wikipedia) through dynamic reasoning.
- Decomposes the query into sub-questions and extracts general time-series templates via another LLM after iterative answers.
- Obtains 50 general templates, entirely excluding dataset-specific details via prompting and manual verification.
- During dataset construction, the LLM fills the templates with domain/instance-specific textual descriptions.

**Step 2 - Automated Evaluation**:
- Employs zero-shot time-series forecasting as a proxy evaluation task (avoiding the high cost of retraining the model at each iteration).
- Uses LSTPrompt and LLMTime as the backbone evaluators.
- Core assumption: Higher-quality text leads to better forecasting performance.

**Step 3 - Feedback-Driven Iterative Optimization**:
- Designs a multi-agent collaborative system that simulates the iterative process of a human prompt engineering team.
- **Stage 1 - Task Planning**: Manager Agent orchestrates the workflow and assigns tasks to independent teams.
- **Stage 2 - Intra-group Discussion**: Two independent teams, each consisting of Planner, Scientist, Engineer, and Observer roles, iteratively optimize text through internal loop conversations.
- **Stage 3 - Inter-group Discussion**: Group leaders engage in structured dialogues hosted by the Manager, comparing and integrating results until reaching a consensus.
- **Stage 4 - Post-processing**: Extracts the final templates, deduplicates them, and removes dataset-specific information to form a fixed, general template library.

**Key Points of Data Synthesis**:
- Templates are built using only 2 datasets but are successfully applied to 12 completely disjoint datasets.
- A separate LLM is responsible for extracting statistical information and filling templates, operations are offline and do not rely on external networks.
- The filled text data remains fixed during the TSG phase.

#### 2. Domain-specific Prototype Matching

To compensate for the coarse granularity of textual descriptions, semantic prototypes are introduced as a complementary domain representation:

- Defines a prototype set $\mathcal{P} \in \mathbb{R}^{N_p \times d}$, where each prototype vector $p \in \mathbb{R}^{1 \times d}$ encodes basic features of time series (e.g., trend, seasonality).
- Prototypes act as a cross-domain shared "dictionary", representing different domains through distinct combinations of prototype selections and weights.
- Recommends a **Prototype Assignment Module** to extract domain-specific weights $m$.
- During inference, prototype weights are calculated by extracting prototypes from target domain samples.

**Design Intuition**: Text provides explicit domain information (high-level semantics), while prototypes provide implicit domain features (fine-grained patterns). The two are complementary.

#### 3. Hybrid Prompt Diffusion Generation

Fuses semantic prototypes ($\mathcal{P}$, $m$) with text embedding $l$ to construct a hybrid prompt, which is injected into the diffusion model via cross-attention layers:

- Text $\rightarrow$ Provides explicit semantics, such as trends, statistical features, and domain knowledge.
- Prototype + Weight $\rightarrow$ Complements domain-level shared patterns to enhance cross-domain generalization.
- Hybrid prompts serve as conditional inputs for the diffusion model.

### Loss & Training

Adopts the standard $\epsilon$-parameterized denoising training objective:

$$L = \mathbb{E}_{x_0 \in D^T, \epsilon \sim \mathcal{N}(0, I), n} \left[ \| \epsilon - \epsilon_{\theta, P}(x_n, n, m, l) \|^2 \right]$$

where $n$ is the denoising step, $m$ represents prototype weights, and $l$ denotes text descriptions. Follows the channel-independent setting to handle heterogeneous time series in a univariate manner.

## Key Experimental Results

### Main Results (MDD metric, lower is better)

| Dataset | Bridge | Bridge w/o Text | Bridge w/o Proto | TimeVQVAE | TimeGAN | Gain |
|--------|--------|----------------|-----------------|-----------|---------|---------|
| Electricity | **0.220** | 0.202 | 0.277 | 1.763 | 2.443 | 87.5% vs TimeVQVAE |
| Wind | **0.316** | 0.319 | 0.362 | 0.777 | 1.115 | 59.3% vs TimeVQVAE |
| Traffic | **0.254** | 0.261 | 0.316 | 1.170 | 1.733 | 78.3% vs TimeVQVAE |
| Temperature | **0.342** | 0.345 | 0.408 | 0.943 | 1.164 | 63.7% vs TimeVQVAE |
| NN5 | **0.591** | 0.628 | 0.748 | 1.424 | 2.758 | 58.5% vs TimeVQVAE |
| Fred-MD | **0.258** | 0.271 | 0.359 | 2.932 | 4.028 | 91.2% vs TimeVQVAE |

Bridge achieves optimal MDD in 11 out of 12 datasets, with KL divergence also leading comprehensively.

### Ablation Study

| Configuration | Key Effects | Description |
|------|---------|------|
| w/o Text | MDD generally rises, controllability drops dramatically | Human evaluation HE score drops by 3+ points, confirming text is key to semantic alignment |
| w/o Prototype | MDD rises significantly (e.g., Taxi from 0.386→0.491) | Prototype contributes significantly to domain-level alignment, though less critical than text |
| Prototype Count | 16 is the optimal balance point | More than 16 yields only marginal improvements |
| Multi-agent vs Single-agent | Collaborative strategy consistently achieves lower MAE | Multi-team yields 1.5-6 MAE lower than single-team Macro |
| Refined Text vs Initial Text | MAE decreases by at least 15% | e.g., AirPassenger: 49.36→40.94 |
| Refined Text vs Rule-based Text | Larger gap in effectiveness | e.g., AirPassenger: 52.41→40.94 |

### Key Findings

1. **Text conciseness is preferred over exhaustiveness**: Overly detailed instance descriptions mislead the model; brief high-level descriptions perform better.
2. **Background knowledge significantly boosts performance**: LLM pre-training knowledge provides extra contextual support (MAE increases by 3-8 points w/o Background).
3. **Direct pattern descriptions outperform fine-grained trend decomposition**: STL decomposition + detailed trend description is inferior to directly providing "overall trend + top-k extreme points".
4. **Explicitly specifying sequence length and statistical values stabilizes performance**.
5. **Few-shot cross-domain generalization is effective**: Outperforms all baselines under 5-shot and 10-shot settings, with 10-shot showing steady improvements over 5-shot.

## Highlights & Insights

- **Innovative problem formulation**: Systematically defines and addresses the "Text-Controlled Time-Series Generation" (TC-TSG) task for the first time, extending text-control paradigms from image/video domains to time series.
- **Automated multi-agent data construction**: Obviates the high cost of manual annotation and ensures that templates constructed on only 2 datasets generalize effortlessly to 12 unseen datasets.
- **Exquisite hybrid prompt design**: Combines high-level semantic control from text and fine-grained domain pattern complementation from semantic prototypes, making both components indispensable.
- **Comprehensive evaluation framework**: Employs multiple dimensions including fidelity (MDD/KL), controllability (J-FTSD), and human evaluation (HE-Rank/HE-Mixed) for rigorous validation.
- **Practical insights**: Findings such as the preference for concise text and the significance of domain background knowledge serve as practical guidelines for subsequent text-to-time-series research.

## Limitations & Future Work

1. **High computational cost**: The multi-agent system requires multiple rounds of LLM queries (for template collection, evaluation, and iterative optimization), which incurs high initial, albeit one-time, costs.
2. **Limited to univariate time series**: Employs a channel-independent setting, which does not directly model correlations between multiple variables.
3. **Manual prototype count selection**: Although 16 is empirically determined as the optimal count, an adaptive selection mechanism is absent.
4. **Limited impact of text encoder choice**: Evaluations indicate that larger LLMs only yield marginal improvements, suggesting that the utilization of text features could still be optimized.
5. **Subjectivity in controllability evaluation**: Human evaluation suffers from annotator bias, and J-FTSD might not fully capture precise semantic alignment.

## Related Work & Insights

- **TimeDP** (Huang et al., 2025): Direct inspiration for Bridge's prototype design, though TimeDP uses prototypes for soft prompts without text control.
- **GenG** (Zhou et al., 2024): The first text-to-time-series generation work, limited to specific domains and lacking instance-level control.
- **LSTPrompt / LLMTime**: Zero-shot forecasting methods used as evaluation backbones, demonstrating LLMs' potential in time-series tasks.
- **ReAct** (Yao et al., 2023): Inspired the reasoning-action interaction framework during the template collection stage.
- **Inspirations for multimodal generation**: The text-controlled paradigm can be extended to more time-series scenarios such as personalized medicine and financial simulations.

## Rating

| Dimension | Score | Description |
|------|------|------|
| Novelty | ⭐⭐⭐⭐ | Defines the TC-TSG task systematically for the first time; novel multi-agent data construction and hybrid prompt design. |
| Technical Depth | ⭐⭐⭐⭐ | Comprehensive framework (data preparation + generation stages) with complex but rational multi-agent design. |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ | 12+2 datasets evaluated across multi-dimensional metrics, with comprehensive ablations and in-depth analysis. |
| Value | ⭐⭐⭐⭐ | Strong cross-domain generalization and few-shot capabilities; code is open-source, showing substantial practical potential. |
| Writing Quality | ⭐⭐⭐⭐ | Clear structure, though some multi-agent details are scattered in an extensive appendix. |
| Overall Rating | ⭐⭐⭐⭐ | Solid work with a clear problem definition, well-designed methodologies, and thorough experimentation. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Unified Uncertainty-Aware Diffusion for Multi-Agent Trajectory Modeling](../../CVPR2025/image_generation/unified_uncertainty-aware_diffusion_for_multi-agent_trajectory_modeling.md)
- [\[ICML 2025\] LSCD: Lomb-Scargle Conditioned Diffusion for Time Series Imputation](lscd_lomb-scargle_conditioned_diffusion_for_time_series_imputation.md)
- [\[CVPR 2025\] Compass Control: Multi Object Orientation Control for Text-to-Image Generation](../../CVPR2025/image_generation/compass_control_multi_object_orientation_control_for_text-to-image_generation.md)
- [\[CVPR 2025\] MCCD: Multi-Agent Collaboration-based Compositional Diffusion for Complex Text-to-Image Generation](../../CVPR2025/image_generation/mccd_multi-agent_collaboration-based_compositional_diffusion_for_complex_text-to.md)
- [\[CVPR 2025\] coDrawAgents: A Multi-Agent Dialogue Framework for Compositional Image Generation](../../CVPR2025/image_generation/codrawagents_a_multi-agent_dialogue_framework_for_compositional_image_generation.md)

</div>

<!-- RELATED:END -->
