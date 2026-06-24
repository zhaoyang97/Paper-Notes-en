---
title: >-
  [Paper Note] Analyzing the Rapid Generalization of SFT via the Perspective of Attention Head Activation Patterns
description: >-
  [ACL 2025][LLM (Other)][Supervised Fine-Tuning] This paper, through a gradient-based analysis of attention head activation patterns, reveals three key mechanisms by which SFT enables LLMs to rapidly adapt to downstream tasks: selective activation of task-specific attention heads, the activation patterns of complex tasks being linear combinations of those of basic tasks, and the ability of a small number of samples to significantly alter activation patterns. Crucially…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Supervised Fine-Tuning"
  - "Attention Head Activation Patterns"
  - "Rapid Generalization"
  - "Complex Task Decomposition"
  - "Parameter-Efficient"
date: 2026-05-08
content_hash: a8d7f9bd7d87ec98
---

# Analyzing the Rapid Generalization of SFT via the Perspective of Attention Head Activation Patterns

**Conference**: ACL 2025  
**arXiv**: [2409.15820](https://arxiv.org/abs/2409.15820)  
**Area**: LLM / Fine-Tuning Mechanism Analysis  
**Keywords**: Supervised Fine-Tuning, Attention Head Activation Patterns, Rapid Generalization, Complex Task Decomposition, Parameter-Efficient

## TL;DR
This paper, through a gradient-based analysis of attention head activation patterns, reveals three key mechanisms by which SFT enables LLMs to rapidly adapt to downstream tasks: selective activation of task-specific attention heads, the activation patterns of complex tasks being linear combinations of those of basic tasks, and the ability of a small number of samples to significantly alter activation patterns. Crucially, a practical strategy is proposed to leverage basic task data to facilitate the learning of complex tasks.

## Background & Motivation

**Background**: LLMs demonstrate exceptional performance on downstream tasks through Supervised Fine-Tuning (SFT), requiring only thousands of instruction instances to learn various tasks. However, on complex tasks like mathematical reasoning, LLM performance remains suboptimal. This is mainly because complex tasks require the synergy of diverse knowledge and skills, whereas the corresponding high-quality instruction data is scarce and difficult to collect.

**Limitations of Prior Work**: The internal mechanisms of LLMs remain opaque—while SFT is known to be effective, how it adapts the model to new tasks is not well understood. Existing parameter analysis methods find that parameter changes after SFT are limited; however, due to the massive parameter scale, interpreting the significance of these minute changes is challenging.

**Key Challenge**: LLMs exhibit remarkable rapid generalization capabilities (learning in a few-shot manner) on simple tasks, yet require a large volume of data for complex tasks. Understanding the internal mechanism and prerequisites of this rapid generalization can guide efficient adaptation to complex tasks.

**Goal**: From the perspective of attention head activation patterns, analyze the changes in internal mechanisms of LLMs during SFT to answer three progressive questions: (1) how do activation patterns change across different tasks? (2) what is the relationship between the activation patterns of complex tasks and basic tasks? (3) how much data is required to significantly alter activation patterns?

**Key Insight**: Attention heads are regarded as the fundamental functional units of Transformers, with different heads responsible for processing different types of information. SFT essentially adjusts the composition of these functional units.

**Core Idea**: SFT adapts to new tasks by selectively enhancing the activation levels of task-relevant attention heads. The activation patterns of complex tasks can be approximated by a linear combination of basic task patterns.

## Method

### Overall Architecture
The inputs are an LLM and a set of task data. A gradient-based analysis method is employed to compute the impact of each attention head on task outputs (activation levels), producing an $L \times H$ activation pattern matrix (where $L$ is the number of layers, and $H$ is the number of attention heads per layer). The changes and relationships of activation patterns before and after SFT, as well as between different tasks, are then analyzed.

### Key Designs

1. **Gradient-Based Quantization of Attention Head Activation**:

    - **Function**: Quantify the contribution of each attention head to task completion
    - **Mechanism**: For a given dataset $\mathcal{T}$, the activation level of the $h$-th attention head in the $l$-th layer is defined as $AL_{l,h} = \frac{1}{N}\sum_i \Gamma_{l,h}^T \frac{\partial L(x_i)}{\partial \Gamma_{l,h}}$, where $\Gamma_{l,h}$ is the attention matrix and the gradient reflects the sensitivity of the model output to attention scores. The activation levels of all attention heads constitute the activation pattern matrix $AP^{\mathcal{T}}$
    - **Design Motivation**: Gradients naturally measure the impact of input features on the output. Combining the absolute values of attention scores with their sensitivity to the loss provides a comprehensive quantification of the attention heads' contribution to the task.

2. **Linear Combination Analysis of Activation Patterns**:

    - **Function**: Reveal the quantitative relationship between the activation patterns of complex tasks and basic tasks
    - **Mechanism**: Linear regression is utilized to fit the change in the activation pattern of complex tasks $\Delta AP^{complex} = \sum_i \alpha_i \Delta AP^{basic_i} + \epsilon$, using the $R^2$ value to measure goodness of fit. Experiments demonstrate that the activation patterns of GSM8K (mathematics) and Code Search Net (coding) can fit the activation pattern of SGSM (solving math problems using code) with an $R^2 = 0.97$.
    - **Design Motivation**: If the activation patterns of complex tasks can be decomposed into combinations of basic tasks, it implies that preparation for complex tasks can be achieved by first learning foundational skills.

3. **Activation Pattern-Based Data Selection and Joint Training**:

    - **Function**: Leverage theoretical findings to improve SFT efficiency for complex tasks
    - **Mechanism**: For a complex task, regression analysis is first performed to determine the required basic skills and their respective weights $\alpha_i$. A pre-training dataset is then constructed by combining basic task data proportional to these weights, $Dataset^{pre} = \{N \times \alpha_i / \sum_i \alpha_i\}$, followed by training on this data prior to fine-tuning on the target task.
    - **Design Motivation**: When complex task data is scarce, abundant basic task data can be utilized to establish the necessary "prior knowledge" for the model.

### Loss & Training
During the analysis phase, gradients are computed without updating parameters. The application phase adopts a standard SFT pipeline: pre-training on basic skill data first, followed by fine-tuning on the target task data.

## Key Experimental Results

### Main Results

| Model | Base | SFT | Random Pre-training | Ours Pre-training |
|------|------|-----|-------------|-----------|
| Llama-7B | 28.68 | 31.78 | 33.82 | **36.82** |
| Llama2-7B | 29.07 | 34.50 | 36.75 | **38.50** |
| Llama3-8B | 46.12 | 49.22 | 50.10 | **52.33** |

### Ablation Study

| Analysis Dimension | Metric | Result | Description |
|---------|------|------|------|
| Activation Pattern Changes after SFT | Gini Coefficient (Llama3) | 0.50→0.33 | Activation shifts from centralized to dispersed |
| Activation Pattern Changes after SFT | CV (Llama3) | 1.19→0.71 | Decreased variability, more uniform |
| SGSM Linear Fitting $R^2$ | Code+GSM8K | 0.97 | Perfect fit with coding + mathematics |
| Infinity Instruct $R^2$ | Reasoning+Programming | 0.95 | Combination of basic skills fits complex task |
| Activation Pattern Changes under Small Data | MSE/Correlation Coefficient | Sharp change in first 200 steps | A small number of samples can reshape the activation pattern |

### Key Findings
- The activation distribution of attention heads after SFT becomes more uniform, indicating that more attention heads are "awakened" to participate in the tasks.
- Activation patterns of different tasks exhibit clear task specificity: mathematics/code tasks cluster together, while textual reasoning tasks form another group.
- The stronger the model capability (Llama3 > Llama2 > Llama), the fewer samples are required to converge on simple tasks.
- OPT-6.7B requires more data to change its activation patterns on complex tasks, suggesting that rapid generalization is difficult to achieve when pre-trained knowledge is insufficient.

## Highlights & Insights
- The finding that "complex tasks = linear combinations of basic skills" is remarkably elegant, with $R^2$ values reaching 0.95–0.97. This provides a solid theoretical foundation for "curriculum learning" and "skill decomposition".
- Significant practical value: when complex task data is scarce, analyzing activation patterns allows for automatic selection and combination of publicly available basic skill data for pre-training, achieving a 5-percentage-point improvement on MathBench.
- The methodology is transferable to the analysis of parameter-efficient fine-tuning methods such as LoRA—activation pattern analysis can help identify which attention heads should be prioritized for fine-tuning.

## Limitations & Future Work
- The linear combination assumption is overly simplified; more complex tasks may require non-linear relationship modeling.
- Only the attention head level is analyzed, while functional analysis of FFN layers is neglected—even though FFNs are believed to store a substantial amount of knowledge.
- Regression analysis requires a predefined set of basic tasks, but in practice, "which basic skills compose a complex task" remains an open question that needs to be solved.
- Lacks comparison with recent LLM-based knowledge-graph reasoning methods.
- Experiments are primarily conducted on Llama-family models; applicability to other architectures (such as Mixture-of-Experts) remains unverified.
- Future work can extend activation pattern analysis to multimodal models to analyze the functional division of labor between visual and language modules.

## Related Work & Insights
- **vs. Attention Head Pruning (Voita et al., 2019)**: Pruning studies demonstrate that attention heads possess distinct functions; this work further reveals how SFT reorganizes these functional units.
- **vs. LoRA/PEFT**: Parameter-efficient fine-tuning methods apply updates to a small subset of parameters, consistent with this paper's finding that "small parameter changes can significantly alter activation patterns."
- **vs. Skill Decomposition (Xia et al., 2024)**: Works like LIMA demonstrate that SFT can be achieved with minimal data; this work provides a mechanistic explanation for this phenomenon from the perspective of activation patterns.

## Rating
- Novelty: ⭐⭐⭐⭐ Approaching SFT mechanism analysis via attention head activation patterns offers a novel perspective. The linear combination discovery is highly inspiring, providing an actionable analytical tool for understanding LLM internal mechanisms.
- Experimental Thoroughness: ⭐⭐⭐⭐ The analysis is comprehensive, spanning three models, seven tasks, and two application scenarios.
- Writing Quality: ⭐⭐⭐⭐ The logical chain is clear, with a natural transition from analysis to application.
- Value: ⭐⭐⭐⭐ Highly instructive for understanding SFT mechanisms and guiding data-efficient fine-tuning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MHA2MLA: Towards Economical Inference by Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs](mha2mla_deepseek_latent_attention.md)
- [\[ACL 2025\] Ranking Unraveled: Recipes for LLM Rankings in Head-to-Head AI Combat](ranking_unraveled_recipes_for_llm_rankings_in_head-to-head_ai_combat.md)
- [\[ACL 2025\] Circuit Stability Characterizes Language Model Generalization](circuit_stability_characterizes_language_model_generalization.md)
- [\[ACL 2025\] Computation Mechanism Behind LLM Position Generalization](computation_mechanism_behind_llm_position_generalization.md)
- [\[ACL 2025\] ConsistencyChecker: Tree-based Evaluation of LLM Generalization Capabilities](consistencychecker_tree_evaluation.md)

</div>

<!-- RELATED:END -->
