---
title: >-
  [Paper Note] Mechanism of Task-oriented Information Removal in In-context Learning
description: >-
  [ICLR 2026][Image Restoration][in-context learning] This paper proposes a novel "information removal" perspective to explain the internal mechanism of In-context Learning (ICL): it finds that under zero-shot settings…
tags:
  - "ICLR 2026"
  - "Image Restoration"
  - "in-context learning"
  - "information removal"
  - "denoising heads"
  - "mechanistic interpretability"
  - "low-rank filter"
date: 2026-05-08
content_hash: 7e5ba10abe7ea1b1
---

# Mechanism of Task-oriented Information Removal in In-context Learning

**Conference**: ICLR 2026
**arXiv**: [2509.21012](https://arxiv.org/abs/2509.21012)  
**Code**: None  
**Area**: Image Restoration
**Keywords**: in-context learning, information removal, denoising heads, mechanistic interpretability, low-rank filter

## TL;DR
This paper proposes a novel "information removal" perspective to explain the internal mechanism of In-context Learning (ICL): it finds that under zero-shot settings, language models encode queries into "non-selective representations" containing information about all possible tasks (leading to near-random outputs), while the core function of few-shot ICL is to simulate a "task-oriented information removal" process—through identified "Denoising Heads" that selectively remove redundant task information from entangled representations, guiding the model to focus on the target task. Ablation experiments confirm that blocking Denoising Heads significantly degrades ICL accuracy.

## Background & Motivation

**Background**: In-context Learning (ICL) is a hallmark capability of large language models—enabling them to perform new tasks by providing a few demonstrations in the prompt, without any fine-tuning. Despite its widespread adoption, the internal mechanism of "how ICL works" remains poorly understood.

**Limitations of Prior Work**:
   - **Limited theoretical perspectives**: Existing explanations include "ICL as implicit gradient descent," "ICL as Bayesian inference," and "induction heads performing copy-paste," but these are either validated on simplified models or cover only specific task types, lacking a unified and in-depth understanding.
   - **Unclear reasons for zero-shot failure**: Under zero-shot settings without demonstrations, LM accuracy on many tasks approaches zero. The model possesses the relevant knowledge yet produces random outputs—why?
   - **Unclear role of demonstrations**: How do few-shot demonstrations alter internal representations to steer the model from "trying to do everything" to "focusing on the target task"? The mechanism remains opaque.

**Key Challenge**: Pretraining equips LMs with the ability to handle diverse tasks, but these capabilities exist in an "entangled" form within hidden states. Under zero-shot settings, the hidden states of a query contain information about all possible tasks, causing incoherent outputs. What ICL demonstrations need to do is not "add information" but "remove interference."

**Goal**: To explain the core mechanism of ICL from the novel perspective of "information removal"—how demonstrations help the model eliminate redundant task information from entangled representations and focus on the target task.

**Key Insight**:
   - First, demonstrate that zero-shot hidden states are "non-selective" (containing information about all tasks).
   - Then, use low-rank filters to artificially simulate information removal, verifying that removing redundant information indeed improves task accuracy.
   - Next, analyze few-shot ICL hidden states to show that their effect is equivalent to task-oriented information removal.
   - Finally, identify the key attention heads (Denoising Heads) responsible for executing the removal operation.

**Core Idea**: The mechanism of ICL is not "learning new knowledge from demonstrations" but "removing redundant information from entangled representations via demonstrations"—denoising rather than learning.

## Method

### Overall Architecture
This paper presents a mechanistic analysis rather than a new model. The analytical framework comprises four progressive discoveries:

**Discovery 1**: Non-selective representations under zero-shot settings
**Discovery 2**: Low-rank filters can simulate task-oriented information removal
**Discovery 3**: Few-shot ICL naturally simulates the information removal process
**Discovery 4**: Key attention heads (Denoising Heads) are the executors of information removal

### Key Designs

1. **Discovery and Measurement of Non-selective Representations**:

    - Function: Analyze the hidden states of query tokens in zero-shot scenarios, demonstrating that these representations contain information about all possible tasks.
    - Mechanism: Design precise metrics to quantify the presence of different task information in hidden states. For example, for a sentiment classification query, examine whether the hidden state simultaneously contains activation signals for "sentiment classification," "topic classification," "translation," and other tasks.
    - Experimental Finding: Under zero-shot settings, hidden states are indeed "non-selective"—information from different tasks is intermixed, preventing the model from determining which task to perform, resulting in near-random outputs (accuracy approaching zero).
    - Design Motivation: This finding explains the fundamental cause of zero-shot failure—not that "the model cannot do it," but that "the model tries to do everything."

2. **Low-rank Filter Experiments**:

    - Function: Design a low-rank projection operation $P$ to filter hidden states as $h' = P \cdot h$, selectively removing information along specific task dimensions.
    - Mechanism: Decompose the hidden state matrix via SVD to identify principal component directions associated with different tasks, then project onto the task-relevant low-rank subspace—equivalent to removing information in the orthogonal complement of that subspace.
    - Experimental Finding: Applying low-rank filtering to zero-shot hidden states enables the model to "focus" on the target task, yielding significant accuracy improvements—validating the hypothesis that "information removal = task direction."
    - Design Motivation: Low-rank filters provide a controllable information removal tool to verify whether artificially removing redundant information yields effects equivalent to ICL.

3. **Hidden State Analysis of Few-shot ICL**:

    - Function: Compare few-shot and zero-shot hidden states to demonstrate that demonstrations are functionally equivalent to task-oriented information removal.
    - Mechanism: Use carefully designed metrics to quantify the "selectivity" of few-shot hidden states—measuring whether redundant task information is compressed and whether target task information is amplified.
    - Experimental Finding: As the number of demonstrations increases, hidden states gradually become more "selective"—redundant information is suppressed while target task information dominates. This process quantitatively aligns with the effects observed in low-rank filter experiments.
    - Design Motivation: Directly comparing natural ICL and artificial filtering demonstrates that ICL is functionally equivalent to information removal.

4. **Identification and Validation of Denoising Heads**:

    - Function: Localize the key attention heads responsible for information removal within the Transformer's multi-head attention mechanism, termed "Denoising Heads."
    - Mechanism:
        - Screen for heads with the highest contribution to information removal by analyzing each attention head's effect on the hidden state "selectivity" metric.
        - The attention patterns of these heads reveal that they primarily attend to task-relevant portions of demonstrations (e.g., label tokens), using this information to modulate the query's hidden state.
    - Validation (Ablation Study):
        - "Blocking" Denoising Heads during inference (zeroing their output or substituting the original hidden state) → significant drop in ICL accuracy.
        - The accuracy degradation is most severe under the extreme "flipped labels" setting, where correct labels are absent from demonstrations—since information removal is most critical in this scenario.
    - Design Motivation: Identifying the specific components that execute information removal advances the mechanistic understanding from "black-box functional description" to "component-level causal validation."

### Analytical Methodology

Key analytical tools employed in this paper include:
- **Hidden State Probing**: Training linear probes to detect the presence of specific task information in hidden states.
- **Causal Ablation**: Verifying the causal role of specific components through targeted interventions.
- **Low-rank Projection**: SVD decomposition combined with low-rank approximation as an information removal tool.
- **Attention Head Analysis**: Quantitatively evaluating each head's contribution to information removal.
- **Carefully Designed Control Experiments**: Such as flipped labels and random labels, to distinguish ICL behavior across different conditions.

## Key Experimental Results

### Experimental Setup
- **Models**: Validated across multiple language models (GPT-2 family, LLaMA at various scales).
- **Tasks**: Text classification (sentiment analysis, topic classification, etc.)—selected for their well-defined label spaces, which facilitate measurement of "task information."
- **Scale**: 87-page paper, 90 figures, 7 tables—an extraordinarily thorough experimental report.

### Main Results

**Discovery 1: Non-selective Representations**

| Setting | Accuracy | Hidden State Selectivity | Notes |
|---------|----------|--------------------------|-------|
| Zero-shot | ~0% | Low (multi-task information entangled) | Model "tries to do everything" |
| Artificial low-rank filtering | Significant improvement | High (target task information dominant) | Removing redundant info ≈ task guidance |
| Few-shot ICL (4-shot) | High | High | Demonstrations naturally achieve information removal |

**Discovery 2: ICL ≈ Information Removal**
- The effects of low-rank filtering and few-shot ICL are highly consistent on quantitative metrics.
- Both render hidden states "more selective"—redundant task information is compressed.

**Discovery 3: Denoising Heads Ablation**

| Configuration | ICL Accuracy Change | Notes |
|---------------|---------------------|-------|
| Normal ICL | Baseline | — |
| Block Denoising Heads | Significant drop (↓15–30%) | Information removal is disrupted |
| Block non-Denoising Heads | Minor impact | Non-critical heads do not affect ICL |
| Flipped Labels + Block Denoising Heads | Worst degradation | Information removal is more critical without correct labels |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|------------|-------|
| Varying number of demonstrations | Information removal degree increases monotonically | More demonstrations = stronger denoising |
| Varying model scale | Larger models have more Denoising Heads | Scale ↑ → information removal capacity ↑ |
| Varying task types | Information removal mechanism consistently present | Validated across sentiment, topic, and other tasks |
| Varying label space size | More labels require more information removal | More possible tasks = more redundant information to remove |

### Key Findings
- **ICL is not "learning new skills" but "filtering interference"**: This is the central finding. LMs already possess capabilities for diverse tasks; demonstrations merely help the model "focus" on the correct task.
- **Denoising Heads are few but critical**: Only a small subset of attention heads is responsible for information removal, yet blocking them has a large impact on ICL performance.
- **Information removal is more critical under flipped labels**: When demonstration labels are flipped (deliberately incorrect), the model still partially succeeds—indicating that the primary role of demonstrations is not to provide correct labels but to signal "which task to perform" (by removing information about other tasks).
- **Denoising Heads differ in location across models but are functionally consistent**: This validates the generality of the mechanism.

## Highlights & Insights

- **A novel perspective on ICL**: Compared to "ICL as implicit gradient descent" or "ICL as Bayesian inference," the "ICL as information removal" view is more intuitive and actionable—it clarifies that demonstrations do not "teach new things" but rather "tell the model what to do."
- **Discovery of non-selective representations**: This work is the first to systematically demonstrate that zero-shot hidden states contain information about all tasks, explaining a longstanding puzzle: why a knowledgeable model produces near-random outputs under zero-shot conditions.
- **The concept of Denoising Heads**: Localizing the information removal operation to specific attention heads represents an important advance in mechanistic interpretability—moving from "functional description" to "component localization."
- **Low-rank filters as an analytical tool**: This provides an elegant experimental framework for artificially simulating information removal, offering a new methodology for ICL mechanism research.
- **Depth and thoroughness**: 87 pages, 90 figures, 7 tables—the authors validate each finding from multiple angles with exceptional rigor.

## Limitations & Future Work

- **Primarily validated on classification tasks**: Whether the information removal mechanism applies to generative tasks (e.g., dialogue, summarization, translation) remains unclear; "task information" in generative settings is harder to define and measure.
- **Reliance on linear probing and low-rank projection**: Information removal may involve nonlinear transformations that linear low-rank approximations only partially capture.
- **Model scale constraints**: Because the analysis requires detailed probing of hidden states, experiments are primarily conducted on medium-scale models (GPT-2 family, smaller LLaMA variants); applicability to very large models (100B+) remains unknown.
- **Formation mechanism of Denoising Heads**: The paper identifies the existence of these heads but does not explain how they emerge during pretraining—a question requiring further study of training dynamics.
- **Unification with other ICL theories**: Whether the information removal perspective is complementary to or in conflict with "implicit gradient descent" and "Bayesian inference" views lacks explicit theoretical reconciliation.

## Related Work & Insights

- **Induction Heads** (Olsson et al.): Identify attention heads that perform "copy-paste" operations. Denoising Heads represent another class of functional attention heads executing "information filtering."
- **Task Vectors**: Find that vectors encoding task directions exist within the model's internal representations. Information removal can be understood as projecting hidden states onto the correct task vector direction.
- **Bayesian Perspective on ICL** (Xie et al. 2022): ICL performs implicit Bayesian inference—selecting the most probable task. Information removal can be viewed as an attention-head-level implementation of this Bayesian inference.
- **Mechanistic Interpretability**: This paper follows the standard paradigm of "localize functional components → causal validation → ablation experiments."
- **Implications**: The information removal perspective may offer practical guidance for prompt engineering—a good prompt should help the model "filter out irrelevant task interpretations" rather than merely "provide task information."

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ProtoTS: Learning Hierarchical Prototypes for Explainable Time Series Forecasting](protots_learning_hierarchical_prototypes_for_explainable_time_series_forecasting.md)
- [\[ICCV 2025\] Exploiting Diffusion Prior for Task-driven Image Restoration](../../ICCV2025/image_restoration/exploiting_diffusion_prior_for_task-driven_image_restoration.md)
- [\[CVPR 2026\] Winner of CVPR2026 NTIRE Challenge on Image Shadow Removal: Semantic and Geometric Guidance for Shadow Removal via Cascaded Refinement](../../CVPR2026/image_restoration/shadow_removal_cascaded_refinement.md)
- [\[ICLR 2026\] Breaking Scale Anchoring: Frequency Representation Learning for Accurate High-Resolution Inference from Low-Resolution Training](breaking_scale_anchoring_frequency_representation_learning_for_accurate_high-res.md)
- [\[CVPR 2026\] Flickerformer: A Duet of Periodicity and Directionality for Burst Flicker Removal](../../CVPR2026/image_restoration/it_takes_two_a_duet_of_periodicity_and_directionality_for_burst_flicker_removal.md)

</div>

<!-- RELATED:END -->
