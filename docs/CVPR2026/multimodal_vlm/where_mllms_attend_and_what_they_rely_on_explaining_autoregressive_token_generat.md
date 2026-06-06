---
title: >-
  [Paper Note] Where MLLMs Attend and What They Rely On: Explaining Autoregressive Token Generation
description: >-
  [CVPR 2026][Multimodal VLM][MLLM interpretability] This paper proposes Eagle, a lightweight black-box attribution framework that performs spatial attribution for autoregressive token generation in MLLMs via a unified obj…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "MLLM interpretability"
  - "attribution analysis"
  - "hallucination diagnosis"
  - "language prior vs. perceptual evidence"
  - "black-box methods"
date: 2026-05-08
content_hash: ef7c3318c4f38615
---

# Where MLLMs Attend and What They Rely On: Explaining Autoregressive Token Generation

**Conference**: CVPR 2026
**arXiv**: [2509.22496](https://arxiv.org/abs/2509.22496)  
**Code**: [https://ruoyuchen10.github.io/EAGLE/](https://ruoyuchen10.github.io/EAGLE/)  
**Area**: Multimodal VLM
**Keywords**: MLLM interpretability, attribution analysis, hallucination diagnosis, language prior vs. perceptual evidence, black-box methods

## TL;DR
This paper proposes Eagle, a lightweight black-box attribution framework that performs spatial attribution for autoregressive token generation in MLLMs via a unified objective combining insight score (sufficiency) and necessity score (indispensability), and quantifies whether each generated token relies on language priors or perceptual evidence. Eagle comprehensively outperforms existing methods in faithfulness, localization, and hallucination diagnosis while substantially reducing GPU memory requirements.

## Background & Motivation
**Background**: MLLMs have achieved remarkable progress in vision-language understanding and generation, yet how generated tokens depend on the visual modality remains poorly understood, limiting interpretability and reliability.

**Limitations of Prior Work**: (a) Attention visualization methods fail to capture complex cross-modal interactions; (b) gradient-based methods (LLaVA-CAM/IGOS++) are susceptible to textual prior interference, sensitive to cumulative effects in long sequences, and incur large memory overhead; (c) activation map methods (TAM) support only single-token attribution and lack generality; (d) subset-selection-based VPS is restricted to grounding/detection tasks, and its objective cannot be directly transferred to MLLMs.

**Key Challenge**: The autoregressive generation paradigm of MLLMs makes classification-style attribution methods difficult to adapt—methods capable of handling multi-token combinatorial attribution are required; meanwhile, activation/gradient methods lack a direct causal link between input and output, rendering them insufficiently faithful.

**Goal**: (a) Provide faithful spatial attribution for arbitrary selected output tokens in MLLMs; (b) quantify whether generated tokens rely on visual evidence or language priors.

**Key Insight**: Design a black-box framework that is independent of internal model structures (attention/gradients), attributing by observing probability changes through forward inference.

**Core Idea**: A unified objective combining sufficiency and necessity + greedy subset search + modality influence analysis.

## Method

### Overall Architecture
(1) Sparse image decomposition into $V=\{\mathbf{x}_1,...,\mathbf{x}_N\}$ superpixel regions via SLICO; (2) design of a unified insight + necessity objective; (3) greedy search to produce an ordered attribution ranking; (4) modality dependence analysis based on the ranking—quantifying language prior vs. perceptual evidence reliance for each token.

### Key Designs

1. **Insight Score (Sufficiency)**:

    - Function: Identifies the minimal set of regions sufficient to drive generation.
    - Mechanism: $s_{insight}(S) = \sum_{i=1}^n p(y_{t_i}=v_i | S, \text{Prompt}, \mathbf{y}_{<t_i})$, measuring the sum of generation probabilities for target tokens when only the subset $S$ of regions is provided.
    - Design Motivation: Locates the visual regions that best "explain" the model's generation decisions.

2. **Necessity Score (Indispensability)**:

    - Function: Identifies regions whose removal significantly reduces generation probabilities.
    - Mechanism: $s_{necessity}(V \setminus S) = \sum_{i=1}^n (1 - p(y_{t_i}=v_i | V \setminus S, \text{Prompt}, \mathbf{y}_{<t_i}))$; the greater the probability drop upon removing $S$, the more indispensable $S$ is.
    - Design Motivation: Insight considers only sufficiency (these regions are enough), whereas necessity complements it with indispensability (these regions must be present).

3. **Unified Objective and Greedy Optimization**:

    - Function: Jointly optimizes sufficiency and necessity to produce an ordered attribution ranking.
    - Mechanism: $\mathcal{F}(S) = s_{insight}(S) + s_{necessity}(V \setminus S)$; greedy search finds the ordered subset $\pi$ maximizing the objective, with total inference count $\frac{1}{2}|V|^2 + \frac{1}{2}|V|$.
    - Design Motivation: The submodularity-inspired objective guarantees diminishing marginal returns; greedy search serves as an efficient approximation to NP-hard subset selection.

4. **Modality Influence Analysis (Language Prior vs. Perception Evidence)**:

    - Function: Quantifies whether each generated token relies more on language priors or visual evidence.
    - Mechanism: Using the ordered subset $\pi$ from attribution, regions are introduced incrementally and the total probability variation for each token is computed as the perceptual influence score $I_{t_i} = \sum_r (p(...|\pi_{:r},...) - \min_j p(...|\pi_{:j},...))$.
    - Design Motivation: Unlike a simple comparison of probabilities with vs. without the image, the incremental introduction captures non-monotonic probability trajectories (rise then fall), more accurately reflecting visual dependence.

## Key Experimental Results

### Main Results — Sentence-Level Faithfulness (MS COCO Image Caption)

| Method | Model | Ins.↑ | Del.↓ | GPU Memory |
|--------|-------|-------|-------|------------|
| LLaVA-CAM | LLaVA-1.5 7B | 0.5298 | 0.5317 | 37.25 GB |
| IGOS++ | LLaVA-1.5 7B | 0.5293 | 0.5168 | 48.18 GB |
| **Eagle** | **LLaVA-1.5 7B** | **0.5970** | **0.4554** | **16.07 GB** |
| LLaVA-CAM | Qwen2.5-VL 7B | 0.5605 | 0.5464 | 47.17 GB |
| **Eagle** | **Qwen2.5-VL 7B** | **0.7006** | **0.4597** | **17.68 GB** |

### Ablation Study — Localization and Hallucination

| Metric | Method | Note |
|--------|--------|------|
| Point Game (bbox) | Eagle vs TAM | **+36.42%** |
| Point Game (mask) | Eagle vs TAM | **+42.63%** |
| Hallucination correction | Eagle | Corrects hallucinations by removing minimal regions |

### Key Findings
- Eagle comprehensively outperforms existing methods across all models (LLaVA-1.5/Qwen2.5-VL/InternVL3.5) and tasks (captioning/VQA).
- Average faithfulness improvements of 20.0% (insertion) and 13.4% (deletion); gains are larger for sensitive tokens (29–42%).
- GPU memory reduced by 50–80%: 16.07 GB vs. 48.18 GB (IGOS++) on LLaVA-1.5.
- Gains on VQA are smaller than on captioning, as VQA-generated tokens rely more heavily on reasoning and language priors.
- Hallucination diagnosis: Eagle precisely localizes visual regions responsible for hallucinations and corrects them by removing fewer than 10% of regions.

## Highlights & Insights
- **Black-box + lightweight**: Independent of gradients or attention maps, compatible with any MLLM (including API-based models), with far lower GPU overhead than gradient methods—critical for practical deployment in the era of large models.
- **Unified insight + necessity**: A single objective simultaneously captures "which regions are sufficient" and "which regions are indispensable," yielding a more complete picture than considering either criterion alone.
- **Practical value of modality analysis**: Explicitly identifies which words rely on visual input (e.g., "red," object names) and which rely on language priors (e.g., articles, prepositions), offering valuable insights for hallucination understanding and model debugging.
- **Token-agnostic property**: Even when applied to tokens dominated by language priors, the visual attribution remains unaffected—gradient methods can fail entirely when the wrong token is selected.

## Limitations & Future Work
- Inference time increases substantially (Eagle ~250–800s vs. gradient methods ~15–60s) due to $O(|V|^2)$ forward inference calls.
- The superpixel segmentation granularity (number of subregions $N$) is a fixed preset; different images may require different granularities.
- Extensibility to video or multi-image inputs has not been validated.
- The baseline of "removing all regions" in the necessity score may be overly extreme.
- Despite the generality of the black-box design, it cannot provide explanations of the model's internal mechanisms.

## Related Work & Insights
- **vs. LLaVA-CAM/IGOS++** (gradient-based methods): Gradient methods depend on selecting the correct token, consume large memory, and are affected by cumulative sequence effects; Eagle's black-box design avoids these issues.
- **vs. TAM** (activation map method): TAM supports attribution for only a single token and is effective only on Qwen2-VL; Eagle supports arbitrary token combinations and is cross-model generalizable.
- **vs. VPS** (subset-selection method): VPS is limited to object detection/grounding; Eagle extends the objective to accommodate autoregressive generation.

## Rating
- Novelty: ⭐⭐⭐⭐ The unified insight/necessity framework combined with modality analysis represents a creative architectural design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 3 models × 3 task types with multiple metrics.
- Writing Quality: ⭐⭐⭐⭐ Clear framework presentation with well-aligned motivation and methodology.
- Value: ⭐⭐⭐⭐ Provides a practical and generalizable tool for MLLM interpretability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Tell Model Where to Look: Mitigating Hallucinations in MLLMs by Vision-Guided Attention](tell_model_where_to_look_mitigating_hallucinations_in_mllms_by_vision-guided_att.md)
- [\[CVPR 2026\] Mixture of States (MoS): Routing Token-Level Dynamics for Multimodal Generation](mos_mixture_of_states_multimodal_generation.md)
- [\[CVPR 2026\] Token Warping Helps MLLMs Look from Nearby Viewpoints](token_warping_helps_mllms_look_from_nearby_viewpoints.md)
- [\[ICML 2026\] Explaining Is Harder than Predicting Alone: Evaluating Concept-Based Explanations of MLLMs as ICL Visual Classifiers](../../ICML2026/multimodal_vlm/explaining_is_harder_than_predicting_alone_evaluating_concept-based_explanations.md)
- [\[CVPR 2026\] What Do Visual Tokens Really Encode? Uncovering Sparsity and Redundancy in Multimodal Large Language Models](what_do_visual_tokens_really_encode_uncovering_sparsity_and_redundancy_in_multim.md)

</div>

<!-- RELATED:END -->
