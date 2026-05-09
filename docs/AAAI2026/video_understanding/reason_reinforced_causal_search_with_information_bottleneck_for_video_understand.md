---
title: >-
  [Paper Note] ReaSon: Reinforced Causal Search with Information Bottleneck for Video Understanding
description: >-
  [AAAI 2026][Video Understanding][Key Frame Selection] This paper proposes a Causal Information Bottleneck (CIB) theoretical framework that formalizes keyframe selection as an information-theoretic problem jointly optimizing *predictive sufficiency* and *causal necessity*. Built upon CIB, the ReaSon reinforcement learning framework trains a selection policy using three CIB-aligned rewards (answer reward, cycle-consistency reward, and counterfactual reward), significantly outperforming existing methods under constrained frame budgets.
tags:
  - AAAI 2026
  - Video Understanding
  - Key Frame Selection
  - Causal Information Bottleneck
  - Reinforcement Learning
  - Counterfactual Reasoning
  - Video Question Answering
date: 2026-05-08
content_hash: cf6a9b7cf21bf694
---

# ReaSon: Reinforced Causal Search with Information Bottleneck for Video Understanding

**Conference**: AAAI 2026
**arXiv**: [2511.12530](https://arxiv.org/abs/2511.12530)
**Code**: [github.com/robin-hlt/AAAI26-ReaSon](https://github.com/robin-hlt/AAAI26-ReaSon)
**Area**: Video Understanding
**Keywords**: Key Frame Selection, Causal Information Bottleneck, Reinforcement Learning, Counterfactual Reasoning, Video Question Answering

## TL;DR

This paper proposes a Causal Information Bottleneck (CIB) theoretical framework that formalizes keyframe selection as an information-theoretic problem jointly optimizing *predictive sufficiency* and *causal necessity*. Built upon CIB, the ReaSon reinforcement learning framework trains a selection policy using three CIB-aligned rewards (answer reward, cycle-consistency reward, and counterfactual reward), significantly outperforming existing methods under constrained frame budgets.

## Background & Motivation

**State of the Field**: Vision-language models (VLMs) have achieved remarkable progress in video understanding, yet remain constrained by input token budgets and the inherent temporal redundancy of video. Recent research has therefore focused on **keyframe selection** strategies that extract compact frame subsets to improve computational efficiency and reasoning accuracy.

**Limitations of Prior Work**:

**Ambiguous definition of keyframes**: Existing methods (e.g., VideoAgent, T*, AKEYS) define keyframes as "informative and compact frame subsets," typically using **visual or semantic relevance** as a proxy for informativeness. However, high visual relevance does not imply evidential necessity — visually relevant frames may be non-essential for reasoning, while causally decisive frames (e.g., cause and effect frames) are often overlooked.

**Lack of causal analysis**: Existing methods implicitly follow the Information Bottleneck (IB) principle, but IB only addresses predictive sufficiency and fails to capture **causal necessity** — i.e., whether removing a frame would substantially alter the output.

**Weak theoretical foundations**: Neither heuristic static selection nor agent-based interactive search is guided by a unified theoretical framework.

**Root Cause**: Visual relevance $\neq$ causal necessity. A frame may be highly visually relevant to a question yet contribute nothing to reasoning (redundant frame), while another visually unremarkable frame may be indispensable for correct inference (causal frame).

**Starting Point**: The paper revisits the concept of keyframes from a causal inference perspective. A keyframe should simultaneously satisfy two conditions — (1) **Predictive sufficiency**: the selected subset is sufficient for accurately inferring the answer; (2) **Causal necessity**: no frame in the subset can be removed without affecting the output. The classical IB is extended to a **Causal Information Bottleneck (CIB)**, and a selection policy is trained via reinforcement learning.

## Method

### Overall Architecture

ReaSon comprises two core modules: a **Predictive Sufficiency Module** and a **Causal Necessity Module**. The predictive sufficiency module first constructs a candidate frame pool via visual element matching, then employs a learnable policy network to select a compact subset. The causal necessity module evaluates the causal indispensability of each frame through counterfactual intervention. Reward signals from both modules jointly train the selection policy via reinforcement learning.

### Key Designs

#### 1. **Causal Information Bottleneck (CIB)**

The classical IB objective is $\max \mathcal{I}(F;O) \text{ s.t. } \mathcal{I}(V,Q;F) \leq \beta$, which only addresses predictive sufficiency. This paper introduces an **interventionable selection variable** $S$ in place of $F$ as the bottleneck variable, extending the objective to:

$$\max \mathcal{I}(S;O) + \mathcal{I}_c(O;\text{do}(S)) \quad \text{s.t. } \mathcal{I}(V,Q;S) \leq \beta$$

Here the first term $\mathcal{I}(S;O)$ ensures predictive sufficiency, the second term $\mathcal{I}_c(O;\text{do}(S))$ quantifies the causal influence of selected frames via causal intervention, and the constraint $\mathcal{I}(V,Q;S) \leq \beta$ limits information capacity to prevent redundant selection.

**The core contribution** lies in treating selection as an interventionable decision, thereby enabling causal necessity analysis — the classical IB bottleneck variable $F$ does not admit such intervention.

#### 2. **Predictive Sufficiency Module**

**Candidate frame pool construction**: Inspired by T*, the method first uses uniformly sampled frames to extract question-relevant visual elements $E_q$ via a frozen VLM, then applies the open-vocabulary detector YOLO-World to match these elements across all frames, constructing a candidate pool of size $M=32$.

**Policy network selection**: Frames and questions are encoded with BLIP; a three-layer LSTM + MLP constitutes the policy network $\pi_\theta(S|f,q)$, assigning selection probabilities to each candidate frame and sampling a keyframe subset of $|s| \leq K=8$ via multinomial sampling.

**Answer reward**: Selected frames and the question are fed to a frozen VLM to generate an answer, which is compared against the ground truth:
$$R_{\text{ans}} = \mathbb{I}[\text{VLM}(s,q) = \text{gt}]$$

**Cycle-consistency reward**: The generated answer is concatenated with the question and fed to the VLM to reversely infer visual elements $E_a$ (without accessing video frames); these are compared against the original $E_q$ via IoU:
$$R_{\text{cycle}} = \text{IoU}(E_q, E_a)$$

This constitutes a semantic cycle verification: visual input → answer reasoning → reverse visual attribution. High alignment indicates that the selected frames faithfully preserve the semantic cues required for reasoning.

#### 3. **Causal Necessity Module**

**Counterfactual selection strategy**: The original policy probabilities are inverted and renormalized to construct counterfactual selection probabilities:
$$\tilde{\pi}(f_i) = \frac{1 - \pi_\theta(f_i)}{\sum_j (1 - \pi_\theta(f_j))}$$

Frames preferred by the original policy are thus suppressed in the counterfactual policy, and vice versa.

**Counterfactual reward**: The original subset $s$ and counterfactual subset $s'$ are each fed to the VLM to obtain logit outputs $o$ and $o'$; KL divergence is then computed:
$$R_{\text{cf}} = D_{\text{KL}}(\text{softmax}(o) \| \text{softmax}(o'))$$

A larger divergence indicates stronger causal influence of the current frame subset — replacing these frames with counterfactual ones leads to substantially different outputs.

### Loss & Training

**Composite reward**: $R = R_{\text{ans}} + \lambda_1 R_{\text{cycle}} + \lambda_2 R_{\text{cf}}$, with $\lambda_1 = \lambda_2 = 0.5$.

**Group policy gradient**: For each training instance, $G=4$ groups of keyframe subsets plus one counterfactual subset are sampled. Group-wise advantage is computed as $\hat{A}_i = R_i - \frac{1}{G}\sum_j R_j$ (mean-centered rather than standard-deviation-normalized to reduce bias). Policy update:
$$\nabla_\theta \mathcal{L} = \frac{1}{G} \sum_{i=1}^G \hat{A}_i \cdot \nabla_\theta \log \pi_\theta(s_i | f, q)$$

Training is conducted on the NExT-QA training set; 8-frame experiments use a single RTX 3090 and 32-frame experiments use an A100.

## Key Experimental Results

### Main Results

**NExT-QA Validation Set & EgoSchema Subset (8 frames)**:

| Method | VLM | Frames | Temporal | Causal | Descriptive | Avg | EgoSchema |
|--------|-----|--------|----------|--------|-------------|-----|-----------|
| VideoAgent | GPT-4 | 8.4 | 64.5 | 72.7 | 81.1 | 71.3 | 60.2 |
| AKEYS | GPT-4o | 26.7 | 72.9 | 79.0 | 86.1 | 78.1 | 68.6 |
| T* | LLaVA-OV-7B | 8 | - | - | - | 80.4 | 66.6 |
| **ReaSon** | LLaVA-Video-7B | 8 | **77.3** | **82.1** | **87.4** | **81.4** | 69.0 |
| **ReaSon** | GPT-4o | 8 | 70.6 | 80.2 | 83.6 | 77.6 | **72.2** |

**Video-MME (without subtitles)**:

| Method | VLM | Frames | Short | Medium | Long | Overall |
|--------|-----|--------|-------|--------|------|---------|
| T* | GPT-4o | 8 | 56.4 | 57.3 | 56.4 | 56.5 |
| **ReaSon** | GPT-4o | 8 | **65.9** | **57.1** | **54.4** | **59.1** |
| T* | GPT-4o | 32 | 69.5 | 63.5 | 59.3 | 64.1 |
| **ReaSon** | GPT-4o | 32 | **76.8** | **64.2** | 58.2 | **66.4** |

### Ablation Study

| Configuration | NExT-QA Avg | EgoSchema |
|---------------|-------------|-----------|
| $R_{\text{ans}}$ only | 80.1 | 66.0 |
| $R_{\text{ans}} + R_{\text{cycle}}$ | 80.5 | 68.2 |
| $R_{\text{ans}} + R_{\text{cycle}} + R_{\text{cf}}$ (full) | **81.4** | **69.0** |

**VLM Generalizability**:

| VLM | NExT-QA (Base → +ReaSon) | EgoSchema (Base → +ReaSon) |
|-----|--------------------------|---------------------------|
| LLaVA-Video-7B | 80.2 → **81.4** | 65.2 → **69.0** |
| Qwen2.5-VL-7B | 79.9 → **80.4** | 65.8 → **68.0** |
| GPT-4o | 72.0 → **77.6** | 70.0 → **72.2** |

### Key Findings

1. **Causal necessity is the primary source of improvement**: Adding the counterfactual reward $R_{\text{cf}}$ yields a 0.8% gain on EgoSchema and a 3.1% gain on NExT-QA Descriptive, demonstrating that causal frame selection substantially improves reasoning accuracy.
2. **Short videos benefit most**: Under the 8-frame setting on Video-MME short videos, ReaSon outperforms T* by 9.5%, as causal cues are more concentrated in shorter videos.
3. **Fewer frames outperform more frames**: ReaSon with 8 frames (7B model) surpasses VideoINSTA (GPT-4) using 90 frames, validating the "quality over quantity" principle.
4. **Model-agnostic plug-and-play**: Consistent improvements are observed across three distinct VLMs, with the largest gains for GPT-4o (NExT-QA +5.6%) and Qwen2.5-VL (EgoSchema +3.8%).

## Highlights & Insights

1. **Strong theoretical contribution**: Keyframe selection is elevated from empirical heuristics to an information-theoretic foundation; CIB provides a unified theoretical framework for analyzing and designing selection strategies.
2. **Elegant counterfactual intervention design**: Constructing counterfactual selection by inverting probabilities is both simple and effective as an approximation of causal necessity.
3. **Cycle-consistency reward**: Using the answer to reversely infer visual elements as a semantic verification constitutes a clever self-supervised consistency check.
4. **High computational efficiency**: Training and inference require only a single RTX 3090 (8-frame setting); the policy network is extremely lightweight (3-layer LSTM + MLP), adding no overhead to VLM inference.
5. **Strong generalizability**: Trained on NExT-QA, the method generalizes directly to EgoSchema and Video-MME with strong performance.

## Limitations & Future Work

1. **Bottleneck on long videos persists**: Gains on Video-MME long videos are limited (the Long category under 32 frames falls below T*), as the fixed candidate pool size ($M=64$) may fail to cover critical information in very long videos.
2. **Candidate pool depends on visual element detection**: Using YOLO-World for visual element detection may miss abstract or non-object cues.
3. **Single Monte Carlo sample approximation**: The counterfactual reward uses a single sample to approximate the expectation, potentially introducing variance.
4. **Training confined to NExT-QA**: The effect of training on larger-scale or more diverse data remains unexplored.

## Related Work & Insights

- Applying the IB principle to frame selection is a natural yet insufficiently explored direction; the CIB extension proposed here provides a theoretical tool for subsequent work (e.g., key segment selection in long videos, multimodal information filtering).
- Counterfactual reasoning is widely used in causal machine learning; this paper demonstrates its practical value in video understanding.
- Group policy gradient is more lightweight than PPO and well-suited to discrete combinatorial optimization problems such as frame selection.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (CIB theoretical framework + counterfactual frame selection; dual innovation at both the theoretical and methodological levels)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (three datasets, three VLMs, comprehensive ablation, visualization analysis)
- Writing Quality: ⭐⭐⭐⭐⭐ (rigorous theoretical derivation, complete notation system, clear logical flow)
- Value: ⭐⭐⭐⭐⭐ (establishes theoretical foundations for keyframe selection; plug-and-play design offers strong practical utility)

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] APVR: Hour-Level Long Video Understanding with Adaptive Pivot Visual Information Retrieval](apvr_hour-level_long_video_understanding_with_adaptive_pivot.md)
- [\[AAAI 2026\] Causality Matters: How Temporal Information Emerges in Video Language Models](causality_matters_how_temporal_information_emerges_in_video_language_models.md)
- [\[AAAI 2026\] TSPO: Temporal Sampling Policy Optimization for Long-form Video Language Understanding](tspo_temporal_sampling_policy_optimization_for_long-form_video_language_understa.md)
- [\[NeurIPS 2025\] Token Bottleneck: One Token to Remember Dynamics](../../NeurIPS2025/video_understanding/token_bottleneck_one_token_to_remember_dynamics.md)
- [\[ICLR 2026\] Map the Flow: Revealing Hidden Pathways of Information in VideoLLMs](../../ICLR2026/video_understanding/map_the_flow_revealing_hidden_pathways_of_information_in_videollms.md)

<!-- RELATED:END -->
