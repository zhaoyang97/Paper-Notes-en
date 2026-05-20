---
title: >-
  [Paper Note] mDP3: A Training-free Approach for List-wise Frame Selection in Video-LLMs
description: >-
  [ICCV 2025][Reinforcement Learning][Frame Selection] This paper proposes mDP3, a training-free and model-agnostic video frame selection method that estimates frame similarity in RKHS via a conditional Gaussian kernel…
tags:
  - "ICCV 2025"
  - "Reinforcement Learning"
  - "Frame Selection"
  - "Determinantal Point Process"
  - "Markov Decision Process"
  - "Video Question Answering"
  - "Training-free"
date: 2026-05-08
content_hash: a2a10bcb2c878321
---

# mDP3: A Training-free Approach for List-wise Frame Selection in Video-LLMs

**Conference**: ICCV 2025
**arXiv**: [2501.02885](https://arxiv.org/abs/2501.02885)  
**Code**: [github.com/sunh-23/MDP3](https://github.com/sunh-23/MDP3)  
**Area**: Reinforcement Learning
**Keywords**: Frame Selection, Determinantal Point Process, Markov Decision Process, Video Question Answering, Training-free

## TL;DR

This paper proposes mDP3, a training-free and model-agnostic video frame selection method that estimates frame similarity in RKHS via a conditional Gaussian kernel, leverages Determinantal Point Processes (DPP) to capture query relevance and list-wise diversity, and models temporal structure via a Markov Decision Process (MDP). Using only 8 input frames, mDP3 significantly outperforms uniform sampling and existing frame selection methods on multiple long-video benchmarks.

## Background & Motivation

### State of the Field

Video-LLMs have achieved remarkable progress in video understanding, yet processing multi-frame inputs introduces ultra-long visual token sequences, posing serious challenges: LLM context length limits, "lost-in-the-middle" degradation caused by irrelevant frames, resource constraints on edge devices, and token costs in API calls.

### Limitations of Prior Work

**Uniform sampling**: Query-agnostic; may miss key frames while including irrelevant ones.

**Query-frame retrieval**: Considers only point-wise relevance between individual frames and the query, ignoring list-level inter-frame relationships.

**Lack of diversity**: Videos contain large numbers of similar frames; redundant selection increases computational burden with negligible information gain.

**Ignoring temporal structure**: The temporal ordering of video frames is critical for understanding causal relationships.

### Root Cause

Effective frame selection must simultaneously satisfy three principles: **query relevance** (selected frames are pertinent to the question), **list-level diversity** (selected frames are mutually complementary in information), and **temporality** (the sequential structure of frames is preserved). Furthermore, frame selection follows the law of diminishing marginal utility — the marginal benefit of adding more frames decreases and can even become negative (e.g., 64 frames underperforming 32 frames).

## Method

### Overall Architecture

mDP3 is a plug-and-play inference-time frame selection module that reuses the visual and text encoders of a pretrained VLM to extract frame and query embeddings. The pipeline proceeds as follows: (1) compute a conditional frame similarity matrix in RKHS; (2) select a frame subset that balances relevance and diversity via MAP inference under DPP; (3) segment the video and dynamically allocate the selection budget across segments via MDP.

### Key Designs

#### 1. **Conditional Multi-Gaussian Kernel (CMGK) — High-Dimensional Conditional Similarity Estimation**

- **Function**: Estimates query-conditioned inter-frame similarity in RKHS, replacing low-dimensional cosine similarity.
- **Mechanism**: Defines a conditional kernel $\tilde{k}(f_i, f_j | q) = g(f_i, q) \cdot k(f_i, f_j) \cdot g(f_j, q)$, where $k, g \in \mathcal{K}$ are convex combinations of multiple kernels. Constructs the conditional similarity matrix:

$$\tilde{\mathbf{L}} = \text{diag}(\mathbf{r}) \cdot \mathbf{L} \cdot \text{diag}(\mathbf{r})$$

where $\mathbf{L}_{ij} = k(f_i, f_j)$ is the inter-frame similarity and $\mathbf{r}_i = g(f_i, q)$ is the frame–query relevance score.
- **Design Motivation**: Directly using cosine similarity to characterize inter-frame relationships is overly coarse; kernel methods in RKHS capture richer high-dimensional relationships. The conditional kernel further injects query information into similarity estimation, downweighting irrelevant frames.

#### 2. **Determinantal Point Process (DPP) — Joint Modeling of Relevance and Diversity**

- **Function**: Applies DPP over the conditional similarity matrix to simultaneously capture query relevance and list-level diversity.
- **Mechanism**: The probability of a subset $S$ is proportional to the determinant of the corresponding submatrix $\det(\tilde{\mathbf{L}}_S)$, i.e., the geometric volume of the subset in RKHS — larger volume implies greater frame dispersion. A weight factor $\lambda$ balances relevance and diversity:

$$\log(\det(\tilde{\mathbf{L}}_S)) = \frac{1}{\lambda}\sum_{i \in S}\log(\mathbf{r}_i^2) + \log(\det(\mathbf{L}_S))$$

Solved via greedy MAP inference, guaranteeing a $(1-1/e)$ approximation ratio; Cholesky decomposition reduces complexity to $\mathcal{O}(nk^2)$.
- **Design Motivation**: DPP naturally models a "repulsion" effect (analogous to the Pauli exclusion principle for fermions in quantum physics), making it well-suited for expressing list-level diversity. Compared to the NP-hard subset selection problem, DPP with greedy inference provides an efficient approximate solution.

#### 3. **Markov Decision DPP (MDP) — Temporal Modeling and Budget Allocation**

- **Function**: Divides the video into contiguous segments, models the frame selection process across segments via MDP, and uses dynamic programming to optimally allocate the total selection budget $k$.
- **Mechanism**: The video is divided into $T = \lceil n/m \rceil$ segments; the DPP for segment $t$ is conditioned on the selections from the previous segment:

$$\mathcal{P}(S_t | S_{t-1}) = \frac{\det(\tilde{\mathbf{L}}_{S_{t-1} \cup S_t})}{\det(\tilde{\mathbf{L}}_t + \mathbf{I}_t)}$$

A state triple $(t, k_t, C_t)$ is defined, and the value function is updated via dynamic programming as $Q^*(t, C_t) = \max_{k_t} Q(t, k_t, C_t)$, with pseudo-polynomial time complexity $\mathcal{O}(nk^3)$.
- **Design Motivation**: Standard DPP treats frames as an exchangeable set, ignoring temporal structure. Query-relevant frames tend to be concentrated in specific segments, necessitating non-uniform budget allocation. MDP with dynamic programming avoids the constraint relaxation required by integer programming and finds the optimal allocation.

### Loss & Training

**Training-free method**: mDP3 requires no training whatsoever, reusing encoders from a pretrained VLM (e.g., CLIP). At inference time, 8 frames are selected from 128 candidate frames and fed to the Video-LLM, with negligible additional computational overhead.

## Key Experimental Results

### Main Results

| Model | Frames | Video-MME (wo/w subs) | MLVU | LVBval |
|------|------|----------------------|------|--------|
| LLaVA-OneVision | 8 | 53.6 / 53.9 | 59.3 | 54.2 |
| **+mDP3** | **8** | **59.6 / 59.1** (+6.0) | **69.8** (+10.5) | **59.0** (+4.8) |
| MiniCPM-V2.6 | 8 | 52.6 / 53.1 | 55.4 | 51.2 |
| **+mDP3** | **8** | **58.0 / 61.8** (+5.4) | **66.6** (+11.2) | **57.1** (+5.9) |
| Ovis2 | 8 | 58.9 / 62.1 | 60.9 | 56.9 |
| **+mDP3** | **8** | **63.9 / 66.1** (+5.0) | **73.9** (+13.0) | **62.7** (+5.8) |
| VILA-V1.5 | 8 | 47.5 / 50.0 | 46.3 | 47.1 |
| **+mDP3** | **8** | **53.3 / 56.6** (+5.8) | **58.6** (+12.3) | **50.8** (+3.7) |

Key comparison: LLaVA-OneVision + mDP3 (8 frames) surpasses the officially reported LLaVA-OneVision* (fine-tuned frame count, 58.2 on Video-MME).

### Ablation Study

| Method | Query Relevance | Diversity | Temporality | High-dim Similarity | Acc. (Video-MME) |
|------|---------|--------|--------|-----------|-----------------|
| Uniform | ✗ | ✗ | ✗ | ✗ | 47.5 |
| KNN (CLIP) | ✓ | ✗ | ✗ | ✗ | 48.5 |
| DPP | ✗ | ✓ | ✗ | ✗ | 49.6 |
| CDPP | ✓ | ✓ | ✗ | ✗ | 51.5 |
| CDPP + HDS | ✓ | ✓ | ✗ | ✓ | 52.1 |
| **mDP3** | **✓** | **✓** | **✓** | **✓** | **53.3** |

Performance gains by video duration (LLaVA-OneVision, LVBval):

| Video Duration | Baseline | +mDP3 | Gain |
|---------|---------|-------|------|
| 8–15s | 68.3 | 70.4 | +2.1 |
| 15s–1m | 66.9 | 73.3 | +6.4 |
| 3–10m | 52.4 | 57.8 | +5.4 |
| 15–60m | 46.8 | 51.8 | +5.0 |

### Key Findings

- **mDP3 is effective across all tested Video-LLMs**, validating its model-agnostic nature.
- **8-frame mDP3 outperforms the best result from 32-frame uniform sampling**: Selection quality matters more than quantity.
- **As few as 2 selected frames can achieve 90.8% of peak performance**: Extreme efficiency.
- **Longer videos benefit more**: Gains on short videos (8–15s) are modest, while gains on long videos (3–60m) are substantial.
- **Query relevance, diversity, and temporality are each indispensable**: Ablations clearly demonstrate each component's contribution.
- **Performance degrades when the number of frames exceeds the training budget (64 frames)**: Current Video-LLMs struggle to generalize beyond their training frame count.

## Highlights & Insights

1. **Theoretical elegance**: Frame selection is formalized as an NP-hard submodular maximization problem; DPP provides a $(1-1/e)$ approximation guarantee, and MDP provides pseudo-polynomial-time optimal budget allocation.
2. **Training-free and plug-and-play**: Requires no additional data or training; directly reuses pretrained VLMs and is applicable to any Video-LLM.
3. **Deep insight into diminishing marginal utility**: The principle that "quality of frames matters more than quantity" is validated both empirically and theoretically via submodularity.
4. **Physics-inspired design**: DPP originates from the fermionic exclusion principle in quantum physics; modeling frame diversity via "repulsion" is intuitively natural.

## Limitations & Future Work

1. **Dependence on pretrained VLM embedding quality**: If the VLM poorly represents frames in a specific domain, frame selection quality degrades accordingly.
2. **Selection from 128 candidate frames**: The initial uniform sampling of 128 frames may already miss critical frames.
3. **Fixed hyperparameter $m$ for segment length**: Adaptive segmentation may yield better performance.
4. **Validated only on VidQA tasks**: Effectiveness on video captioning, temporal grounding, and other tasks remains to be verified.
5. **$\mathcal{O}(nk^3)$ complexity**: Although fast in practice, the theoretical complexity still leaves room for improvement.

## Related Work & Insights

- Frame-Voyager is a frame selector trained specifically for VILA; mDP3 surpasses it without any training.
- Prior applications of DPP to video summarization (Gong et al., Zheng & Lu) provide the foundation for the segmented DPP formulation in this work.
- Long-video benchmarks such as Video-MME highlight the critical role of frame selection in long-video understanding.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The combination of DPP and MDP for frame selection is novel, with solid theoretical contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 4 Video-LLMs, 3 benchmarks, comprehensive ablations and analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Mathematical derivations are rigorous and clear, though the heavy notation requires some background knowledge.
- **Value**: ⭐⭐⭐⭐⭐ — Training-free, model-agnostic, and significantly performance-boosting; highly practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] RL-Selector: Reinforcement Learning-Guided Data Selection via Redundancy Assessment](reinforcement_learning-guided_data_selection_via_redundancy_assessment.md)
- [\[AAAI 2026\] A Multi-Agent Conversational Bandit Approach to Online Evaluation and Selection of User-Aligned LLM Responses](../../AAAI2026/reinforcement_learning/a_multi-agent_conversational_bandit_approach_to_online_evaluation_and_selection_.md)
- [\[CVPR 2026\] See It, Say It, Sorted: An Iterative Training-Free Framework for Visually-Grounded Multimodal Reasoning in LVLMs](../../CVPR2026/reinforcement_learning/see_it_say_it_sorted_an_iterative_training-free_framework_for_visually-grounded_.md)
- [\[NeurIPS 2025\] Strategic Costs of Perceived Bias in Fair Selection](../../NeurIPS2025/reinforcement_learning/strategic_costs_of_perceived_bias_in_fair_selection.md)
- [\[AAAI 2026\] Thinker: Training LLMs in Hierarchical Thinking for Deep Search via Multi-Turn Interaction](../../AAAI2026/reinforcement_learning/thinker_training_llms_in_hierarchical_thinking_for_deep_search_via_multi-turn_in.md)

</div>

<!-- RELATED:END -->
