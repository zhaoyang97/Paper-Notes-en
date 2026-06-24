---
title: >-
  [Paper Note] Universal Retrieval for Multimodal Trajectory Modeling
description: >-
  [ICML 2025][Multimodal VLM][trajectory retrieval] This work systematically defines the multimodal trajectory retrieval task for the first time. It constructs the Unified Agent Trajectory Dataset (UATD) containing 7,747 demonstrations and 82,793 states, alongside the GAE-Bench benchmark containing 714,628 positive sample pairs. Additionally, the VLM2Vec-based GAE-Retriever framework is proposed, achieving an average improvement of 10.22 percentage points over the strongest bas…
tags:
  - "ICML 2025"
  - "Multimodal VLM"
  - "trajectory retrieval"
  - "GUI agents"
  - "multimodal embedding"
  - "contrastive learning"
  - "VLM"
date: 2026-05-08
content_hash: 0fffee2d025a287c
---

# Universal Retrieval for Multimodal Trajectory Modeling

**Conference**: ICML 2025  
**arXiv**: [2506.22056](https://arxiv.org/abs/2506.22056)  
**Code**: None  
**Area**: Multimodal / GUI Agent  
**Keywords**: trajectory retrieval, GUI agents, multimodal embedding, contrastive learning, VLM

## TL;DR

This work systematically defines the multimodal trajectory retrieval task for the first time. It constructs the Unified Agent Trajectory Dataset (UATD) containing 7,747 demonstrations and 82,793 states, alongside the GAE-Bench benchmark containing 714,628 positive sample pairs. Additionally, the VLM2Vec-based GAE-Retriever framework is proposed, achieving an average improvement of 10.22 percentage points over the strongest baseline, VLM2Vec-V2.2, across 5 GUI environments.

## Background & Motivation

**Value and Challenges of Trajectory Data**: Trajectory data recorded in human-computer interaction (such as instructional videos, operation guides, and GUI navigation logs) contains rich state-action sequence knowledge, which is highly valuable for downstream tasks such as in-context reasoning, reinforcement learning, and world modeling. With the deployment and research advancement of AI agent products, the volume of trajectory data is growing exponentially.

**Limitations of Prior Work**: Current methods utilizing trajectory data (such as retrieving reusable subroutines from memory) rely solely on text features for similarity search, **ignoring multimodal signals** (such as screenshots and UI layouts). More importantly, existing works lack a systematic task definition, unified data format, and standardized benchmark for trajectory retrieval.

**Core Problem**: How to effectively model and retrieve multimodal trajectory data? Answers are needed at three levels: (1) unified representation of heterogeneous trajectory data; (2) definition of retrieval tasks covering both temporal and semantic relations; and (3) retrieval models capable of efficiently processing long multimodal sequences. This paper selects GUI environments as the initial domain for exploration, given the high application value and abundant data resources in Web automation.

## Method

### Overall Architecture

The contributions of this paper span three levels:
1. **Data Level**: Construction of the Unified Agent Trajectory Dataset (UATD), compiling heterogeneous trajectories from 5 open-source GUI sources into a standardized format.
2. **Task Level**: Definition of GAE-Bench, a multimodal trajectory retrieval benchmark containing 6 categories and 12 sub-tasks, covering both temporal and semantic retrieval relationships.
3. **Model Level**: Proposal of GAE-Retriever, a VLM-based contrastive learning retrieval framework that addresses memory bottlenecks for long multimodal sequences through token selection and GradCache.

### Key Designs

1. **UATD Unified Trajectory Representation**:
    - **Function**: Unifies 5 heterogeneous GUI data sources (Mind2Web, AutoWebGLM, WebArena, WebLINX, GUIAct) into a standardized trajectory format.
    - **Mechanism**: Trajectories are modeled as a deterministic MDP $\mathcal{E}=(\mathcal{S}, \mathcal{A}, \mathcal{O}, \mathcal{T})$, unifiedly represented as $\tau = (s_1, a_1, s_2, a_2, \ldots, s_n, a_n)$. States are represented by raw screenshots paired with text descriptions, and actions as action/target/value triplets (in JSON format), with each trajectory accompanied by a custom action space definition. For sources lacking screenshots (e.g., AutoWebGLM), `gpt-4o-mini` is used to complete the HTML, which is then rendered using Playwright.
    - **Design Motivation**: To eliminate dependence on platform-specific text representations; the unified format facilitates cross-environment generalization.

2. **GAE-Bench 12 Extraction Modes**:
    - **Function**: Systematically extracts 6 categories of retrieval pairs from a single trajectory to form a comprehensive multimodal trajectory retrieval benchmark.
    - **Mechanism**: Defines temporal retrieval (retrieving the second half given the first half and vice versa, and cross-granularity trajectory-to-state retrieval) and semantic retrieval (q→gold trajectory, q→silver trajectory, q→state). Silver trajectories are generated via a three-step process: entities are identified using Named Entity Recognition (NER) $\rightarrow$ alternative expressions are generated $\rightarrow$ queries are rewritten. GAE-Bench contains 714,628 positive pairs in total; GAE-Bench-lite limits trajectory length to $\le 10$ steps, containing 563,900 pairs.
    - **Design Motivation**: Temporal retrieval captures sequential relations within trajectories, while semantic retrieval captures functional similarities across trajectories. The 12 modes comprehensively cover different granularities (states/trajectories/sub-trajectories) and directions.

3. **GAE-Retriever Efficient Multimodal Retrieval**:
    - **Function**: Builds a trajectory retrieval model based on VLM2Vec + Qwen2-VL, addressing the memory and computational bottlenecks of sequences containing multiple high-resolution screenshots.
    - **Mechanism**: **Token Selection**—UI connectivity graphs are constructed in RGB space, and redundant visual tokens are skipped after similarity-based clustering, with a mask ratio of 0.5 used during training. **GradCache**—Gradient caching decouples the backpropagation of the encoder from that of the contrastive loss, supporting large-scale contrastive learning with a sub-batch size of 1 and an accumulated batch size of 2,048. The InfoNCE loss is used: $\mathcal{L} = -\log \frac{\exp(f(\mathbf{k})^T f(\mathbf{v}^+) / t)}{\sum_{\mathbf{v} \in \mathcal{B}} \exp(f(\mathbf{k})^T f(\mathbf{v}) / t)}$
    - **Design Motivation**: Trajectory data contains multiple high-resolution screenshots, which causes token explosion when encoded directly. Contrastive learning relies on large batch sizes for in-batch negatives; GradCache overcomes GPU memory limitations.

### Loss & Training

Based on Qwen2-VL-2B-Instruct, training is conducted using LoRA (rank=8) on 16 H800 GPUs for 256 steps, totaling 1,044 GPU hours. The learning rate is set to $5 \times 10^{-5}$ with a 5% warm-up ratio and a maximum token length of 65,536. Token selection is only enabled during training (introducing no additional learnable parameters) and disabled during evaluation. Evaluation is performed on 8 H800 GPUs with a batch size of 6, taking 22.5 GPU hours.

## Key Experimental Results

### Main Results (Recall@1/5/10, 5 Data Sources)

| Method | Mind2Web R@1/5/10 | AutoWebGLM R@1/5/10 | WebArena R@1/5/10 | WebLINX R@1/5/10 | GUIAct R@1/5/10 |
|------|-------------------|---------------------|-------------------|------------------|-----------------|
| Qwen2-VL-2B | 0.7/14.5/18.2 | 1.2/6.3/10.7 | 1.4/8.8/12.2 | 3.1/14.2/18.0 | 3.1/8.1/9.4 |
| ColQwen2-v1.0 | 3.2/22.0/29.9 | 3.9/17.7/26.3 | 2.9/13.7/20.0 | 4.2/19.6/25.1 | 6.2/15.5/19.2 |
| GME-Qwen2VL-2B | 3.7/24.2/33.4 | 8.7/27.9/37.4 | 4.2/17.7/24.7 | 5.2/22.4/29.7 | 6.0/16.7/20.7 |
| VLM2Vec-V2.2 | 10.2/44.0/60.1 | 15.7/51.2/67.1 | 9.1/29.1/37.8 | 10.7/38.4/50.5 | 12.2/33.1/40.6 |
| ShowUI-2B | 1.0/13.3/17.0 | 0.8/6.0/8.2 | 1.6/8.5/11.7 | 3.3/13.7/17.3 | 3.1/7.9/9.2 |
| **GAE-Retriever** | **15.0/50.7/67.6** | **22.1/63.6/76.3** | **10.3/31.7/44.1** | **13.7/41.7/54.1** | **25.7/59.2/67.9** |

### Ablation Study (Comparison with the Strongest Baseline VLM2Vec-V2.2)

| Data Source | R@1 Gain | R@5 Gain | R@10 Gain |
|--------|---------|---------|----------|
| Mind2Web | +4.8 | +6.7 | +7.5 |
| AutoWebGLM | +6.4 | +12.4 | +9.2 |
| WebArena | +1.2 | +2.6 | +6.3 |
| WebLINX | +3.0 | +3.3 | +3.6 |
| GUIAct | +13.5 | +26.1 | +27.3 |
| **Average** | **+5.8** | **+10.2** | **+10.8** |

### Key Findings

- GAE-Retriever achieves the best performance across all five data sources on R@1/5/10, with the most significant improvement observed on GUIAct (R@1 +13.5, R@10 +27.3).
- Multimodal backbone models (Qwen2-VL/Qwen2.5-VL) show extremely weak retrieval capabilities (R@1 < 4.0), demonstrating that retrieval requires specialized training.
- Qwen2.5-VL-3B performs worse than the smaller Qwen2-VL-2B, indicating that model scale does not directly translate to retrieval capabilities.
- Trajectory planning models (ShowUI, UI-TARS, TongUI) show no significant difference in retrieval capability compared to base backbones; planning capabilities do not transfer to retrieval.
- The VLM2Vec series consistently outperforms other retrieval models, indicating that modal data fusion and cross-batch training are crucial for retrieval.
- Under OOD settings, GAE-Retriever even outperforms IND (In-Distribution) on certain tasks, demonstrating strong generalization capability.
- Semantic retrieval tasks (q→τ, q→s) are relatively straightforward, whereas temporal retrieval (trajectory→trajectory) presents the greatest difficulty.

## Highlights & Insights

- **Groundbreaking Task Definition**: Systematically defines the "multimodal trajectory retrieval" task for the first time, establishing a complete data-benchmark-methodology framework and laying the foundation for this emerging field.
- **Comprehensive Coverage with 12 Extraction Modes**: Integrates temporal and semantic relations, 6 retrieval directions, and 3 granularities (state/trajectory/sub-trajectory). The systematic nature of the task definition far exceeds concurrent works.
- **Empirical Evidence of VLM >> CLIP**: VLMs are inherently superior to CLIP-based models in handling multimodal inputs of arbitrary lengths; conversely, screenshot-specific retrieval models (e.g., UniSE-MLLM) perform the worst.
- **Token Selection + GradCache**: These training tricks effectively balance the processing of multiple high-resolution screenshots under GPU memory constraints.
- **High Practical Value**: The framework directly supports downstream agent applications such as in-context learning, world models, and trajectory replay.

## Limitations & Future Work

- Validation is limited to GUI environments; trajectory retrieval in embodied/robotic scenarios remains to be explored.
- Relies on the visual understanding capabilities of pretrained VLMs, which might require adaptation for visual observations beyond GUIs.
- The quality of automated silver trajectory generation affects the accuracy of the semantic retrieval benchmark.
- The trajectory length constraint in GAE-Bench-lite ($\le 10$ steps) may not represent retrieval difficulties in longer trajectory scenarios.
- Integrating retrieval results into downstream agent decision-making systems has not been deeply explored.
- Training resource requirements are high (16 $\times$ H800 GPUs, 1,044 GPU hours), which sets a high barrier to replication.

## Related Work & Insights

- **VLM2Vec / VLM2Vec-V2** (Jiang et al., 2025; Meng et al., 2025): Represents the core backbone of utilizing VLMs for general retrieval.
- **Mind2Web** (Deng et al., 2023): A Web navigation benchmark; serves as one of the data sources for UATD.
- **AGUVIS** (Xu et al., 2024): Unifies GUI visual agent representations, influencing the design of state representation.
- **ShowUI** (Lin et al., 2024): Unifies action space definitions; inspired the token selection mechanism.
- **GradCache** (Gao et al., 2021): A gradient caching approach that decouples the backpropagation of the encoder and contrastive loss.
- **Insights**: Treating "trajectory" as a first-class citizen for representation learning and retrieval constitutes a critical infrastructure for agent intelligence.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Pioneering task definition + comprehensive data-benchmark-methodology framework, with thorough coverage by 12 extraction modes.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive comparison of 5 data sources across 13 baseline methods, featuring detailed per-task analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Clearly structured with detailed descriptions of the dataset and benchmark, and rigorous formal grammar definitions.
- **Value**: ⭐⭐⭐⭐⭐ Lays the infrastructure for agent trajectory research, carrying long-term value for datasets and benchmarks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Generalized Contrastive Learning for Universal Multimodal Retrieval](../../NeurIPS2025/multimodal_vlm/generalized_contrastive_learning_for_universal_multimodal_re.md)
- [\[ACL 2025\] MegaPairs: Massive Data Synthesis For Universal Multimodal Retrieval](../../ACL2025/multimodal_vlm/megapairs_massive_data_synthesis_for_universal_multimodal_retrieval.md)
- [\[ICLR 2026\] U-MARVEL: Unveiling Key Factors for Universal Multimodal Retrieval via Embedding Learning](../../ICLR2026/multimodal_vlm/u-marvel_unveiling_key_factors_for_universal_multimodal_retrieval_via_embedding_.md)
- [\[ACL 2025\] CART: A Generative Cross-Modal Retrieval Framework with Coarse-To-Fine Semantic Modeling](../../ACL2025/multimodal_vlm/cart_a_generative_cross-modal_retrieval_framework_with_coarse-to-fine_semantic_m.md)
- [\[CVPR 2025\] GENIUS: A Generative Framework for Universal Multimodal Search](../../CVPR2025/multimodal_vlm/genius_a_generative_framework_for_universal_multimodal_search.md)

</div>

<!-- RELATED:END -->
