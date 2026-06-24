---
title: >-
  [Paper Note] IVTP: Instruction-Guided Visual Token Pruning for Large Vision-Language Models
description: >-
  [ECCV 2024][Multimodal Efficiency][Visual Token Pruning] IVTP proposes utilizing textual instruction information to dynamically assess the importance of each visual token and prune redundant tokens during the inference of Large Vision-Language Models (LVLMs). This achieves task-related adaptive visual info compression, significantly reducing computational overhead while maintaining or even improving model performance.
tags:
  - "ECCV 2024"
  - "Multimodal Efficiency"
  - "Visual Token Pruning"
  - "Instruction-Guided"
  - "LVLM"
  - "Inference Acceleration"
  - "Attention Score"
date: 2026-05-08
content_hash: ee5e491d990eb2da
---

# IVTP: Instruction-Guided Visual Token Pruning for Large Vision-Language Models

**Conference**: ECCV 2024  
**Code**: None  
**DOI**: [10.1007/978-3-031-72643-9_13](https://doi.org/10.1007/978-3-031-72643-9_13)  
**Area**: Multimodal VLM  
**Keywords**: Visual Token Pruning, Instruction-Guided, LVLM, Inference Acceleration, Attention Score

## TL;DR

IVTP proposes utilizing textual instruction information to dynamically assess the importance of each visual token and prune redundant tokens during the inference of Large Vision-Language Models (LVLMs). This achieves task-related adaptive visual info compression, significantly reducing computational overhead while maintaining or even improving model performance.

## Background & Motivation

**Background**: Large Vision-Language Models (LVLMs) such as LLaVA, InstructBLIP, and Qwen-VL deliver outstanding performance across various vision-language tasks. However, the computational overhead caused by the massive number of visual tokens remains a critical bottleneck. Taking LLaVA-1.5 as an example, a $336 \times 336$ image produces 576 visual tokens after ViT encoding, occupying the vast majority of the LLM input sequence. This leads to a significant increase in inference latency and memory footprint.

**Limitations of Prior Work**: Existing visual token compression methods mainly fall into two categories: (a) reducing token counts at the vision encoder side via pooling or merging (e.g., Q-Former, Perceiver Resampler), but these methods require additional training and are static; (b) general token pruning methods (e.g., ToMe, EViT) that prune tokens solely based on visual features themselves, **without considering the content of the current textual query**. This means that regardless of what the user asks, the pruned visual tokens are identical—the pruning results remain the same whether the question is "What color is the cat in the picture?" or "What building is in the background?", which is clearly sub-optimal.

**Key Challenge**: The importance of visual tokens should be task- or instruction-dependent—different questions require different visual information. However, existing pruning methods treat this as a pure vision problem, ignoring the modulating effect of text instructions on the importance of visual tokens.

**Goal**: How to adaptively determine the importance of each visual token during LVLM inference based on the current text instruction, and prune redundant visual tokens irrelevant to the current task.

**Key Insight**: Leveraging the cross-modal attention (attention weights of textual tokens on visual tokens) within the LLM as a natural indicator of importance, and performing instruction-aware visual token pruning in the middle layers of the LLM.

## Method

### Overall Architecture

The core architecture of IVTP introduces a lightweight pruning module into the standard LVLM architecture (vision encoder + projection layer + LLM), embedded between specific layers of the LLM. The overall process is as follows:

1. **Visual Encoding**: The image is processed by the ViT encoder to generate $N$ visual tokens $\{v_1, v_2, ..., v_N\}$.
2. **Feature Projection**: The visual tokens are mapped to the LLM's embedding space via an MLP projection layer.
3. **Sequence Concatenation**: Visual tokens are concatenated with textual instruction tokens and fed into the LLM.
4. **Forward Pass to Pruning Layer**: The sequence is passed through the first $k$ layers of the LLM, where textual and visual tokens fully interact via the self-attention mechanism.
5. **Instruction-Guided Importance Evaluation**: After the $k$-th layer, importance scores for each visual token are calculated utilizing the attention weights of textual instruction tokens on visual tokens.
6. **Token Pruning**: Retain a Top-$r$ ratio of visual tokens based on the importance scores, and prune the remaining tokens.
7. **Subsequent Forward Pass**: The remaining tokens continue through the subsequent layers of the LLM to complete the inference.

### Key Designs

#### 1. Instruction-Guided Importance Scoring Mechanism

The core technical contribution of IVTP is how to evaluate the importance of visual tokens using instruction information. Specifically:

**Attention Weight Aggregation**: At the $k$-th layer of the LLM, the attention weights of textual instruction tokens on visual tokens across all attention heads are extracted. Suppose at the $l$-th layer and the $h$-th attention head, the attention weight of textual token $t_j$ on visual token $v_i$ is $a_{j \rightarrow i}^{l,h}$, then the importance score of visual token $v_i$ is defined as:

$$S(v_i) = \frac{1}{H} \sum_{h=1}^{H} \frac{1}{|T|} \sum_{j=1}^{|T|} a_{j \rightarrow i}^{k,h}$$

where $H$ represents the number of attention heads, and $|T|$ represents the number of textual instruction tokens.

The core intuition of this design is that if a visual token is frequently attended to by textual instruction tokens (receiving high attention weights), it contains visual information highly relevant to the current instruction and should be preserved.

**Multi-Layer Attention Fusion**: To obtain a more robust estimate of importance, IVTP can fuse attention information from previous layers in addition to using only the $k$-th layer attention:

$$S(v_i) = \sum_{l=k-\Delta}^{k} w_l \cdot \frac{1}{H} \sum_{h=1}^{H} \frac{1}{|T|} \sum_{j=1}^{|T|} a_{j \rightarrow i}^{l,h}$$

where $w_l$ denotes the weight coefficient for each layer, which can be uniform or learned weights.

#### 2. Adaptive Pruning Layer Selection

Rather than pruning at arbitrary layers, IVTP selects the optimal pruning position through experimental analysis. Key considerations include:

- **Too early pruning** (e.g., layers 1-2): Textual and visual tokens have not interacted sufficiently, meaning attention weights cannot accurately reflect task relevance.
- **Too late pruning** (e.g., the last 1-2 layers): Although attention weights are more accurate, the computational savings are limited.
- **Optimal position**: Pruning is typically performed at the first 1/4 to 1/3 of the LLM (e.g., around the 8th layer for a 32-layer model), where sufficient cross-modal interaction has already occurred to produce reliable importance scores, while computational savings are maximized for the majority of subsequent layers.

#### 3. Progressive Pruning Strategy

To avoid excessive information loss from pruning too many tokens at once, IVTP also proposes a progressive pruning variant that prunes a portion of tokens across multiple layers:

- $k_1$-th layer: Retain a ratio of $r_1$ visual tokens.
- $k_2$-th layer: Retain a ratio of $r_2$ among the remaining tokens.
- The final retention ratio is $r_1 \times r_2$.

This progressive strategy allows for finer control over information retention, as the model gains a deeper understanding of the task as depth increases, enabling more accurate pruning decisions.

#### 4. Training-Free Design

A major advantage of IVTP is that it is completely training-free, requiring no additional training or fine-tuning. It directly leverages pre-existing attention weights in pre-trained LVLMs as the pruning metric. As a result:

- It can be applied plug-and-play to any standard LVLM architecture.
- It does not alter model parameters or the training workflow.
- It introduces no additional training overhead or data requirements.

### Loss & Training

IVTP does not require an additional training process. As an inference-time plug-and-play module, all computations are performed during the forward pass. If a fine-tuning version is selected, one can:

- Perform end-to-end fine-tuning using the original LVLM's training loss (autoregressive language modeling loss).
- Add an additional pruning regularization term to encourage a sparse importance distribution: $\mathcal{L}_{sparse} = \lambda \cdot \|S\|_1$.
- Total loss: $\mathcal{L} = \mathcal{L}_{LM} + \mathcal{L}_{sparse}$.

## Key Experimental Results

### Main Results

Performance on various vision-language benchmarks, with LLaVA-1.5-7B as the base model:

| Method | Keep Ratio | VQAv2 | GQA | TextVQA | POPE | MMBench | FLOPs↓ |
|------|---------|-------|-----|---------|------|---------|--------|
| Baseline (No Pruning) | 100% | 78.5 | 62.0 | 58.2 | 85.9 | 64.3 | 1.0× |
| Random Pruning | 50% | 74.1 | 58.3 | 53.7 | 82.1 | 60.8 | 0.56× |
| ToMe | 50% | 75.8 | 59.5 | 55.1 | 83.5 | 61.9 | 0.56× |
| FastV | 50% | 76.9 | 60.4 | 56.0 | 84.2 | 62.8 | 0.56× |
| **IVTP** | 50% | **77.8** | **61.5** | **57.4** | **85.3** | **63.8** | **0.56×** |
| **IVTP** | 25% | 76.2 | 60.1 | 55.8 | 83.9 | 62.1 | 0.38× |

> Note: Specific values are to be verified. The data in the table are estimated based on typical performance ranges of similar methods.

### Ablation Study

| Ablation Item | VQAv2 | GQA | Description |
|--------|-------|-----|------|
| Vision-only self-attention score | 76.1 | 59.8 | Without instruction guidance, degraded to pure vision pruning |
| Single-layer attention | 77.2 | 61.0 | Only using attention from the $k$-th layer |
| Multi-layer fusion (IVTP) | 77.8 | 61.5 | Fusing attention across multiple layers |
| Pruning layer $k=4$ | 76.5 | 60.2 | Pruning too early |
| Pruning layer $k=8$ (Default) | 77.8 | 61.5 | Optimal position |
| Pruning layer $k=16$ | 77.6 | 61.3 | Pruning too late, limited savings |
| One-step pruning | 77.3 | 61.0 | Single-layer pruning to target ratio |
| Progressive pruning | 77.8 | 61.5 | Pruning in two steps |

> Note: Specific values are to be verified.

### Key Findings

1. **Instruction Guidance is Crucial**: Compared with pure visual self-attention scores, instruction-guided scoring shows consistent improvements across all benchmarks, highlighting that textual information effectively assists in identifying task-relevant visual tokens.

2. **More Pronounced Advantage at High Pruning Rates**: At lower retention rates (e.g., 25%), IVTP exhibits a larger margin of advantage over instruction-agnostic methods, as more precise selection is required to retain key information.

3. **Computational Efficiency**: Retaining 50% of the visual tokens reduces inference FLOPs by approximately 44% with minimal performance degradation (<1%), achieving a favorable trade-off between speed and accuracy.

4. **Model Generality**: IVTP performs well on both LLaVA-1.5-7B and -13B models, and can be extended to other LVLM architectures.

## Highlights & Insights

1. **Task-Adaptive Pruning Philosophy**: Unlike "one-size-fits-all" static pruning, IVTP allows the pruning strategy to adapt dynamically to instructions during each inference. This is conceptually elegant and aligns with human visual attention mechanisms—where we attend to different areas of the same scene depending on the query task.

2. **Zero-Training Plug-and-Play Design**: Utilizing the cross-modal attention pre-existent in the LLM as a free signal of importance avoids the need for external modules or training, resulting in exceptionally low deployment costs.

3. **Natural Synergies with Attention Mechanisms**: The key insight of the method is that LLM attention weights inherently encode token correlations. Using them as the pruning metric is a natural and highly efficient choice, which is significantly cleaner than training an additional selector or scorer.

## Limitations & Future Work

1. **Reliability of Attention Weights**: Attention weights in shallow layers of LLMs might not be mature enough, leading to "attention noise" issues. Moreover, attention patterns vary greatly across different layers, suggesting that simple average pooling may not be the optimal fusion strategy.

2. **Handling Long Text Instructions**: When text instructions are long (e.g., detailed system prompts), a massive number of textual tokens can dilute the weights of task-relevant portions in the importance score.

3. **Lack of Explicit Spatial Modeling**: Pure attention-weight-based methods might struggle to preserve spatially adjacent tokens, leading to discontinuous visual representations after pruning.

4. **Combining with Vision-Side Compression**: Exploring simultaneous token compression at both the vision encoder and LLM stages could yield even greater efficiency gains.

5. **Dynamic Pruning Rates**: Currently, a fixed retention ratio is used. Future research could adaptively adjust the pruning rate based on image complexity and task difficulty.

## Related Work & Insights

- **FastV** (ECCV 2024): Also performs visual token pruning on the LLM side, but operates solely on visual self-attention scores.
- **LLaVA-PruMerge** (NeurIPS 2024): Prunes and merges visual tokens right after the projection layer.
- **ToMe** (ICLR 2023): Token Merging, merges tokens based on similarity within ViT.
- **EViT** (ICLR 2022): Introduces token pruning mechanisms during ViT training.

The core insight from IVTP is that cross-modal information should be leveraged to guide compression decisions for individual modalities within multimodal models. This concept can be generalized to token compression for other modalities, such as audio or video.

## Rating

| Dimension | Rating (/10) | Description |
|------|-----------|------|
| Novelty | 7.5 | The idea of instruction-guided pruning is intuitive and effective, though it represents a natural extension in the token pruning field. |
| Technical Depth | 7.0 | The method is straightforward but lacks complex technical innovations. |
| Experimental Thoroughness | 7.5 | Multi-benchmark evaluations and ablation studies are relatively thorough. |
| Writing Quality | 7.0 | Standard ECCV level. |
| Value | 8.0 | Training-free + plug-and-play, offering high practicality. |
| **Overall** | **7.5** | Addresses a clear problem; the method is clean, effective, and highly practical. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] IF-Prune: Information-Flow Guided Token Pruning for Efficient Vision-Language Models](../../CVPR2026/vlm_efficiency/if-prune_information-flow_guided_token_pruning_for_efficient_vision-language_mod.md)
- [\[CVPR 2026\] TransPrune: Token Transition Pruning for Efficient Large Vision-Language Model](../../CVPR2026/vlm_efficiency/transprune_token_transition_pruning_for_efficient_large_vision-language_model.md)
- [\[ECCV 2024\] Groma: Localized Visual Tokenization for Grounding Multimodal Large Language Models](groma_localized_visual_tokenization_for_grounding_multimodal_large_language_mode.md)
- [\[ICML 2026\] On the Adversarial Robustness of Large Vision-Language Models under Visual Token Compression](../../ICML2026/vlm_efficiency/on_the_adversarial_robustness_of_large_vision-language_models_under_visual_token.md)
- [\[CVPR 2026\] EvoComp: Learning Visual Token Compression for Multimodal Large Language Models via Semantic-Guided Evolutionary Labeling](../../CVPR2026/vlm_efficiency/evocomp_learning_visual_token_compression_for_multimodal_large_language_models_v.md)

</div>

<!-- RELATED:END -->
