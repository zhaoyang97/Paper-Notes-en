---
title: >-
  [Paper Note] Global Compression Commander: Plug-and-Play Inference Acceleration for High-Resolution Large Vision-Language Models
description: >-
  [AAAI 2026][Multimodal VLM][Token Compression] This paper proposes GlobalCom², a **plug-and-play, training-free** token compression framework tailored for high-resolution VLMs with dynamic cropping architectures. It leve…
tags:
  - "AAAI 2026"
  - "Multimodal VLM"
  - "Token Compression"
  - "High-Resolution VLM"
  - "Dynamic Cropping"
  - "Plug-and-Play Acceleration"
  - "Global-Local Guidance"
date: 2026-05-08
content_hash: df7ca19a4dd05e11
---

# Global Compression Commander: Plug-and-Play Inference Acceleration for High-Resolution Large Vision-Language Models

**Conference**: AAAI 2026
**arXiv**: [2501.05179](https://arxiv.org/abs/2501.05179)
**Code**: [https://github.com/xuyang-liu16/GlobalCom2](https://github.com/xuyang-liu16/GlobalCom2)
**Area**: Multimodal VLM / Model Compression
**Keywords**: Token Compression, High-Resolution VLM, Dynamic Cropping, Plug-and-Play Acceleration, Global-Local Guidance

## TL;DR

This paper proposes GlobalCom², a **plug-and-play, training-free** token compression framework tailored for high-resolution VLMs with dynamic cropping architectures. It leverages the global thumbnail as a "commander" to guide differentiated compression across local crop regions, achieving >90% of original performance while compressing 90% of visual tokens.

## Background & Motivation

High-resolution LVLMs (e.g., LLaVA-NeXT, InternVL3) widely adopt **dynamic cropping** strategies: decomposing a high-resolution image into one global thumbnail plus multiple local crops, each encoded separately by a ViT and then concatenated. While this improves fine-grained understanding, it causes a dramatic explosion in visual token count (LLaVA-NeXT: 5×576; LLaVA-OV: up to 10×729), making the quadratic complexity of LLM inference a critical bottleneck.

Existing token compression methods (FastV, SparseVLM, PruMerge, etc.) are primarily designed for **single-view** VLMs. When directly applied to HR-LVLMs with dynamic cropping, three key problems arise:
1. **Ignoring global context**: The global thumbnail's information is not utilized to assess per-crop importance.
2. **Insensitivity to information density differences**: Semantic density varies greatly across crops (e.g., a crop with players vs. one with only grass), yet existing methods compress them uniformly.
3. **Positional bias**: Attention-based methods such as FastV systematically assign more tokens to crops at later positions regardless of content importance—under extreme compression, this can cause severe multimodal hallucinations (POPE drops by 14.8 points).

## Core Problem

**How to achieve content-aware, differentiated token compression within the hierarchical visual structure of dynamic cropping?** The core challenge lies in the distinct roles of thumbnails and crops (the former provides global context, the latter provides local detail), the significant variation in information density across crops, and the tendency of existing methods to either treat all crops uniformly or compress the wrong regions due to positional bias.

## Method

### Overall Architecture

GlobalCom² follows a **global-to-local** hierarchical compression philosophy, inspired by the human visual process of grasping the overall scene before examining details. The entire framework operates after ViT encoding and before the LLM, serving as a **plug-and-play module at the vision encoding stage**, consisting of two pathways:

- **Blue pathway** (thumbnail compression): Retains thumbnail tokens via TopK selection based on [CLS] attention scores.
- **Yellow pathway** (crop compression): Two-stage, globally guided crop compression:
    - (a) **Adaptive compression adjustment**: Dynamically allocates different retention ratios to each crop based on its information richness from the global perspective.
    - (b) **Holistic token evaluation**: Evaluates the importance of each token by combining both global and local perspectives, then applies TopK retention.

### Key Designs

1. **Thumbnail Compression**: The attention scores between the [CLS] token and all patch tokens in the last ViT layer serve as importance scores $s_i^G$. The Top-$k$ ($k = R \times N$) tokens are retained. This step is relatively standard; its key contribution is that the thumbnail's attention distribution is reused downstream to guide crop compression.

2. **Adaptive Compression Adjustment**: This is the core innovation of GlobalCom². Each crop is mapped to its corresponding region on the thumbnail, and the [CLS] attention scores within that region are summed to obtain a crop-level **information richness score** $s_j^G = \sum_{i \in \text{crop}_j} s_i^G$. Softmax normalization (temperature $\tau = 10$) yields relative importance weights $\sigma_j$, and the retention ratio for each crop is adjusted as:
$$r_j = R \times \left(1 + \sigma_j - \frac{1}{n}\right)$$
This ensures that information-rich crops retain more tokens while redundant crops are compressed more aggressively, with the total token count across all crops still satisfying the target ratio $R$.

3. **Holistic Token Evaluation**: For each token within a crop, importance is assessed by combining two perspectives:
    - **Local score** $s_{j,i}^L$: Intra-crop [CLS]-to-patch attention score, capturing local saliency.
    - **Global score** $\hat{s}_{j,i}^G$: The thumbnail's 1D attention scores are reshaped to 2D and bilinearly interpolated to the original resolution; the values corresponding to each crop region serve as global importance.
    - Combined score: $s_{j,i} = \alpha \cdot \hat{s}_{j,i}^G + (1 - \alpha) \cdot s_{j,i}^L$ ($\alpha = 0.5$).

4. **Adaptation for Models Without [CLS] (e.g., SigLIP)**: For LLaVA-OneVision, which uses SigLIP (no [CLS] token), a **negative cosine similarity** is proposed as a substitute: the global mean vector $\mathbf{g}$ of all tokens is computed, and tokens with lower similarity to the global mean are considered more informationally unique ($s_i = -\cos(\mathbf{x}_i, \mathbf{g})$). Experiments confirm its effectiveness is close to the [CLS]-based approach.

5. **Extension to Video Understanding**: The "global thumbnail → local crops" logic is analogized to "global video representation → individual frames." Global average pooling yields a global representation, enabling adaptive per-frame compression allocation.

### Loss & Training

**No training is involved.** The entire method is training-free and plug-and-play, with only two hyperparameters: temperature $\tau = 10$ (controlling the sharpness of inter-crop allocation) and the global-local mixing coefficient $\alpha = 0.5$. Experiments demonstrate insensitivity to both parameters.

## Key Experimental Results

**LLaVA-NeXT-7B Main Results (Table 1)**:

| Retention Ratio | Method | GQA | VQAT | POPE | MME | MM-Vet | Avg. (%) |
|---|---|---|---|---|---|---|---|
| 100% | Original | 64.2 | 64.9 | 86.5 | 1519.0 | 43.9 | 100.0% |
| 50% | FastV | 61.8 | 59.6 | 85.5 | 1490.3 | 37.6 | 95.5% |
| 50% | **GlobalCom²** | **63.9** | **62.3** | **88.1** | **1552.9** | **40.4** | **98.5%** |
| 25% | SparseVLM | 59.9 | 58.3 | 85.0 | 1465.9 | 38.5 | 94.6% |
| 25% | **GlobalCom²** | **61.5** | **60.9** | **87.6** | **1493.5** | **40.7** | **96.7%** |
| 10% | FastV | 55.9 | 55.7 | 71.7 | 1282.9 | 27.2 | 85.4% |
| 10% | FasterVLM | 56.9 | 56.5 | 83.6 | 1359.2 | 35.0 | 89.9% |
| 10% | **GlobalCom²** | **57.1** | **58.4** | **83.8** | **1365.5** | **36.4** | **91.6%** |

**LLaVA-OneVision** (Figure 6): At R=10%, 90.5% of original performance is retained while consuming only 35.4% of the original GPU memory.

**Efficiency Analysis (Table 4, R=10%)**:

| Method | TFLOPs | Memory | Throughput | Performance |
|---|---|---|---|---|
| Original | 41.7 | 23.0 GB | 3.8 samples/s | 100% |
| SparseVLM | 5.4 (↓87%) | 24.2 (↑5.2%) | 5.9 (1.6×) | 85.7% |
| FasterVLM | 3.8 (↓91%) | 13.6 (↓40%) | 6.7 (1.8×) | 89.5% |
| **GlobalCom²** | **3.8 (↓91%)** | **13.9 (↓40%)** | **6.7 (1.8×)** | **90.8%** |

Note: SparseVLM requires an explicit attention matrix, making it incompatible with FlashAttention, which causes memory to increase rather than decrease.

**Combination with Question-Aware Methods (R=10%)**:
- +FastV → average score improves by 5.3%, POPE improves by 8.2
- +SparseVLM → average score improves by 5.2%, POPE improves by 4.5

### Ablation Study

- **Adaptive compression adjustment strategies** (Table 2): Softmax(sum) > Softmax(max) > n_top-k > Uniform. The optimal strategy outperforms uniform compression by 1.4% average score, confirming that allocating based on overall crop information richness is superior to attending only to the strongest token.
- **Token evaluation sources** (Table 3): Global+Local (96.7%) > Local only (95.6%) > Global only (94.7%). Local scores perform better on fine-grained tasks (VQAT, POPE), while global scores perform better on general perception (MME, SQA), demonstrating their complementarity.
- **[CLS]-free alternatives** (Table 5): $s_i^{sim}$ (negative cosine similarity) is close to $s_i^{[CLS]}$ (95.8% vs. 96.4%), and substantially outperforms negative patch attention $s_i^{attn}$ (93.2%).
- **Hyperparameter robustness** (Figure 10): Performance remains stable for $\tau \in [5, 20]$ and $\alpha \in [0.3, 0.7]$.

## Highlights & Insights

- **"Global-to-local" compression philosophy**: A highly intuitive design—consulting the global view before deciding local compression intensity, simulating coarse-to-fine human visual processing. Simple, effective, and clear in insight.
- **Diagnosis of the positional bias bug**: The paper systematically reveals the positional bias problem of methods like FastV on HR-LVLMs: regardless of crop input order (forward/reverse), tokens at later positions consistently receive higher attention scores, independent of actual content importance. This finding is valuable in its own right.
- **Truly plug-and-play**: Operating after ViT encoding and before the LLM, it requires no modification to model architecture, no training, and is compatible with FlashAttention—highly practical.
- **Strong extensibility**: Adaptation schemes are demonstrated for both [CLS]-free models and video understanding, indicating good generality.
- **Advantage amplified under extreme compression**: At R=10%, GlobalCom² shows the largest margin over competing methods (1.7% average score above the second-best), maintaining robustness in scenarios where others degrade significantly.

## Limitations & Future Work

- **Compression limited to the vision encoding stage**: While this ensures FlashAttention compatibility, it precludes leveraging text query information for question-aware compression. The authors partially address this via combination experiments (+FastV/SparseVLM), but the two-stage pipeline has room for further optimization.
- **Hyperparameter dependency**: Although $\tau$ and $\alpha$ are relatively insensitive, their optimal values may differ across models and tasks, and the equal-weight mixing at $\alpha = 0.5$ is coarse—a task-adaptive learned $\alpha$ would be more principled.
- **Evaluation limited to the LLaVA family**: Systematic evaluation on more mainstream HR-LVLMs such as InternVL and Qwen2-VL is absent (Qwen2-VL is only tested on video tasks), leaving generalizability to be verified.
- **Coarse granularity of adaptive adjustment**: Adaptation operates at the crop level rather than at a finer sub-region level. When information density varies significantly within a single crop (e.g., left half contains text, right half is background), further differentiation is not possible.
- **No comparison or integration with token merging methods**: Interactions with merging strategies such as ToMe are unexplored. Discarding unimportant tokens outright may lose information, whereas merging provides a softer alternative.
- **Weak theoretical grounding for the [CLS]-free substitute**: Although $s_i = -\cos(\mathbf{x}_i, \mathbf{g})$ is empirically effective, the theoretical justification is thin—it is unclear why deviation from the global mean equates to information richness, though a connection to information bottleneck theory may exist.

## Related Work & Insights

| Method | Compression Stage | Dynamic Cropping-Aware | Training | FlashAttn Compatible | R=10% Performance |
|---|---|---|---|---|---|
| FastV (ECCV'24) | LLM pre-filling | ✗ | Free | ✗ | 85.4% |
| SparseVLM (ICML'25) | LLM pre-filling | ✗ | Free | ✗ | 86.1% |
| FasterVLM (2024.12) | Vision encoding | ✗ | Free | ✓ | 89.9% |
| PruMerge (ICCV'25) | Vision encoding | ✗ | Free | ✓ | 80.6% |
| **GlobalCom² (AAAI'26)** | **Vision encoding** | **✓** | **Free** | **✓** | **91.6%** |

The comparison with FasterVLM is most telling: both operate at the vision encoding stage using [CLS] attention, but FasterVLM compresses all crops uniformly, while GlobalCom² introduces globally guided differentiated compression. At R=10%, GlobalCom² surpasses FasterVLM by 1.7% average score. On LLaVA-NeXT-13B, as retention drops from R=25% to R=10%, FasterVLM's MMB score degrades by 5.9 points while GlobalCom²'s degrades by only 2.9, demonstrating significantly better robustness under compression.

The fundamental distinction from FastV/SparseVLM lies in the compression stage and the positional bias issue: the latter two evaluate token importance inside the LLM using attention scores, introducing positional bias and FlashAttention incompatibility.

**New Insight**: The positional bias finding suggests that any method performing question-aware compression inside the LLM may be susceptible to this effect. Crop-order permutation experiments should be adopted as a standard sanity check in future evaluations.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The hierarchical compression paradigm of using global guidance for local decisions is clear and innovative, though employing [CLS] attention as an importance signal is not novel (already present in FasterVLM); the core contribution lies in the systematic design tailored specifically for HR-LVLMs.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Multiple model scales (7B/13B/0.5B), multiple retention ratios (75%/50%/25%/10%), thorough ablations (strategy, score source, hyperparameters, combinability, video, efficiency), and informative visualizations.
- **Writing Quality**: ⭐⭐⭐⭐ — The analysis section (Section 3) is rigorous; the observe-then-design narrative is logically clear, and the positional bias analysis is particularly compelling.
- **Value**: ⭐⭐⭐⭐ — High practical value: training-free, plug-and-play, and FlashAttention-compatible for direct industrial deployment; academically, it opens a new "structure-aware" direction for token compression in HR-LVLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] LLMC+: Benchmarking Vision-Language Model Compression with a Plug-and-play Toolkit](llmc_benchmarking_vision-language_model_compression_with_a_plug-and-play_toolkit.md)
- [\[AAAI 2026\] Plug-and-Play Clarifier: A Zero-Shot Multimodal Framework for Egocentric Intent Disambiguation](plug-and-play_clarifier_a_zero-shot_multimodal_framework_for_egocentric_intent_d.md)
- [\[CVPR 2026\] Prune2Drive: A Plug-and-Play Framework for Accelerating Vision-Language Models in Autonomous Driving](../../CVPR2026/multimodal_vlm/prune2drive_a_plug-and-play_framework_for_accelerating_vision-language_models_in.md)
- [\[AAAI 2026\] PlantTraitNet: An Uncertainty-Aware Multimodal Framework for Global-Scale Plant Trait Inference from Citizen Science Data](planttraitnet_an_uncertainty-aware_multimodal_framework_for_global-scale_plant_t.md)
- [\[ICCV 2025\] FALCON: Resolving Visual Redundancy and Fragmentation in High-resolution Multimodal Large Language Models via Visual Registers](../../ICCV2025/multimodal_vlm/falcon_resolving_visual_redundancy_and_fragmentation_in_high.md)

</div>

<!-- RELATED:END -->
