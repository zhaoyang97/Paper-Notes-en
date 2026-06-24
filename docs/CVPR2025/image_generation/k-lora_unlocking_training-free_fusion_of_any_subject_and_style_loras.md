---
title: >-
  [Paper Note] K-LoRA: Unlocking Training-Free Fusion of Any Subject and Style LoRAs
description: >-
  [CVPR 2025][Image Generation][LoRA] This paper proposes K-LoRA, which compares the importance of subject and style LoRAs by accumulating the absolute values of Top-K elements in each attention layer, adaptively selects the entire layer's LoRA weights, and integrates a timestep scaling factor to achieve training-free, high-quality subject-style fusion.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "LoRA"
  - "style transfer"
  - "subject-style fusion"
  - "Top-K selection"
  - "training-free"
  - "FLUX"
  - "SDXL"
date: 2026-05-08
content_hash: 0d3811485f126009
---

# K-LoRA: Unlocking Training-Free Fusion of Any Subject and Style LoRAs

**Conference**: CVPR 2025  
**arXiv**: [2502.18461](https://arxiv.org/abs/2502.18461)  
**Code**: [Project Page](https://k-lora.github.io/K-LoRA.io/)  
**Area**: Image Generation  
**Keywords**: LoRA, style transfer, subject-style fusion, Top-K selection, training-free, FLUX, SDXL

## TL;DR

This paper proposes K-LoRA, which compares the importance of subject and style LoRAs by accumulating the absolute values of Top-K elements in each attention layer, adaptively selects the entire layer's LoRA weights, and integrates a timestep scaling factor to achieve training-free, high-quality subject-style fusion.

## Background & Motivation

**Background**: LoRA has become the mainstream fine-tuning method for diffusion model personalization (subject learning) and stylization. Consequently, a vast number of independently trained subject and style LoRAs exist in the community.

**Limitations of Prior Work**:
- Direct arithmetic merging ($\alpha \cdot \Delta W_c + \beta \cdot \Delta W_s$) leads to smoothed style textures and lost subject details.
- ZipLoRA requires additional training of a fusion ratio vector, and its performance is highly sensitive to seeds and hyperparameters.
- B-LoRA only fine-tunes two attention modules to achieve style decoupling, which is prone to color overfitting.
- Joint training requires access to both subject and style datasets simultaneously, making it inapplicable to community-shared LoRAs.

**Key Challenge**: Element-wise merging causes mutual interference, whereas complete separation fails to retain both subject and style. Users desire a "plug-and-play" solution—fusing any arbitrary subject and style LoRAs without retraining.

**Goal**: To design a training-free LoRA fusion method capable of preserving the complete information learned by both the original subject and style LoRAs.

## Method

### Overall Architecture

K-LoRA performs the following steps inside each attention layer during every diffusion forward pass:
1. Take the absolute values of the two LoRA weight matrices.
2. Select the Top-K largest elements from each and compute their respective sums.
3. Compare the two sums and select the entire weight matrix of the LoRA with the larger sum.
4. Regulate the priority of the subject and style at different stages using a timestep scaling factor.

### Key Designs

**1. Top-K Layer-wise Selection Mechanism**
- **Function**: For each attention layer, $S_c = \sum_{i \in \text{Top-K}(|\Delta W_c|)} |\Delta W_{c,i}|$ and $S_s$ are calculated. After comparison, the entire layer's weight is selected. Here, $K = r_c \cdot r_s$ (the product of the ranks of the two LoRAs).
- **Mechanism**: Key finding—only a few dominant elements in the LoRA matrix govern the generation quality, and original performance can be reconstructed using over 50% of the layers. Element-wise merging disrupts the relationships between these dominant elements, whereas layer-wise selection preserves the intact weight structure.
- **Design Motivation**: By performing a binary selection at the layer level without modifying the original LoRA weights, the learned concepts are preserved to the maximum extent. Taking the absolute values for Top-K prevents positive-negative cancellation, and defining $K = r_c \cdot r_s$ adapts to the information density of different LoRAs based on their ranks.

**2. Timestep Scaling Factor**
- **Function**: The style LoRA score is multiplied by a scaling factor $S' = \gamma \cdot (\alpha \cdot \frac{t_{now}}{t_{all}} + \beta)$, where $\alpha=1.5, \beta=0.5$.
- **Mechanism**: Key finding—(i) early diffusion steps are responsible for reconstructing the subject layout and macro-textures, while (ii) later steps construct style details. The scaling factor increases over timesteps, biasing the selection toward the subject LoRA in early stages and the style LoRA in later stages.
- **Design Motivation**: To achieve a smooth transition from subject construction to style rendering, aligning with the "coarse-to-fine" generation pattern of diffusion models.

**3. Weight Balancing Factor $\gamma$**
- **Function**: $\gamma = \frac{\sum_l \sum_i |\Delta W_{c_{l,i}}|}{\sum_l \sum_j |\Delta W_{s_{l,j}}|}$, which computes the ratio of the sum of absolute values between content and style across all layers.
- **Mechanism**: LoRA weights from different sources (e.g., local training vs. community downloads) often have highly discrepant magnitude scales. Direct comparison of Top-K sums would otherwise fail.
- **Design Motivation**: To normalize the overall magnitudes of the two LoRAs, enabling Top-K comparison on a fair scale.

### Loss & Training

**Training-free**. During inference:
- Top-K selection is executed in every LoRA attention layer at each forward step.
- The hyperparameters $\alpha=1.5, \beta=0.5$ are robust and applicable to almost all scenarios.
- Compatible with both SDXL and FLUX models.

## Key Experimental Results

### Main Results

| Method | Style Sim↑ | CLIP Score↑ | DINO Score↑ | Requires Training |
|---|---|---|---|---|
| Direct Merge | 48.9% | 66.6% | 43.0% | No |
| Joint Training | 68.2% | 57.5% | 17.4% | Yes |
| B-LoRA | 58.0% | 63.8% | 30.6% | Yes |
| ZipLoRA | 60.4% | 64.4% | 35.7% | Yes |
| **K-LoRA** | **58.7%** | **69.4%** | **46.9%** | **No** |

K-LoRA significantly outperforms in subject fidelity (CLIP +5%, DINO +11.2%), while achieving style similarity close to ZipLoRA.

### User Study

| Method | User Preference | GPT-4o Preference |
|---|---|---|
| ZipLoRA | 29.2% | 5.6% |
| B-LoRA | 18.1% | 11.1% |
| **K-LoRA** | **52.7%** | **83.3%** |

### Ablation Study

| Setting | Subject Fidelity | Style Fidelity | Stability |
|---|---|---|---|
| Full K-LoRA | ✓ | ✓ | High |
| Top-K Only (No Scaling Factor) | ✓ | Partially Lost | Medium |
| Fixed Selection (No Top-K) | Partially Blurry | ✓ | Low |
| Random Selection | Unstable | Unstable | Very Low |
| Top-K Without $\gamma$ | ✓ | Lost (Cross-source LoRA) | Low |

### Key Findings

1. **Only 50% of attention layers are needed to reconstruct original LoRA performance**: Demonstrating that LoRA sparsity can be effectively leveraged.
2. **Clear division of labor between subject and style across timesteps**: Early steps determine the subject layout, whereas later steps govern style textures.
3. **Layer-wise selection significantly outperforms element-wise fusion**: Preserving the intact weight structure maintains concept integrity better than element-wise mixing.
4. **$\gamma$ is indispensable for cross-source LoRAs**: The magnitude of community LoRA weights can differ by several folds, causing the method to fail entirely without normalization.
5. **$K = r_c \cdot r_s$ is the sweet spot**: If K is too small, it fails to differentiate importance; if K is too large, it introduces noisy elements.

## Highlights & Insights

- Simple and practical: entirely training-free and directly applicable to the vast pool of community-shared LoRAs.
- Supported by two core findings: LoRA sparsity and timestep-based division of labor.
- The binary layer-wise selection decision avoids the concept dilution issues inherent in element-wise mixing.
- User studies and GPT-4o evaluation consistently demonstrate its superiority.

## Limitations & Future Work

- Binary layer-wise selection cannot achieve fine-grained style-content mixing within the same layer.
- Limited capability in fusing extreme style discrepancies (e.g., abstract vs. realistic).
- Although hyperparameters $\alpha$ and $\beta$ are general, they might not be optimal for all scenarios.
- Only validated fusion of two LoRAs (subject + style), without extending to multi-LoRA fusion.
- Lacks sensitivity analysis regarding different combinations of LoRA ranks.

## Related Work & Insights

- **ZipLoRA**: Trains a fusion ratio vector; serves as the primary baseline for K-LoRA.
- **B-LoRA**: Discovers specialized roles of different attention layers and fine-tunes only two core modules.
- **Multi-LoRA Composition**: The concept of iteratively updating LoRAs inspired the layer-wise selection design in K-LoRA.
- **Textual Inversion / DreamBooth**: Foundational methods for personalization in diffusion models.

## Rating

⭐⭐⭐⭐ — The method is simple yet highly effective, and its training-free design makes it exceptionally appealing to community users. The two core findings (sparsity and timestep division of labor) are highly insightful. The 11.2% improvement in DINO Score represents a substantial advancement. However, its theoretical depth is somewhat limited, and the coarse granularity of the binary layer-wise selection may restrict performance in highly complex scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] ZipLoRA: Any Subject in Any Style by Effectively Merging LoRAs](../../ECCV2024/image_generation/ziplora_any_subject_in_any_style_by_effectively_merging_loras.md)
- [\[ECCV 2024\] Implicit Style-Content Separation using B-LoRA](../../ECCV2024/image_generation/implicit_style-content_separation_using_b-lora.md)
- [\[CVPR 2025\] Symbolic Representation for Any-to-Any Generative Tasks](symbolic_representation_for_any-to-any_generative_tasks.md)
- [\[CVPR 2025\] HSI: A Holistic Style Injector for Arbitrary Style Transfer](hsi_a_holistic_style_injector_for_arbitrary_style_transfer.md)
- [\[CVPR 2025\] SaMam: Style-aware State Space Model for Arbitrary Image Style Transfer](samam_style-aware_state_space_model_for_arbitrary_image_style_transfer.md)

</div>

<!-- RELATED:END -->
