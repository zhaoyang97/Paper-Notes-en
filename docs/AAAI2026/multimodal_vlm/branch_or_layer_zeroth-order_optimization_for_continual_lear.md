---
title: >-
  [Paper Note] Branch, or Layer? Zeroth-Order Optimization for Continual Learning of Vision-Language Models
description: >-
  [AAAI 2026][Multimodal VLM][Zeroth-order optimization] This paper systematically investigates the application of zeroth-order (ZO) optimization in PEFT-based vision-language continual learning (VLCL). It finds that naively replacing first-order (FO) optimization with ZO causes training instability, and proposes a progressive ZO-FO hybrid strategy ranging from branch-wise to layer-wise granularity. Building on the theoretical finding that visual modality exhibits larger gradient variance, the paper further proposes MoZO (gradient sign normalization + visual perturbation constraint), achieving state-of-the-art performance across four benchmarks.
tags:
  - AAAI 2026
  - Multimodal VLM
  - Zeroth-order optimization
  - continual learning
  - CLIP
  - parameter-efficient fine-tuning
  - modality-aware optimization
date: 2026-05-08
content_hash: 4b187f9bc91b76f4
---

# Branch, or Layer? Zeroth-Order Optimization for Continual Learning of Vision-Language Models

**Conference**: AAAI 2026
**arXiv**: [2506.12409](https://arxiv.org/abs/2506.12409)
**Code**: N/A
**Area**: Multimodal VLM
**Keywords**: Zeroth-order optimization, continual learning, CLIP, parameter-efficient fine-tuning, modality-aware optimization

## TL;DR
This paper systematically investigates the application of zeroth-order (ZO) optimization in PEFT-based vision-language continual learning (VLCL). It finds that naively replacing first-order (FO) optimization with ZO causes training instability, and proposes a progressive ZO-FO hybrid strategy ranging from branch-wise to layer-wise granularity. Building on the theoretical finding that visual modality exhibits larger gradient variance, the paper further proposes MoZO (gradient sign normalization + visual perturbation constraint), achieving state-of-the-art performance across four benchmarks.

## Background & Motivation
CLIP-based continual learning (VLCL) has advanced rapidly in recent years, with PEFT strategies such as LoRA and MoE adapters enabling competitive performance at low computational cost. However, existing methods almost universally rely on first-order (FO) optimization (SGD/Adam), whose deterministic update trajectories tend to converge to sharp local minima within the low-dimensional parameter subspaces defined by PEFT, leading to overfitting on the current task and exacerbating catastrophic forgetting. ZO optimization estimates gradients via random perturbations, which naturally facilitates escaping local minima and eliminates the need for backpropagation, thereby reducing memory consumption. Nevertheless, the applicability of ZO in VLCL has never been systematically studied.

## Core Problem
**How can ZO optimization be effectively integrated into VLCL to improve performance?** Specifically: (1) What problems arise from naively replacing all FO updates with ZO? (2) To which modality branch (visual vs. language) should ZO be applied? (3) Within a single branch, to which layers should ZO be applied (consecutive vs. interleaved)? (4) How should behavioral differences between modalities under ZO optimization be addressed?

## Method

### Overall Architecture
Built upon a frozen CLIP-ViT-B/16 backbone, only the adapter modules (MoE adapter or LoRA) attached to each layer are trained. The core idea is to apply FO and ZO optimization collaboratively across different modality branches and network layers: a subset of trainable units employs ZO optimization (random perturbation-based gradient estimation for exploration), while the remainder retains FO optimization (exact gradients for stability). The MoZO strategy is then applied for further refinement.

### Key Designs

1. **Branch-wise ZO Exploration**: Three branch-level configurations are examined — Dual (ZO on both branches), Vision (ZO on visual branch + FO on language branch), and Language (ZO on language branch + FO on visual branch). Experiments show that Dual w/ ZO causes severe loss oscillation and performance collapse (average drops of 8.5%/9.5% on Last./Avg.); single-branch ZO performs comparably or even better than the baseline, with Language w/ ZO generally outperforming Vision w/ ZO, as the language branch has lower-dimensional tensors and is more robust to random perturbations.

2. **Layer-wise ZO Exploration**: Following the identification of effective single-branch ZO configurations, four layer-level schemes are examined: Hop-odd (ZO on odd-indexed layers), Hop-even (ZO on even-indexed layers), Prefix(six) (ZO on first 6 layers), and Suffix(six) (ZO on last 6 layers). A key finding is that interleaved configurations (Hop-odd/even) significantly outperform consecutive ones (Prefix/Suffix), as shallow layers capture local features while deep layers encode abstract semantics; uniform optimization ignores this diversity, whereas ZO-FO interleaving better matches each layer's distinct needs for exploration and stability. Dual w/ ZO + layer-wise improves average performance by 9.4% over full-branch ZO.

3. **MoZO (Modality-aware ZO)**: Analysis of gradient variance distributions reveals that the visual branch exhibits substantially larger gradient variance than the language branch under ZO, causing optimization instability. MoZO comprises two components:

    - **Gradient Sign Normalization**: The sign of ZO-estimated gradients is taken, retaining only directional information while discarding magnitude, thereby suppressing abnormally large gradient values.
    - **Modality-Differentiated Perturbation**: A smaller perturbation factor $\epsilon_v < \epsilon_l$ is assigned to the visual branch to constrain the magnitude of parameter exploration.

### Loss & Training
- ZO gradient estimation: $\nabla_{ZO}\mathcal{L}(\theta) \approx \frac{\mathcal{L}(\theta + \varepsilon\Delta) - \mathcal{L}(\theta)}{\varepsilon} \cdot \Delta$, where $\Delta$ is a random direction vector and $\varepsilon=0.001$.
- A conservative ZO strategy is adopted: multiple candidate updates are evaluated and the one yielding the lowest loss is applied, rather than a single-estimate aggressive update.
- MoZO update rule: the visual branch is perturbed by $\epsilon_v \xi_t$ and the language branch by $\epsilon_l \xi_t$, with $\epsilon_v < \epsilon_l$.
- The FO/ZO mixing ratio $\lambda=1$ is validated on the first task and kept fixed thereafter.

## Key Experimental Results

The baseline is MoE4Adapter (CVPR 2024 SOTA) with CLIP-ViT-B/16 backbone.

| Dataset (Config) | Metric | Baseline (FO) | Dual ZO | Vision ZO | Language ZO | Best Layer-wise |
|---|---|---|---|---|---|---|
| CIFAR Inc10 | Last. | 80.47 | 69.29 (-11.2) | 76.05 | 80.94 (+0.5) | **82.41** (Vis. Hop-even) |
| CIFAR Inc10 | Avg. | 86.97 | 77.36 (-9.6) | 83.93 | 87.00 | **88.51** (Lan. Hop-odd) |
| TinyImg Inc10 | Last. | 77.52 | 67.64 (-9.9) | 72.98 | 76.74 | **79.39** (Vis. Hop-odd) |
| TinyImg Inc10 | Avg. | 85.21 | 75.88 (-9.3) | 82.08 | 85.03 | **87.05** (Lan. Hop-odd) |
| TinyImg Inc20 | Last. | 52.13 | 42.40 (-9.7) | 49.65 | 49.14 | **52.27** (Vis. Hop-even) |
| TinyImg Inc20 | Avg. | 60.55 | 47.64 (-12.9) | 57.90 | 58.69 | **60.98** (Vis. Hop-even) |
| ImgR Inc20 | Last. | 65.36 | 58.56 (-6.8) | 62.54 | 64.38 | **65.68** (Vis. Hop-odd) |
| ImgR Inc20 | Avg. | 71.53 | 65.92 (-5.6) | 69.84 | 70.38 | **72.16** (Lan. Hop-even) |

**Further gains from MoZO** (Dual w/ Hop-even → MoZO):

| Dataset | Last. | Avg. |
|---|---|---|
| CIFAR Inc10 | 79.12 → 79.87 (+0.75) | 86.81 → 87.25 (+0.44) |
| TinyImg Inc20 | 51.95 → 52.46 (+0.51) | 60.53 → 61.23 (+0.70) |
| ImgR Inc20 | 64.99 → 65.80 (+0.81) | 71.29 → 71.82 (+0.53) |

**Memory efficiency** (MoE setting):

| Configuration | MoE Memory | LoRA Memory |
|---|---|---|
| Baseline (full FO) | 19.96 GB | 15.11 GB |
| Dual w/ ZO | 2.17 GB (-89.1%) | 1.73 GB |
| Vision w/ ZO | 6.93 GB (-65.3%) | 5.71 GB |
| Language w/ ZO | 12.39 GB (-37.9%) | 11.09 GB |

### Ablation Study
- **ZO strategy selection**: Aggressive ZO* (single-estimate direct update) yields the worst performance (Dual: 66.67 Last.); conservative ZO (best among multiple candidates) is intermediate; adding sign gradient normalization achieves the best result (Language + Sign: 78.52 Last.), validating the importance of gradient magnitude control.
- **Interleaved vs. consecutive layers**: Interleaved configurations (Hop-odd/even) consistently outperform consecutive ones (Prefix/Suffix), as interleaving yields substantially lower gradient variance and more stable training.
- **Consistency under LoRA**: Replacing MoE with LoRA preserves all trends — layer-wise ZO remains effective, interleaved outperforms consecutive, and MoZO provides additional gains.
- **Significance analysis**: Across 5 runs, Language w/ ZO exhibits the smallest variance and best performance, confirming that the language branch is better suited for ZO optimization.

## Highlights & Insights
- **Systematic empirical exploration**: The progressive research path from branch-wise to layer-wise granularity is methodologically rigorous, with each step unveiling the optimal application of ZO in VLCL.
- **Theory-driven, experiment-validated design**: The finding that the visual modality exhibits larger gradient variance is not assumed a priori but is empirically verified through gradient variance distribution analysis, after which a targeted solution is proposed.
- **Substantial memory savings**: By eliminating backpropagation, Dual ZO reduces memory consumption by 89%, providing a practical solution for resource-constrained scenarios.
- **Novel perspective on PEFT optimization**: The paper identifies the under-recognized issue of FO optimization converging to sharp local minima in low-dimensional subspaces; ZO-FO co-optimization represents a novel and principled remedy.

## Limitations & Future Work
- **Restricted to CLIP**: Validation is limited to CLIP-ViT-B/16; larger models (e.g., ViT-L/14) and other VLM architectures (e.g., BLIP-2, LLaVA) remain untested.
- **Image-text modalities only**: Other modalities such as audio and video are not explored, a limitation the authors themselves acknowledge.
- **Small-scale datasets**: CIFAR-100, Tiny-ImageNet, and ImageNet-R are relatively small-scale; the approach has not been validated on larger continual learning benchmarks.
- **Modest MoZO gains**: Although MoZO consistently improves performance, the margins are small (0.4–0.8%), which appears modest relative to the substantial gains introduced by layer-wise ZO.
- **Limited comparison with other CL methods**: Comparisons are primarily conducted against MoE4Adapter with different ZO configurations; comprehensive evaluation against other VLCL methods such as PROOF and CLAP4CLIP is absent.
- **Underspecified hyperparameter selection**: The paper does not provide detailed descriptions of the specific values of $\epsilon_v$ and $\epsilon_l$ or the tuning procedure.
- **Training time overhead**: The conservative ZO strategy requires evaluating multiple candidate updates, which may increase training time despite reduced memory usage; no training time comparison is provided.

## Related Work & Insights
- **vs. MoE4Adapter (CVPR 2024)**: The direct baseline of this work, which achieves SOTA in PEFT-based VLCL via MoE architecture. This paper introduces ZO optimization on top of it, surpassing its full-FO performance via layer-wise configuration while substantially reducing memory.
- **vs. ZeroFlow (2025)**: Another work applying ZO to continual learning, but ZeroFlow does not differentiate between modalities and is primarily designed for CNN architectures. This paper is the first to systematically study ZO in the multimodal VLM setting and addresses cross-modal optimization discrepancies.
- **vs. MeZO (NeurIPS 2023)**: MeZO applies ZO to LLM fine-tuning but does not consider the continual learning setting or multimodal branch differences. This paper finds that naive full ZO replacement fails in VLCL, and that careful branch-level and layer-level configuration is necessary.
- **vs. BOFA**: A concurrent CLIP-based continual learning work at AAAI 2026 that employs orthogonal low-rank fusion. Both address the same problem but from different perspectives — BOFA focuses on parameter space orthogonality, while this paper focuses on optimizer design.
- **Broader implications**: The ZO-FO hybrid optimization paradigm can potentially be generalized to other PEFT scenarios (e.g., VLM instruction tuning), particularly under memory constraints. The finding that visual modality exhibits larger gradient variance suggests that visual and language branches may require distinct optimization strategies beyond simply tuning the learning rate, offering insights for multimodal training design. The superiority of interleaved layers over consecutive ones resonates with the role of skip connections in ResNets, suggesting that per-layer optimizer diversity benefits feature learning.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — First systematic study of ZO in VLCL with a novel perspective; however, ZO itself is not a new technique and the MoZO design (sign normalization + reduced perturbation) is relatively intuitive.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Branch-wise and layer-wise exploration is comprehensive with rich ablations; however, datasets are small-scale and comparisons with broader VLCL methods are lacking.
- **Writing Quality**: ⭐⭐⭐ — The logical structure is clear, but the writing contains redundancies and some passages restate the same observations; notation is not fully consistent.
- **Value**: ⭐⭐⭐⭐ — Provides a new optimization perspective and practical solution for PEFT-based VLCL with clear memory advantages; real-world impact is somewhat limited by the CLIP + small-dataset experimental scope.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Enhanced Continual Learning of Vision-Language Models with Model Fusion](../../ICLR2026/multimodal_vlm/enhanced_continual_learning_of_vision-language_models_with_model_fusion.md)
- [\[CVPR 2026\] Continual Learning with Vision-Language Models via Semantic-Geometry Preservation](../../CVPR2026/multimodal_vlm/continual_learning_with_vision-language_models_via_semantic-geometry_preservatio.md)
- [\[AAAI 2026\] Harnessing Textual Semantic Priors for Knowledge Transfer and Refinement in CLIP-Driven Continual Learning](harnessing_textual_semantic_priors_for_knowledge_transfer_and_refinement_in_clip.md)
- [\[AAAI 2026\] BOFA: Bridge-Layer Orthogonal Low-Rank Fusion for CLIP-Based Class-Incremental Learning](bofa_bridge-layer_orthogonal_low-rank_fusion_for_clip-based_.md)
- [\[AAAI 2026\] BiPrompt: Bilateral Prompt Optimization for Visual and Textual Debiasing in Vision-Language Models](biprompt_bilateral_prompt_optimization_for_visual_and_textual_debiasing_in_visio.md)

<!-- RELATED:END -->
