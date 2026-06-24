---
title: >-
  [Paper Note] ASAG: Toward the Frontiers of Reliable Diffusion Sampling via Adversarial Sinkhorn Attention Guidance
description: >-
  [AAAI 2026][Attention Guidance] This paper proposes ASAG (Adversarial Sinkhorn Attention Guidance), which reinterprets self-attention scores in diffusion models from the perspective of optimal transport theory. By injecting adversarial transport costs into attention layers via the Sinkhorn algorithm to deliberately reduce query-key similarity, ASAG systematically disrupts misleading attention alignment and improves both conditional and unconditional sampling quality. The meth…
tags:
  - "AAAI 2026"
  - "Attention Guidance"
  - "Optimal Transport"
  - "Sinkhorn Algorithm"
  - "Diffusion Sampling"
  - "Plug-and-Play"
date: 2026-05-08
content_hash: 492236bc0d57f7f4
---

# ASAG: Toward the Frontiers of Reliable Diffusion Sampling via Adversarial Sinkhorn Attention Guidance

**Conference**: AAAI 2026
**arXiv**: [2511.07499](https://arxiv.org/abs/2511.07499)  
**Code**: None  
**Area**: Diffusion Models / Image Generation
**Keywords**: Attention Guidance, Optimal Transport, Sinkhorn Algorithm, Diffusion Sampling, Plug-and-Play

## TL;DR
This paper proposes ASAG (Adversarial Sinkhorn Attention Guidance), which reinterprets self-attention scores in diffusion models from the perspective of optimal transport theory. By injecting adversarial transport costs into attention layers via the Sinkhorn algorithm to deliberately reduce query-key similarity, ASAG systematically disrupts misleading attention alignment and improves both conditional and unconditional sampling quality. The method is lightweight, plug-and-play, and requires no retraining.

## Background & Motivation

**Background**: Diffusion models improve generation quality through guidance methods such as Classifier-Free Guidance (CFG). The core idea of CFG is to "enhance the conditional output by deliberately degrading the unconditional output"—constructing a "worse" reference point so that the conditional path becomes more prominent by contrast. Subsequent methods (e.g., PAG, SAG) follow this paradigm but rely on heuristic perturbation functions (e.g., identity mixing, blurred conditioning) to construct degraded outputs.

**Limitations of Prior Work**: Existing guidance methods lack theoretical grounding for their perturbation functions. Why does identity mixing work? Why is blurring a good degradation strategy? These design choices are manually crafted, lacking interpretability and optimality guarantees. Different tasks may also require different perturbation strategies, making manual design inefficient.

**Key Challenge**: There is a need for a principled, theory-driven approach to constructing optimal attention degradation strategies, rather than relying on heuristic design.

**Goal**: To provide a theoretical foundation for attention guidance from the perspective of Optimal Transport (OT), and to design a principled degradation strategy accordingly.

**Key Insight**: The authors observe that the self-attention mechanism in diffusion models can be understood as an OT problem—attention scores between queries and keys correspond to the coupling matrix in a transport plan, and softmax normalization corresponds to marginal constraints.

**Core Idea**: The Sinkhorn algorithm (the standard algorithm for solving entropy-regularized OT) is used to inject adversarial transport costs into attention layers—deliberately increasing the transport cost between queries and keys to systematically disrupt attention alignment and construct theoretically grounded degraded outputs.

## Method

### Overall Architecture
ASAG is embedded as a plug-and-play module into the sampling process of diffusion models. At each denoising step, an adversarial cost is injected into the query-key similarity matrix of self-attention layers, and Sinkhorn iterations adjust the attention distribution to produce a degraded guidance signal. This signal is combined with the standard conditional output to yield an enhanced generation direction. The entire process does not modify model weights.

### Key Designs

1. **Optimal Transport Interpretation of Attention**:

    - Function: Provides a mathematical framework for the attention mechanism.
    - Mechanism: Self-attention $A = \text{softmax}(QK^\top / \sqrt{d})$ is interpreted as a coupling matrix in optimal transport. $Q$ and $K$ correspond to the support points of two distributions, and the attention weight $A_{ij}$ represents the amount of "information" transported from position $j$ to position $i$. Row-wise softmax normalization corresponds to the row marginal constraint in OT. From this perspective, standard attention minimizes transport cost (i.e., maximizes query-key similarity).
    - Design Motivation: This interpretation reformulates "perturbing attention" as an optimization problem of "increasing transport cost," providing a theoretical framework.

2. **Adversarial Sinkhorn Cost Injection**:

    - Function: Principally degrades attention alignment quality.
    - Mechanism: An adversarial cost matrix $C$ is injected into the attention similarity matrix $S = QK^\top / \sqrt{d}$, yielding the modified attention $\tilde{A} = \text{Sinkhorn}(S - \lambda C)$, where $\lambda$ controls degradation strength. The cost matrix $C$ is designed to maximize pixel-level query-key dissimilarity—for each pair $(i,j)$, $C_{ij}$ is proportional to the cosine similarity between $q_i$ and $k_j$ (higher-similarity pairs are penalized more). The Sinkhorn algorithm ensures the modified attention still satisfies doubly stochastic constraints (row and column sums equal to 1), preserving the mathematical properties of the attention matrix.
    - Design Motivation: Directly adding noise to attention breaks its probabilistic properties (e.g., non-negativity, normalization). The Sinkhorn algorithm maintains the validity of the attention matrix while increasing transport cost. "Penalizing high-similarity pairs" precisely targets the most informative attention connections.

3. **Adaptive Guidance Scale**:

    - Function: Dynamically adjusts degradation strength according to the denoising stage.
    - Mechanism: A larger $\lambda$ value is applied during early denoising steps (when global structure is formed) for stronger degradation, gradually decreasing in later steps (when fine details are generated). This schedule prevents excessive perturbation in later stages from blurring details.
    - Design Motivation: Different denoising stages have different requirements for guidance strength—early stages require strong guidance to determine global layout, while later stages require weaker guidance to preserve details.

### Loss & Training
ASAG requires no training whatsoever. It operates as a plug-and-play module at inference time: given any pretrained diffusion model (e.g., Stable Diffusion, SDXL), standard self-attention is replaced with adversarial Sinkhorn attention during sampling. The main hyperparameters are the degradation strength $\lambda$ and the number of Sinkhorn iterations.

## Key Experimental Results

### Main Results: Text-to-Image Generation
Comparison with multiple guidance methods on COCO-30K and PartiPrompts (Stable Diffusion v1.5 / SDXL):

| Method | FID ↓ | IS ↑ | CLIP Score ↑ | Human Pref. ↑ |
|---|---|---|---|---|
| CFG (baseline) | 12.8 | 32.4 | 0.312 | - |
| PAG | 11.9 | 33.8 | 0.318 | 42.3% |
| SAG | 12.1 | 33.2 | 0.316 | 38.7% |
| **ASAG (Ours)** | **11.2** | **34.6** | **0.321** | **51.8%** |

### Downstream Application Enhancement

| Application | Baseline | +ASAG | Gain |
|---|---|---|---|
| IP-Adapter (CLIP-I) | 0.784 | 0.812 | +3.6% |
| ControlNet Canny (FID) | 18.3 | 16.7 | -1.6 |
| ControlNet Depth (FID) | 19.1 | 17.4 | -1.7 |
| Unconditional (FID) | 15.6 | 14.1 | -1.5 |

### Ablation Study

| Configuration | FID ↓ | Note |
|---|---|---|
| ASAG (full) | 11.2 | Full method |
| w/o Sinkhorn (direct noise) | 12.4 | Without Sinkhorn; degenerates to random perturbation |
| w/o adversarial cost | 12.0 | Uniform cost instead of adversarial |
| w/o adaptive schedule | 11.8 | Fixed $\lambda$ |

### Key Findings
- **ASAG outperforms PAG/SAG on both FID and human preference**: FID decreases from 12.8 to 11.2, with a human preference rate of 51.8%.
- **Strong plug-and-play compatibility**: Consistently improves downstream applications including IP-Adapter and ControlNet, with CLIP-I improving by 3.6%.
- **Necessity of Sinkhorn iterations**: Directly adding noise without Sinkhorn yields FID of 12.4, demonstrating the importance of preserving the validity of the attention matrix.
- **Negligible computational overhead**: Additional Sinkhorn iterations per step (typically 5–10) increase inference time by only approximately 3–5%.

## Highlights & Insights
- **Theory over heuristics**: This is the first work to provide a theoretical explanation for attention guidance from an optimal transport perspective, reformulating the question of "why perturbing attention improves generation" within a rigorous OT framework.
- **Elegant adversarial cost design**: "Penalizing the highest-similarity pairs" precisely targets the most informative attention connections, yielding greater efficiency than random or uniform perturbation.
- **Strong transferability**: The method is not tied to any specific model architecture and can be directly applied to any self-attention-based diffusion model.

## Limitations & Future Work
- The method operates only on self-attention layers and does not address cross-attention (text-image cross-attention may admit a similar OT interpretation).
- Sinkhorn iterations introduce a small additional computational cost, which may require optimization in real-time generation scenarios.
- The optimal value of hyperparameter $\lambda$ may vary across models and tasks; an adaptive selection strategy is lacking.
- Validation is limited to text-to-image generation; applications in video generation, 3D generation, and other settings remain unexplored.

## Related Work & Insights
- **vs. PAG (Perturbed Attention Guidance)**: PAG replaces attention with an identity matrix as perturbation, lacking theoretical justification; ASAG's adversarially designed costs based on OT theory are more effective.
- **vs. SAG (Self-Attention Guidance)**: SAG uses a blurred map to guide attention, which is equally heuristic; ASAG provides a principled alternative.
- **vs. CFG**: CFG performs linear extrapolation in the conditional-unconditional direction, while ASAG perturbs at the attention level; the two approaches are orthogonal and can be combined.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The OT perspective on attention guidance is a genuinely novel angle.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple models and applications, though the dataset scope is limited.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear; the plug-and-play nature is appealing to practitioners.
- Value: ⭐⭐⭐⭐ High practical utility; the theoretical contribution lays a foundation for future research on guidance methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Boosting Adversarial Transferability via Ensemble Non-Attention](boosting_adversarial_transferability_via_ensemble_non-attention.md)
- [\[ICLR 2026\] Harpoon: Generalised Manifold Guidance for Conditional Tabular Diffusion](../../ICLR2026/others/harpoon_generalised_manifold_guidance_for_conditional_tabular_diffusion.md)
- [\[AAAI 2026\] Local Guidance for Configuration-Based Multi-Agent Pathfinding](local_guidance_for_configuration-based_multi-agent_pathfinding.md)
- [\[AAAI 2026\] Sampling Control for Imbalanced Calibration in Semi-Supervised Learning](sampling_control_for_imbalanced_calibration_in_semi-supervised_learning.md)
- [\[AAAI 2026\] Controllable Financial Market Generation with Diffusion Guided Meta Agent](controllable_financial_market_generation_with_diffusion_guided_meta_agent.md)

</div>

<!-- RELATED:END -->
