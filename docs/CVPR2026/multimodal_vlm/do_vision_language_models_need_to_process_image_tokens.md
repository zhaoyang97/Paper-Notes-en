---
title: >-
  [Paper Note] Do Vision Language Models Need to Process Image Tokens?
description: >-
  [CVPR 2026][Multimodal VLM][Vision-Language Model] This paper systematically reveals that image token representations in VLMs stabilize in shallow layers and are interchangeable across layers, while text tokens undergo continuous dynamic reconstruction—the necessity of image processing depth depends highly on the output task type.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Vision-Language Model
date: 2026-05-08
content_hash: 5276e5b02aeb021f
---
# Do Vision Language Models Need to Process Image Tokens?

**Conference**: CVPR 2026  
**arXiv**: [2604.09425](https://arxiv.org/abs/2604.09425)  
**Code**: Available  
**Area**: Multimodal VLM  
**Keywords**: Vision-Language Models, Image tokens, Representation analysis, Computational efficiency, Modal redundancy

## TL;DR

This paper systematically reveals that image token representations in VLMs stabilize in shallow layers and are interchangeable across layers, while text tokens undergo continuous dynamic reconstruction—the necessity of image processing depth depends highly on the output task type.

## Background & Motivation

**Background**: VLMs achieve multimodal reasoning by combining vision encoders with LLMs. However, processing dense image tokens through deep Transformer layers incurs significant computational overhead. Recent studies suggest that visual signals might be inefficiently utilized in multimodal tasks.

**Limitations of Prior Work**: It remains unclear whether visual tokens continue to provide meaningful informational transformations across deep VLM layers. Previous works mainly assumed visual redundancy and designed pruning mechanisms but lacked a systematic understanding of representation dynamics.

**Key Challenge**: VLMs apply the same processing depth to both image and text tokens, yet the representation evolution patterns of these two modalities may be fundamentally different.

**Goal**: To systematically analyze the evolution, interchangeability, task dependency, and recoverability of image tokens in VLMs from a representation perspective.

**Key Insight**: Utilize three metrics—matrix entropy, intrinsic dimension, and trajectory curvature—to track the structural evolution of representations across models ranging from 3B to 72B.

**Core Idea**: Image representations rapidly converge to a bounded complexity region in shallow layers; deep processing primarily maintains rather than reconstructs visual information.

## Method

### Overall Architecture

The paper does not propose a new model but decomposes the question "whether image tokens need to traverse all Transformer layers" into observable and verifiable experiments. The analysis chain answers five progressive questions: How do image and text token representations evolve between layers (RQ1)? If image representations stabilize early, does this stability imply functional interchangeability across depths (RQ2)? Is the processing depth of image tokens equally important for different tasks (RQ3)? Can performance be recovered through fine-tuning after cutting deep image processing (RQ4)? And can generating longer reasoning chains compensate for reduced visual processing (RQ5)? The first two questions are addressed with representation geometry metrics and substitution experiments, while the latter three are addressed with "truncation + recovery" intervention experiments.

```mermaid
graph TD
    A["VLM (3B–72B) + Multimodal Input<br/>Image tokens / Text tokens"] --> B["Three-Metric Representation Analysis (RQ1)<br/>Matrix Entropy · Intrinsic Dim · Trajectory Curvature"]
    B -->|Image stabilizes early, Text reconstructs continuously| C["Layer Substitution Protocol (RQ2)<br/>Shallow image representations replace deep ones"]
    C -->|Output semantics ≈ Unchanged → Functionally interchangeable| D["Visual Depth Truncation Analysis (RQ3)<br/>Remove image tokens after layer lc"]
    D -->|Single-token prediction (MCQ)| E["Robust, early truncation possible"]
    D -->|Multi-token generation (Caption)| F["Sensitive, requires deeper visual processing"]
    E --> G["Distillation LoRA Fine-tuning Recovery (RQ4)<br/>Approximating full model output"]
    F --> G
    G -->|Coarse semantics recoverable / Precise alignment partially irreversible| H["Reasoning Chain Compensation Analysis (RQ5)<br/>Can longer CoT offset reductions?"]
```

### Key Designs

**1. Three-Metric Representation Analysis Framework: Quantifying layer-wise transformations geometrically**

To determine if image tokens undergo "meaningful transformation" in deep layers, the authors track three complementary geometric quantities. Matrix entropy characterizes the spectral concentration of representations within a layer—low entropy indicates compression into few principal directions, while high entropy suggests dispersion. Intrinsic dimension estimates the effective degrees of freedom used by the local manifold, reflecting the complexity upper bound. Trajectory curvature quantifies the magnitude of representation direction reconstruction between adjacent layers, defined as:

$$\bar{C}_l = \frac{1}{N}\sum_i \arccos\!\left(\frac{\langle v_l^{(i)}, v_{l-1}^{(i)}\rangle}{\|v_l^{(i)}\|\,\|v_{l-1}^{(i)}\|}\right)$$

Where $v_l^{(i)}$ is the update direction of the $i$-th token at the $l$-th layer. Results were highly consistent: image token entropy and intrinsic dimension converge rapidly in shallow layers with near-constant curvature, while text token metrics fluctuate, scale, and turn sharply throughout.

**2. Layer Substitution Protocol: Distinguishing structural stability from functional interchangeability**

Geometric stability does not strictly imply functional interchangeability. This study uses an intervention experiment to distinguish them. A hybrid hidden state $Z_{hybrid} = (Z_{l_a}^{img}, Z_{l_b}^{txt})$ is constructed: image token representations from shallow layer $l_a$ are concatenated with text token representations from deep layer $l_b$ for subsequent forward propagation. If stabilization implies interchangeability, substituting deep image tokens with shallow ones should not alter output semantics. Experiments confirmed this: output semantic similarity remained stable at approximately 1.0 regardless of the layer gap $|l_a - l_b|$ for image tokens, whereas it dropped significantly for text tokens.

**3. Visual Depth Truncation Analysis: Distinguishing interchangeability from discardability and exposing task dependency**

Interchangeability does not mean image tokens can be discarded. To test this, image token activations are removed after a truncation layer $l_c$. A key finding is that degradation depends on the output structure: single-token predictions (e.g., MCQ) are robust to early truncation, while multi-token generation (e.g., Captioning) is highly sensitive, with metrics like BLEU improving monotonically with visual depth. This explains why the necessity of visual processing depends on whether the task requires "pointing" or "describing."

### Loss & Training

Regarding recovery after truncation (RQ4), the authors employ distillation-based LoRA fine-tuning. The full model output $y_{target} = f_{base}(x)$ serves as the target to optimize the truncated model $\tilde{f}_K$. Results show that coarse semantics (e.g., Captioning) can be effectively redistributed to remaining layers, but tasks requiring precise visual alignment (e.g., ChartQA) show limited recovery, indicating that deep visual alignment capabilities are partially irreversible.

## Key Experimental Results

### Main Results

| Experiment | Image Tokens | Text Tokens |
|------|-----------|-----------|
| Matrix Entropy | Rapidly stabilizes | Continuously fluctuates |
| Intrinsic Dimension | Early convergence | Alternating scaling |
| Trajectory Curvature | Near-constant | Large and variable |
| Substitution Similarity | ~1.0 (Depth independent) | Decreases with layer gap |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| MCQ Truncation | Smooth degradation | Single-token prediction is robust |
| VQA Truncation | Significant degradation | Exact matching requires deep processing |
| Caption Truncation | Severe degradation | Multi-token generation is most sensitive |
| Post-Distillation | Good Caption recovery | Coarse semantics can be redistributed |
| Post-Distillation | Poor ChartQA recovery | Precise visual alignment is irreversible |

### Key Findings

- Image representations exhibit a consistent early stabilization pattern across all 6 model families (3B-72B), indicating this is a structural property of multimodal Transformers rather than a scale-dependent phenomenon.
- Under deterministic decoding, reducing visual depth perturbs intermediate reasoning trajectories more than the final output—image tokens influence reasoning structure more than final conclusions.
- Fine-tuning not only recovers average performance but also reduces variability across different decoding strategies.

## Highlights & Insights

- **Systematic Evidence of Modal Asymmetry**: Three independent metrics consistently reveal structural asymmetry where visual tokens converge early while text tokens evolve continuously.
- **Distinction between "Interchangeable" and "Discardable"**: Functional interchangeability implies deep processing does not change semantics, but it does not mean image tokens are not required in deep layers to maintain context.
- **Fine-grained Task Dependency Analysis**: Differences in visual depth requirements for single-token prediction, multi-token generation, and open-ended reasoning provide specific guidance for VLM architecture design.

## Limitations & Future Work

- Analysis was primarily conducted on a finite set of datasets like BLINK and Flickr8K.
- Truncation experiments used "hard truncation" (complete removal), without exploring gentler strategies like progressive sparsification.
- The indirect influence of image tokens on text tokens via attention mechanisms was not fully decomposed.

## Related Work & Insights

- **vs FiT/SparseVLM**: While those works assume redundancy to design pruning, this paper explains *why* pruning works from a representation perspective.
- **vs ShortV**: Whereas ShortV explored the limited novelty of deep visual representations, this paper provides a more comprehensive analysis of representation dynamics.

## Rating

- Novelty: ⭐⭐⭐⭐ New perspective for understanding VLM efficiency via representation dynamics.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 6 model families × multiple tasks × multiple metrics; very systematic.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear and elegant structure driven by research questions.
- Value: ⭐⭐⭐⭐ Significant implications for future VLM architecture design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] What Do Visual Tokens Really Encode? Uncovering Sparsity and Redundancy in Multimodal Large Language Models](what_do_visual_tokens_really_encode_uncovering_sparsity_and_redundancy_in_multim.md)
- [\[CVPR 2026\] Hierarchical Process Reward Models are Symbolic Vision Learners](hierarchical_process_reward_models_are_symbolic_vision_learners.md)
- [\[ACL 2026\] What Do Vision-Language Models Encode for Personalized Image Aesthetics Assessment?](../../ACL2026/multimodal_vlm/what_do_vision-language_models_encode_for_personalized_image_aesthetics_assessme.md)
- [\[CVPR 2026\] WeMMU: Enhanced Bridging of Vision-Language Models and Diffusion Models via Noisy Query Tokens](wemmu_enhanced_bridging_of_vision-language_models_and_diffusion_models_via_noisy.md)
- [\[CVPR 2026\] Grounding Everything in Tokens for Multimodal Large Language Models](grounding_everything_in_tokens_for_multimodal_large_language_models.md)

</div>

<!-- RELATED:END -->
