---
title: >-
  [Paper Note] Meta-learning In-Context Enables Training-Free Cross Subject Brain Decoding
description: >-
  [CVPR 2026][3D Vision][Brain decoding] This paper proposes BrainCoDec, a framework that performs fMRI-based visual decoding generalizable to new subjects without any fine-tuning. It employs a two-stage hierarchical in-co…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "Brain decoding"
  - "meta-learning"
  - "in-context learning"
  - "fMRI"
  - "cross-subject generalization"
date: 2026-05-08
content_hash: b535e547c4a6a920
---

# Meta-learning In-Context Enables Training-Free Cross Subject Brain Decoding

**Conference**: CVPR 2026
**arXiv**: [2604.08537](https://arxiv.org/abs/2604.08537)
**Code**: [https://github.com/ezacngm/brainCodec](https://github.com/ezacngm/brainCodec)
**Area**: 3D Vision
**Keywords**: Brain decoding, meta-learning, in-context learning, fMRI, cross-subject generalization

## TL;DR

This paper proposes BrainCoDec, a framework that performs fMRI-based visual decoding generalizable to new subjects without any fine-tuning. It employs a two-stage hierarchical in-context learning approach: first estimating encoder parameters for each voxel, then aggregating across voxels via functional inversion. Top-1 retrieval accuracy improves from 3.9% (MindEye2) to 22.7%.

## Background & Motivation

1. **Background**: fMRI-based visual decoding has achieved significant progress — by learning mappings from brain activity to visual semantic spaces, conditional generative models can reconstruct viewed images from neural signals. Methods such as MindEye2 have achieved high-fidelity reconstruction in single-subject settings.

2. **Limitations of Prior Work**: Existing models cannot generalize across subjects. Due to large inter-individual differences in neural signals (anatomical structure, functional organization, neural plasticity, etc.), training or fine-tuning a dedicated model for each new subject requires substantial data collection and computational resources.

3. **Key Challenge**: Cross-subject differences in neural representations render mapping functions learned for one individual invalid for another. Existing approaches either rely on anatomical alignment (flatmaps) or employ 1D pooling or surface-based learning, all of which implicitly or explicitly require anatomical registration.

4. **Goal**: Achieve zero-shot cross-subject visual decoding — adapting to a new subject using only a small number of examples (e.g., 200 image–brain pairs), without requiring anatomical alignment or stimulus overlap.

5. **Key Insight**: Brain decoding is reformulated as the functional inversion of an encoding model — first estimating per-voxel forward model parameters (image → brain activity) via in-context learning, then inverting this forward model to decode images.

6. **Core Idea**: A meta-optimized Transformer learns the voxel-level encoding function of a new subject in-context, followed by cross-voxel contextual aggregation for functional inversion decoding — all without any gradient updates.

## Method

### Overall Architecture

BrainCoDec operates through two stages of hierarchical inference:

- **Stage 1 (Encoder Parameter Estimation)**: For each voxel, a set of (image embedding, voxel activation) pairs is provided as context, and the pretrained BrainCoRL Transformer infers the response function parameters $\omega_q$ for that voxel. This is repeated independently for all voxels of interest.
- **Stage 2 (Contextual Functional Inversion)**: The encoder parameters $\omega_k$ and the corresponding activations $\beta_k$ for all voxels are concatenated into context tokens $c_k = [\omega_k, \beta_k]$, which are fed into a second Transformer $P_\gamma$ for cross-voxel aggregation, yielding the predicted image embedding $\hat{\mathcal{I}}$.

### Key Designs

1. **Stage 1: In-Context Encoder Parameter Estimation**

    - **Function**: Infer the visual response function parameters for each voxel of a new subject without fine-tuning.
    - **Mechanism**: Following BrainCoRL, for voxel $v_q$, a context $\{(\mathcal{I}_t, \beta_{t,q})\}_{t=1}^n$ is constructed, where $\mathcal{I}_t$ denotes image embeddings (CLIP/DINO/SigLIP) and $\beta_{t,q}$ denotes the response of that voxel to the $t$-th image. Transformer $T_\theta$ takes these pairs as input and outputs voxel parameters: $\omega_q = T_\theta(\{(\mathcal{I}_t, \beta_{t,q})\}_{t=1}^n)$.
    - **Design Motivation**: Each voxel exhibits distinct tuning properties (e.g., selectivity for faces or scenes). Contextual examples enable the model to infer the functional role of a given voxel.

2. **Stage 2: Contextual Functional Inversion**

    - **Function**: Integrate information across multiple voxels to infer image embeddings from brain activity.
    - **Mechanism**: Each voxel is represented as $c_k = [\omega_k, \beta_k]$; tokens from all voxels form a variable-length sequence fed into Transformer $P_\gamma$. A [CLS] token produces the output image embedding. No positional encoding is used to ensure permutation invariance. Logit scaling $\alpha_{\text{scaled}} = \frac{\log(l) \cdot q \cdot k}{\sqrt{d}}$ is applied to handle variable-length contexts.
    - **Design Motivation**: Traditional inversion requires an overdetermined system where the number of voxels far exceeds the embedding dimension. A learned approach can handle underdetermined systems and compensate for biases in encoder estimation.

3. **Three-Stage Training Pipeline**

    - **Function**: Progressively transition from synthetic to real fMRI data for robust training.
    - **Mechanism**: (1) *Pre-training* — synthetic weights and Gaussian noise simulate voxel responses with a fixed context of 200 voxels; (2) *Context extension* — variable-length voxel counts (200–4000, randomly sampled) are introduced to adapt the model to varying context lengths; (3) *Supervised fine-tuning* — training on real fMRI data using leave-one-subject-out cross-validation.
    - **Design Motivation**: This three-stage pipeline mirrors LLM training best practices. Synthetic pre-training provides large-scale training signal; variable-length context training improves generalization; real-data fine-tuning bridges the domain gap.

### Loss & Training

- Combined cosine-contrastive loss: $\mathcal{L} = \mathcal{L}_{\cos} + \alpha \mathcal{L}_{\text{infoNCE}}$, jointly optimizing reconstruction fidelity and instance-level discriminability.
- Embedding vectors are normalized to unit length.
- Evaluation uses nearest-neighbor retrieval (Top-1/Top-5 accuracy, Mean Rank, cosine similarity).

## Key Experimental Results

### Main Results

**Cross-subject decoding on NSD (held-out subjects, CLIP backbone):**

| Method | S1 Top-1 | S2 Top-1 | S5 Top-1 | S7 Top-1 | Mean Top-1 | Mean Top-5 |
|--------|----------|----------|----------|----------|------------|------------|
| MindEye2 (w/ anatomical alignment) | 4.11% | 3.82% | 2.87% | 2.51% | 3.90% | 9.81% |
| TGBD | 1.27% | 0.56% | 0.84% | 0.39% | 0.82% | 3.09% |
| **BrainCoDec-200** | **25.5%** | **22.9%** | **23.2%** | **19.2%** | **22.7%** | **54.0%** |

**Cross-scanner generalization on BOLD5000 (only 20 context images):**

| Backbone | Top-1 Acc | Top-5 Acc | Mean Rank | Cosine Sim |
|----------|-----------|-----------|-----------|------------|
| CLIP | 31.45±12.80% | 81.67±9.42% | 3.49±0.76 | 0.72±0.02 |

### Ablation Study

| Configuration | Cosine Similarity | Note |
|---------------|-------------------|------|
| BrainCoDec (leave-one-subject-out) | ~0.55 | Full model |
| BrainCoDec (no held-out subject) | ~0.56 | Target subject included in training; marginal gain |
| Synthetic pre-training only | ~0.25 | Large gap without real data |
| Gradient inversion | ~0.20 | Direct optimization performs worst |

### Key Findings

- **Decisive improvement over prior methods**: Top-1 accuracy increases from 3.9% (MindEye2) to 22.7%, an approximately 6× gain, without anatomical alignment.
- **High data efficiency**: Only 200 context images and 4,000 voxels are sufficient to approach performance with the full 9,000-image set.
- **Cross-scanner generalization**: Tested directly on BOLD5000 (3T) with a model trained on NSD (7T); 31.45% Top-1 is achieved with only 20 context images.
- **Robustness across functional regions**: Masking category-selective regions (e.g., face-selective FFA) has minimal impact on most categories, indicating that the model learns distributed representations.
- **Interpretable attention maps**: Last-layer attention weights align closely with known functional regions (face stimuli → FFA/EBA; scenes → PPA/OPA/RSC).
- **Negligible gap between leave-one-out and no-held-out settings**: This confirms genuine cross-subject generalization capability.

## Highlights & Insights

- **"Decoding as inversion of encoding"**: Reformulating decoding as forward model estimation followed by inversion leverages the structural information of the encoding model as a strong constraint. This paradigm is transferable to other inverse problems (e.g., image restoration, signal processing).
- **Hierarchical in-context learning**: The two stages perform in-context learning along the "stimulus" and "voxel" dimensions respectively, each with clear semantic meaning — an elegant design. The architecture of voxel-level parallelism combined with functional inversion aggregation naturally accommodates varying numbers of voxels.
- **Synthetic pre-training pipeline**: Pre-training requires no real fMRI data, reducing dependence on expensive neural recordings. The three-stage pipeline of synthetic pre-training → variable-length context training → real-data fine-tuning mirrors LLM training best practices.

## Limitations & Future Work

- **Image embedding decoding only**: Current evaluation is limited to retrieval tasks; end-to-end image reconstruction is not demonstrated (though the paper notes compatibility with IP-Adapter).
- **Context size constraint**: 200 context images still require approximately 20 minutes of fMRI scanning, which may be excessive for clinical applications.
- **Restricted to visual cortex**: Only higher visual cortex voxels are used; whole-brain decoding is not explored.
- **Directions for improvement**: (a) Integrating generative models for end-to-end image reconstruction; (b) Reducing the required number of context images (e.g., 10–50); (c) Extending to more accessible neural signals such as EEG/MEG; (d) Exploring cross-modal decoding (video, speech).

## Related Work & Insights

- **vs. MindEye2**: MindEye2 uses MNI anatomical alignment for cross-subject adaptation but achieves only 3.9% Top-1, far below BrainCoDec's 22.7%. The key difference is that BrainCoDec bypasses anatomical alignment through functional in-context learning.
- **vs. TGBD**: TGBD attempts template-guided brain decoding but achieves only 0.82% Top-1, demonstrating that approaches that ignore subject-specific information perform poorly.
- **vs. BrainCoRL**: Stage 1 of BrainCoDec directly adopts BrainCoRL's encoder parameter estimation; the innovation lies in the addition of the Stage 2 functional inversion decoder, which translates encoding capability into decoding capability.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The hierarchical in-context learning approach to brain decoding is highly original; the formalization of "decoding = inversion of encoding" is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Leave-one-subject-out cross-validation on four NSD subjects, cross-scanner evaluation on BOLD5000, ROI dropout analysis, attention visualization, and multi-backbone validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is clearly articulated, methods are described in detail, and figures are visually refined and highly informative.
- Value: ⭐⭐⭐⭐⭐ Represents a critical step toward a general-purpose brain decoding foundation model; substantial practical performance gains with far-reaching implications for BCI research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Meta-Learning an In-Context Transformer Model of Human Higher Visual Cortex](../../NeurIPS2025/3d_vision/meta-learning_an_in-context_transformer_model_of_human_higher_visual_cortex.md)
- [\[CVPR 2026\] tttLRM: Test-Time Training for Long Context and Autoregressive 3D Reconstruction](tttlrm_test-time_training_for_long_context_and_autoregressive_3d_reconstruction.md)
- [\[CVPR 2026\] Fast3Dcache: Training-free 3D Geometry Synthesis Acceleration](fast3dcache_training-free_3d_geometry_synthesis_acceleration.md)
- [\[CVPR 2026\] Deformation-based In-Context Learning for Point Cloud Understanding](deformation-based_in-context_learning_for_point_cloud_understanding.md)
- [\[CVPR 2026\] Learning Multi-View Spatial Reasoning from Cross-View Relations](learning_multi-view_spatial_reasoning_from_cross-view_relations.md)

</div>

<!-- RELATED:END -->
