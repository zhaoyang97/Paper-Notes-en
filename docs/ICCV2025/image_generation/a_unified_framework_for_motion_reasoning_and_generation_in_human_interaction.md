---
title: >-
  [Paper Note] A Unified Framework for Motion Reasoning and Generation in Human Interaction
description: >-
  [ICCV 2025][Image Generation][human interaction motion] This paper proposes MoLaM, a unified interactive motion-language model that, through a three-stage training strategy and a newly constructed Inter-MT² dataset (82.7…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "human interaction motion"
  - "motion-language model"
  - "multi-turn dialogue"
  - "instruction tuning"
  - "motion reasoning"
date: 2026-05-08
content_hash: ce728437f4f875c4
---

# A Unified Framework for Motion Reasoning and Generation in Human Interaction

**Conference**: ICCV 2025  
**arXiv**: N/A (CVF OpenAccess)  
**Code**: [https://molam-motion-language.github.io/](https://molam-motion-language.github.io/)  
**Area**: Image Generation  
**Keywords**: human interaction motion, motion-language model, multi-turn dialogue, instruction tuning, motion reasoning

## TL;DR

This paper proposes MoLaM, a unified interactive motion-language model that, through a three-stage training strategy and a newly constructed Inter-MT² dataset (82.7K multi-turn instructions), is the first to simultaneously achieve understanding, generation, editing, and reasoning of dyadic interaction motion within a single framework.

## Background & Motivation

Three core limitations of existing motion-language models motivate this work:

**Why is a unified framework needed?** Existing approaches (e.g., MotionGPT, TM2T) predominantly focus on unidirectional tasks (text-to-motion or motion-to-text) and handle only **single-person** motion. In practical applications such as VR social interaction and robotic collaboration, coordinated motion between two or more persons is the central requirement. Two-stage pipelines (first converting motion to text, then reasoning with an LLM) introduce **error accumulation** and **interpretive ambiguity** — a single motion sequence admits multiple plausible interpretations that a single textual description cannot fully capture.

**Why is a new dataset needed?** Existing dyadic interaction motion datasets (InterHuman, Inter-X) provide text annotations but lack instruction data in multi-turn dialogue format, without which a model cannot learn to dynamically adapt motion generation and understanding within a conversational context.

**Why are existing tokenizers insufficient?** Dyadic interaction motion is intrinsically a pair of coupled motion sequences requiring simultaneous encoding of joint positions, velocities, and rotations for both persons. Existing VQ-based tokenizers struggle to accurately capture the relative joint-position relationships between the two persons.

## Method

### Overall Architecture

MoLaM consists of three core components:

1. **Motion Tokenizer**: Based on RQ-VAE, encodes dyadic interaction motion sequences into discrete motion tokens.
2. **Large Language Model (LLM)**: Built on LLaMA-3.1-8B, processes both text and motion modalities in a unified manner.
3. **Motion Decoder**: Reconstructs continuous motion sequences from generated motion tokens.

The key design principle is a **unified vocabulary**: motion tokens and text tokens share the same vocabulary, enabling the LLM to autoregressively generate both text and motion.

### Key Designs

#### Interaction Motion Representation

The motion representation at time step $i$ is defined as $m_i = [j^p_g, j^v_g, j^r, c_f]$, comprising:
- Global joint positions $j^p_g \in \mathbb{R}^{3N_j}$
- Global joint velocities $j^v_g \in \mathbb{R}^{3N_j}$
- 6D representation of local rotations $j^r \in \mathbb{R}^{6N_j}$
- Binary foot contact features $c_f \in \mathbb{R}^4$

**Why RQ-VAE instead of standard VQ-VAE?** RQ-VAE (Residual Quantization VAE) reduces quantization information loss through multi-level residual quantization. The encoder processes motion pairs $\{m_a, m_b\}$ via 2D convolutions along the temporal axis, producing latent vectors $\{z^{1:L}_a, z^{1:L}_b\}$ ($L = M/l$), where each vector is quantized into $D$ ordered discrete codes $RQ(z_i; C, D) = (k^i_1, \cdots, k^i_D) \in [K]^D$.

#### Inter-MT² Dataset Construction

The dataset contains 82.7K multi-turn sessions and 153K interaction motion samples, covering three instruction categories:
- **Motion Editing** (41.7%): modifying the emotional tone or relational dynamics of a motion.
- **Motion Reasoning** (16.7%): inferring the causes or consequences of a motion.
- **Story Generation** (41.7%): creating narratives centered on a given motion.

The data generation pipeline proceeds as follows:
1. Motion-text pairs from Inter-X and InterHuman serve as the foundation.
2. GPT-4o generates multi-turn instructions conditioned on motion labels.
3. InterGEN synthesizes interaction motions matching the new instructions.
4. Real motions (56K) and synthesized motions (96K) are mixed together.

### Loss & Training

A three-stage training strategy is central to MoLaM's effectiveness:

**Stage 1: Motion Tokenizer Training.** The RQ-VAE encoder, decoder, and quantizer are trained to minimize reconstruction loss, codebook alignment loss, and commitment loss. Parameters are frozen after training.

**Stage 2: Cross-Modal Pre-training.** The LLM is continually pre-trained on Inter-X, InterHuman, and a subset of single-person data (Motion-X) using LoRA adapters, with next-token prediction as the objective: $L = -\log \sum_T p_\theta(y_i | y_{<i})$. Single-person data provides prior knowledge for linguistically describing motion.

**Stage 3: Inter-MT² Instruction Tuning.** Instruction tuning is performed on the multi-turn Inter-MT² data, enabling the model to handle complex, multi-turn interaction scenarios.

## Key Experimental Results

### Main Results

**Motion Reasoning Task (Table 2):**

| Method | Logical Consistency↑ | Content Alignment↑ | Naturalness↑ | METEOR | MAUVE |
|------|------------|----------|--------|--------|-------|
| TM2T + LLaMA-3.1 | 3.852 | 3.050 | 6.348 | 0.226 | 0.009 |
| TM2T + GPT-4o | 4.266 | 3.455 | 6.790 | 0.227 | 0.019 |
| MotionGPT* | 1.855 | 1.303 | 3.574 | 0.096 | 0.005 |
| MotionGPT*_I | 3.690 | 3.160 | 5.291 | 0.218 | 0.417 |
| MoLaM w/o Inter-MT² | 2.770 | 2.141 | 4.968 | 0.145 | 0.004 |
| **MoLaM (Ours)** | **5.252** | **4.511** | **6.981** | **0.260** | **0.794** |

MoLaM surpasses the best two-stage method by more than 1.9 points on logical consistency, and MAUVE scores rise dramatically from 0.019 to 0.794.

**Standard Motion Tasks (Table 4):**

| Method | M2T R-Top3↑ | T2M R-Top3↑ | T2M FID↓ | Reaction MPJPE↓ | Reaction FID↓ |
|------|-------------|-------------|----------|----------------|---------------|
| InterGEN | - | 0.645 | 0.078 | - | - |
| MoMask* | - | 0.612 | 0.066 | 1.602 | 0.112 |
| MotionGPT*_I | 0.503 | 0.331 | 0.118 | 1.436 | 0.380 |
| MoLaM w/o Inter-MT² | 0.894 | 0.561 | 0.082 | 0.984 | 0.031 |
| **MoLaM** | **0.901** | **0.568** | **0.059** | **0.691** | **0.019** |

### Ablation Study

**Motion Editing Task Ablation (Table 3):**

| Method | FID↓ | MPJPE↓ |
|------|------|--------|
| TM2T + InterGEN | 0.110 | 0.811 |
| MotionGPT* | 0.251 | 4.002 |
| MotionGPT*_I | 0.161 | 3.982 |
| MoLaM w/o Inter-MT² | 0.080 | 0.908 |
| **MoLaM** | **0.064** | **0.758** |

A user study (30 participants) with MANOVA analysis confirms that MoLaM significantly outperforms the variant without Inter-MT² in content similarity (p=0.010) and instruction alignment (p=0.001).

### Key Findings

1. **Unified architecture > two-stage pipeline**: eliminates error accumulation and interpretive ambiguity.
2. **Inter-MT² dataset is critical**: removing it causes significant degradation across all metrics.
3. **RQ-VAE > VQ tokenizer**: MotionGPT's VQ tokenizer fails to capture precise relative positions between the two persons.
4. **Extensible to multiple persons**: via incremental generation, MoLaM can be extended to interactions involving three or more persons.

## Highlights & Insights

- **Data flywheel thinking**: GPT-4o and InterGEN are leveraged to construct a large-scale synthetic dataset, addressing the scarcity of multi-turn interaction motion data.
- **Elegant unified vocabulary design**: sharing the vocabulary between motion and text avoids the complexity of cross-modal bridging.
- **Progressive logic of three-stage training**: tokenizer → modality alignment → instruction tuning, each stage with a clear objective.
- **Potential for multi-person extension**: the framework is agnostic to the number of persons and is theoretically scalable to group scenarios.

## Limitations & Future Work

- The quality of synthesized motions is bounded by InterGEN's generative capacity, resulting in a gap in top-3 retrieval accuracy (0.701 vs. 0.870 for real data).
- Only SMPL-X joint representation is supported; facial expressions and hand details are excluded.
- The user study involves a relatively small sample (30 participants), which may be insufficient to fully validate the robustness of subjective evaluations.
- Multi-person extension requires incremental generation and lacks genuine joint modeling of all persons simultaneously.

## Related Work & Insights

Compared with MotionChain (multi-turn single-person motion dialogue), MoLaM extends the framework to dyadic interaction scenarios. Unlike the interaction motion work of Wu et al., MoLaM additionally incorporates multi-turn dialogue and complex reasoning capabilities. This work suggests that **large-scale synthetic instruction data combined with a unified multimodal architecture** may serve as a general paradigm for data-scarce domains.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] InfiniDreamer: Arbitrarily Long Human Motion Generation via Segment Score Distillation](infinidreamer_arbitrarily_long_human_motion_generation_via_segment_score_distill.md)
- [\[ICCV 2025\] PINO: Person-Interaction Noise Optimization for Long-Duration and Customizable Motion Generation of Arbitrary-Sized Groups](pino_person-interaction_noise_optimization_for_long-duration_and_customizable_mo.md)
- [\[ICCV 2025\] ScoreHOI: Physically Plausible Reconstruction of Human-Object Interaction via Score-Guided Diffusion](scorehoi_physically_plausible_reconstruction_of_human-object_interaction_via_sco.md)
- [\[NeurIPS 2025\] Toward a Unified Geometry Understanding: Riemannian Diffusion Framework for Graph Generation and Prediction](../../NeurIPS2025/image_generation/toward_a_unified_geometry_understanding_riemannian_diffusion_framework_for_graph.md)
- [\[ICCV 2025\] SMGDiff: Soccer Motion Generation using Diffusion Probabilistic Models](smgdiff_soccer_motion_generation_using_diffusion_probabilistic_models.md)

</div>

<!-- RELATED:END -->
