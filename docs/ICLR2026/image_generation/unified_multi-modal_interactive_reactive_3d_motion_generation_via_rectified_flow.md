---
title: >-
  [Paper Note] Unified Multi-Modal Interactive & Reactive 3D Motion Generation via Rectified Flow
description: >-
  [ICLR 2026][Image Generation][Dyadic motion generation] DualFlow proposes the first unified framework for dyadic interactive/reactive 3D motion generation under text+music multi-modal conditions via Rectified Flow and Re…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "Dyadic motion generation"
  - "Rectified Flow"
  - "Retrieval-Augmented Generation"
  - "Contrastive learning"
  - "Multi-modal conditioning"
date: 2026-05-08
content_hash: e3ded0ccbfe4520a
---

# Unified Multi-Modal Interactive & Reactive 3D Motion Generation via Rectified Flow

**Conference**: ICLR 2026
**arXiv**: [2509.24099](https://arxiv.org/abs/2509.24099)
**Code**: [https://gprerit96.github.io/dualflow-page](https://gprerit96.github.io/dualflow-page)
**Area**: 3D Motion Generation
**Keywords**: Dyadic motion generation, Rectified Flow, Retrieval-Augmented Generation, Contrastive learning, Multi-modal conditioning

## TL;DR
DualFlow proposes the first unified framework for dyadic interactive/reactive 3D motion generation under text+music multi-modal conditions via Rectified Flow and Retrieval-Augmented Generation (RAG). It introduces contrastive flow matching and synchronization loss, achieving 2.5% FID improvement and 76% R-precision improvement on the MDD dataset, with 2.5× faster inference.

## Background & Motivation

**Background**: Dyadic motion generation is essential for VR/AR, game AI, and human-robot collaboration. Existing methods treat interactive (simultaneous two-person generation) and reactive (generating person B's motion conditioned on person A's motion) as separate tasks with incompatible architectures.

**Limitations of Prior Work**: (1) Interactive and reactive models employ different architectures and training objectives, precluding seamless task switching; (2) existing methods support only single-modal conditioning (text or music) and cannot handle joint conditioning; (3) diffusion-based methods require 50+ denoising steps, resulting in slow inference.

**Key Challenge**: Dyadic motion requires simultaneously modeling mutual responsiveness between two persons, physical plausibility, and multi-modal signal alignment, yet existing methods lack unified modeling capability.

**Goal**: How to unify interactive and reactive motion generation within a single architecture while supporting text+music multi-modal conditioning?

**Key Insight**: Leveraging the straight transport paths of Rectified Flow for fast inference, switching between tasks via symmetric/asymmetric masking mechanisms, and providing semantic guidance through RAG.

**Core Idea**: Unify task switching via a dual-branch architecture with cascaded DualFlow blocks, combined with a contrastive Rectified Flow objective and an LLM-decomposed RAG module for multi-modal semantic alignment.

## Method

### Overall Architecture
Inputs consist of text (encoded via CLIP-L/14), music (encoded via Jukebox), and motion sequences. Twenty cascaded DualFlow blocks process dyadic motion latents. Under the interactive setting, both branches are symmetrically activated; under the reactive setting, only the reactor branch is activated and conditioned on the actor's motion via causal cross-attention.

### Key Designs

1. **Multi-Modal Motion Retrieval (RAG)**:

    - Function: Provides semantic anchors for dyadic motion generation.
    - Mechanism: GPT-4o decomposes text descriptions into three dimensions—spatial relationships, body actions, and rhythm. Separate CLIP retrieval databases $(D^S, D^B, D^R)$ and a music retrieval database $D^M$ (Jukebox features) are constructed. The similarity score $s_i^q = \langle f_i^q, f_p^q \rangle \cdot e^{-\lambda \cdot \frac{|l_i - l_p|}{\max\{l_i, l_p\}}}$ accounts for both semantic similarity and temporal compatibility.
    - Design Motivation: Direct retrieval from raw text overlooks the nuanced dimensions of interactive motion; LLM-based decomposition improves retrieval quality.

2. **Contrastive Rectified Flow**:

    - Function: Enhances semantic alignment within the flow matching framework.
    - Mechanism: The standard flow loss $\mathcal{L}_{\text{flow}} = \mathbb{E}[\|\mathbf{v}_\theta(\mathbf{x}_t, t, c) - (\mathbf{x}_0 - \epsilon)\|_2^2]$ is augmented with a triplet contrastive loss $\mathcal{L}_{\text{triplet}} = \mathbb{E}[\max(0, d(\hat{\mathbf{v}}, \mathbf{v}^+) - d(\hat{\mathbf{v}}, \mathbf{v}^-) + m)]$.
    - Contrastive sample construction: Leveraging the hierarchical structure of RAG, positive samples are motions of similar style or text description, while negative samples are motions with large style differences or low text similarity (>0.6).

3. **Synchronization Loss**:

    - Function: Enforces spatial relationship consistency between the two persons.
    - Mechanism: Weighted inter-person joint distance loss $\mathcal{L}_{\text{sync}} = \sum_{j_1,j_2} w_d(j_1,j_2) w_j(j_1,j_2) \|d_p(j_1,j_2) - d_{gt}(j_1,j_2)\|^2$.
    - Distance weights $w_d$ assign higher importance to joint pairs that are naturally closer; anatomical weights $w_j$ differentiate body regions such as hands, upper limbs, and lower limbs.

4. **Task Switching Mechanism**:

    - Interactive: Both branches are symmetrically activated; Motion Cross-Attention coordinates motion between the two persons.
    - Reactive: The actor branch is masked; motion cross-attention is replaced by causal cross-attention with a Look-Ahead window of $L=10$.

### Loss & Training
$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CRF}} + \lambda_{\text{geo}} \mathcal{L}_{\text{geo}} + \lambda_{\text{inter}} \mathcal{L}_{\text{inter}}$, where $\mathcal{L}_{\text{CRF}} = \mathcal{L}_{\text{flow}} + \lambda_{\text{triplet}} \mathcal{L}_{\text{triplet}}$, $\mathcal{L}_{\text{geo}}$ comprises foot contact, joint velocity, and bone length losses, and $\mathcal{L}_{\text{inter}}$ comprises distance map, relative orientation, and synchronization losses.

## Key Experimental Results

### Main Results on MDD Dataset (Duet Task)

| Method | R-Prec@3↑ | FID↓ | MMDist↓ | BAS↑ |
|--------|-----------|------|---------|------|
| MDM(Both) | 0.163 | 1.739 | 2.244 | 0.190 |
| InterGen(Both) | 0.302 | 0.426 | 1.532 | 0.185 |
| **DualFlow(Both)** | **0.513** | **0.415** | **0.513** | 0.200 |
| GT | 0.522 | 0.065 | 0.077 | 0.170 |

### Reactive Task

| Method | FID↓ | MMDist↓ | R-Prec@3↑ |
|--------|------|---------|-----------|
| DuoLando(Both) | — | — | — |
| **DualFlow(Both)** | **0.686** | **1.056** | **Best** |

### Key Findings
- DualFlow requires only 20 inference steps (vs. the standard 50-DDIM), yielding a 2.5× speedup.
- Interactive task: FID improved by 2.5%, R-precision improved by 76%, MMDist improved by 3×.
- Reactive task: FID improved by 1.7%, R-precision improved by 2.5×.
- Ablation studies confirm that both the contrastive loss and the RAG module contribute significantly.
- The synchronization loss effectively improves temporal coordination between the two persons.

## Highlights & Insights
- **First unified dyadic motion generation framework**: Seamless switching between interactive and reactive tasks via a masking mechanism eliminates the need to maintain two separate systems.
- **Novel adaptation of RAG to dyadic scenarios**: LLM-based decomposition of text into spatial relationships, body actions, and rhythm is an elegant solution for handling interaction descriptions.
- **Practical advantage of Rectified Flow**: High-quality results are achievable in just 20 inference steps, making the approach suitable for real-time applications.

## Limitations & Future Work
- Reliance on GPT-4o for text decomposition introduces additional computational cost and API dependency.
- The current framework supports only dyadic scenarios; extension to multi-person (>2) settings requires architectural modifications.
- Motion quality evaluation relies on automated metrics; perceptual quality assessment requires further user studies.
- Extending DualFlow to fine-grained hand motion generation is a promising direction.

## Related Work & Insights
- **vs. InterGen**: This diffusion-based dyadic model requires 50 denoising steps, whereas DualFlow requires only 20 steps while achieving superior performance.
- **vs. MDM**: Direct extension of single-person diffusion models to dyadic scenarios performs poorly due to the absence of interaction modeling.
- **First dyadic RAG**: Unlike existing single-person motion RAG methods, DualFlow introduces an interaction-aware retrieval mechanism.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of unified interactive/reactive generation, multi-modal conditioning, and dyadic RAG is unprecedented.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets, multiple settings, and thorough ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Well-structured, though some sections are overly dense.
- Value: ⭐⭐⭐⭐ Represents a clear advancement for the dyadic motion generation field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Free Lunch for Stabilizing Rectified Flow Inversion](free_lunch_for_stabilizing_rectified_flow_inversion.md)
- [\[ICLR 2026\] Multi-agent Coordination via Flow Matching](multi-agent_coordination_via_flow_matching.md)
- [\[CVPR 2026\] InterEdit: Navigating Text-Guided Multi-Human 3D Motion Editing](../../CVPR2026/image_generation/interedit_navigating_textguided_multihuman_3d_moti.md)
- [\[ICCV 2025\] MotionDiff: Training-Free Zero-Shot Interactive Motion Editing via Flow-Assisted Multi-View Diffusion](../../ICCV2025/image_generation/motiondiff_training-free_zero-shot_interactive_motion_editing_via_flow-assisted_.md)
- [\[ICLR 2026\] Image Can Bring Your Memory Back: A Novel Multi-Modal Guided Attack against Image Generation Model Unlearning](image_can_bring_your_memory_back_a_novel_multi-modal_guided_attack_against_image.md)

</div>

<!-- RELATED:END -->
