---
title: >-
  [Paper Note] Topo-R1: Detecting Topological Anomalies via Vision-Language Models
description: >-
  [CVPR 2025][Multimodal VLM][Topological Anomaly Detection] This work reveals that existing VLMs (including GPT-5.2 and Gemini-2.5) exhibit near-zero performance on topological anomaly detection ($F1@0.5 < 1.5\%$). It proposes the Topo-R1 framework, which endows VLMs with topological awareness via SFT + GRPO incorporating a topology-aware composite reward (integrating type-aware Hungarian matching and clDice), achieving a peak $F1@0.5$ of 45.2%.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Topological Anomaly Detection"
  - "Tubular Structures"
  - "Reinforcement Learning"
  - "GRPO"
  - "clDice"
date: 2026-05-08
content_hash: 2fe3a63531cb70af
---

# Topo-R1: Detecting Topological Anomalies via Vision-Language Models

**Conference**: CVPR 2025  
**arXiv**: [2603.13054](https://arxiv.org/abs/2603.13054)  
**Code**: Coming soon  
**Area**: Multimodal VLM  
**Keywords**: Topological Anomaly Detection, Tubular Structures, Reinforcement Learning, GRPO, clDice

## TL;DR

This work reveals that existing VLMs (including GPT-5.2 and Gemini-2.5) exhibit near-zero performance on topological anomaly detection ($F1@0.5 < 1.5\%$). It proposes the Topo-R1 framework, which endows VLMs with topological awareness via SFT + GRPO incorporating a topology-aware composite reward (integrating type-aware Hungarian matching and clDice), achieving a peak $F1@0.5$ of 45.2%.

## Background & Motivation

**Background**: Although many topology-preserving segmentation methods (utilizing losses such as clDice and Betti matching) exist for tubular structures (e.g., blood vessels, nerve fibers, road networks), they heavily rely on annotated training data and suffer from poor cross-domain transferability.

**Limitations of Prior Work**: When deploying to new domains without annotated data, it is impossible to automatically detect topological errors in segmentation results. Topological errors are highly subtle—a single missing pixel can sever a vessel, yet pixel-level metrics remain virtually unaffected (e.g., Dice can reach 0.91 despite topological failure).

**Key Challenge**: VLMs are natural candidates as general-purpose visual reasoning tools. However, experiments demonstrate that all SOTA VLMs (both closed-source and open-source) perform near randomly on topological anomaly detection (a "needle in a haystack" problem: finding highly sparse topological errors within densely connected networks).

**Goal**: How to endow VLMs with the capability to perceive topological anomalies?

**Key Insight**: Redefining topological anomaly detection as a structured visual reasoning task (localizing and classifying four types of topological errors), and equipping the VLM through automated data generation, SFT, and GRPO.

**Core Idea**: Training VLMs to perceive topological anomalies in tubular structures by employing topology-aware RL rewards (clDice + type-aware Hungarian matching).

## Method

### Overall Architecture

Input: Original image and binary segmentation mask. Output: A structured detection set $\{(bbox, error\_type)\}$. Two-stage training: SFT (elevating performance from near-zero to a baseline level) $\rightarrow$ GRPO (further enhancing precision and recall using a topology-aware composite reward).

### Key Designs

1. **Four-Class Topological Error Taxonomy**:

    - Organized along two orthogonal axes: connectivity errors (affecting $\beta_0$) vs. branching errors (affecting branching complexity).
    - **Broken connection**: Disconnects continuous segments, increasing $\beta_0$.
    - **Spurious connection**: Incorrectly bridges different segments, decreasing $\beta_0$ or creating loops (increasing $\beta_1$).
    - **Missing branch**: Missing terminal branches.
    - **Extra branch**: False branches.
    - **Design Motivation**: Exhaustiveness—every local topological perturbation falls precisely into one category; Verifiability—automatically verified via changes in Betti numbers.

2. **Automated Data Generation Pipeline**:

    - Three domain data sources: road networks (60%), crack detection (20%), and retinal blood vessels (20%).
    - Controlled topological errors are injected into clean masks, with operations performed on morphological skeletons.
    - Betti Number Verification: Computing $(\beta_0, \beta_1)$ before and after injection to confirm that a true topological change has occurred.
    - Difficulty Curriculum: 0 errors (20%) $\rightarrow$ 1 error (20%) $\rightarrow$ 2-5 errors (40%) $\rightarrow$ 6-10 errors (20%).
    - Final Dataset: 12.9K SFT samples + 50.3K RL samples + 4.2K test samples.

3. **Topology-Aware Composite Reward (Core Contribution)**:

    - $R_{total} = 0.10 \cdot R_{fmt} + 0.85 \cdot R_{acc} + 0.05 \cdot R_{topo}$
    - **Type-aware Hungarian Matching**: Performs optimal bipartite matching grouped by error types, ensuring that a prediction counts as a TP only if both its type and location are correct.
    - **Accuracy Reward** = Detection $F1$ (soft TP) + Localization Quality + Type Coverage.
    - **clDice Reward**: For successfully matched detection regions, the skeleton overlap between the corrupted mask and the GT mask is calculated. Regions with topological errors exhibit lower clDice $\rightarrow$ leading to higher reward. Only matches with correct types receive this reward.
    - Piecewise continuous $IoU \to Score$ mapping $\phi(IoU)$: Provides dense intermediate reward signals.

### Loss & Training

- Based on Qwen2.5-VL-3B / Qwen3-VL-4B/8B / InternVL-2.5-2B.
- SFT: Full-parameter fine-tuning with 12.9K samples.
- GRPO: Sampling $G$ candidate outputs, evaluating them with the composite reward, and updating via a relative advantage reinforcement learning strategy.

## Key Experimental Results

### Main Results

| Model | Method | F1@0.3 | F1@0.5 | F1@0.75 | aF1 | mPS-F1@0.5 |
|------|------|--------|--------|---------|-----|-----------|
| GPT-5.2 | Zero-shot | 3.2 | 1.5 | — | — | 8.6 |
| Gemini-2.5-Flash | Zero-shot | — | — | — | — | 10.5 |
| Qwen3-VL-4B | Zero-shot | 0.1 | 0.0 | 0.0 | 0.0 | 0.9 |
| Qwen3-VL-4B | SFT | 31.9 | 23.0 | 12.1 | 12.8 | 37.7 |
| **Qwen3-VL-4B** | **Topo-R1** | **58.3** | **45.2** | **22.5** | **24.7** | **58.5** |
| Qwen2.5-VL-3B | Topo-R1 | 57.8 | 43.0 | 18.4 | 21.4 | 56.2 |

### Ablation Study (Reward Design)

| Reward Configuration | F1@0.5 | mPS-F1@0.5 |
|------------|--------|-----------|
| Raw IoU (without piecewise mapping) | 14.9 | — |
| Piecewise IoU mapping (Ours) | **43.0** | **56.2** |

### Key Findings

- **All VLMs operate near-randomly under zero-shot settings**: Even GPT-5.2 only achieves $F1@0.5 \approx 1.5\%$, and in-context learning is similarly ineffective (at most 0.5%).
- **SFT provides a necessary foundation but is insufficient**: SFT enables the model to learn the basic error classification taxonomy, but it lacks exploration capability, frequently resulting in empty predictions.
- **GRPO yields decisive improvements**: Performance escalates from 23.0% under SFT to 45.2% $F1@0.5$ via Topo-R1 (+22% absolute gain), with precision showing particularly substantial growth, demonstrating that RL guides the model to perform more precise detection.
- **Topology-aware rewards are irreplaceable**: Removing either the piecewise IoU mapping or the clDice reward leads to significant performance degradation.
- **Model scale is not the decisive factor**: The 3B Topo-R1 outperforms all closed-source LLMs by an order of magnitude, indicating that the key to topological awareness lies in the training methodology rather than model size.

## Highlights & Insights

- **An excellent paradigm of "exposing VLM failures, then teaching them"**: First demonstrating the complete failure (near-zero performance) of SOTA VLMs on a specific task, followed by a systematic solution. This methodology is universally applicable to research aiming to instill new capabilities into VLMs.
- **Using clDice as an RL reward is an elegant design**: Topological correctness is inherently about skeletal connectivity; clDice perfectly measures skeleton overlap and is only activated upon correct type matching, preventing incorrect detections from being rewarded.
- **Automated validation of data quality via Betti numbers**: Mathematically verifiable data annotations are inherently more reliable than manual annotations.
- **Cross-domain generalization potential**: Although training is conducted on roads, blood vessels, and cracks, the framework can be directly extended to any tubular structures such as nerve fibers and lymphatic vessels.

## Limitations & Future Work

- **Room for absolute performance improvement remains**: An $F1@0.5$ of 45.2% is still far from practical utility, especially with $F1@0.75$ sitting at only 22.5%.
- **Limited to 2D tubular structures**: Topological errors in 3D data (e.g., 3D angiographies) present much higher complexity.
- **Limitations of the four-class error taxonomy**: Certain complex topological changes may involve multiple concurrent error classes.
- **Dependency on segmentation mask quality**: The approach assumes the availability of ready-to-use segmentation masks; poor mask quality can degrade anomaly detection performance.

## Related Work & Insights

- **vs. AnomalyGPT / MMAD**: Existing VLMs for industrial anomaly detection focus on texture and appearance anomalies, without addressing topological structures. Topo-R1 tackles structural anomalies, which are vastly more challenging and critical.
- **vs. clDice loss**: clDice was originally used as a loss function for training segmentation networks. Redesigning it here as an RL reward represents an elegant transition from a training objective to an evaluation signal.
- **Insight**: The paradigm of embedding domain-specific mathematical invariants (such as Betti numbers) into RL rewards can be generalized to other structure-preserving tasks.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Brand-new task definition + pioneering exposure of topological perception deficits in VLMs + innovative topological reward design.
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 open-source + 4 closed-source models, detailed ablation study, comprehensive comparison across zero-shot, few-shot, SFT, and RL settings.
- Writing Quality: ⭐⭐⭐⭐ Equation-dense yet logically lucid, with highly rigorous problem formulation.
- Value: ⭐⭐⭐⭐ Establishes a new paradigm for topological quality assessment in medical imaging, remote sensing, and autonomous driving.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] ASAP: Advancing Semantic Alignment Promotes Multi-Modal Manipulation Detecting and Grounding](asap_advancing_semantic_alignment_promotes_multi-modal_manipulation_de.md)
- [\[ICLR 2026\] Shuffle-R1: Efficient RL Framework for Multimodal Large Language Models via Data-centric Dynamic Shuffle](../../ICLR2026/multimodal_vlm/shuffle-r1_efficient_rl_framework_for_multimodal_large_language_models_via_data-.md)
- [\[ICLR 2026\] Detecting Misbehaviors of Large Vision-Language Models by Evidential Uncertainty Quantification](../../ICLR2026/multimodal_vlm/detecting_misbehaviors_of_large_vision-language_models_by_evidential_uncertainty.md)
- [\[CVPR 2025\] FastVLM: Efficient Vision Encoding for Vision Language Models](fastvlm_efficient_vision_encoding_for_vision_language_models.md)
- [\[CVPR 2025\] What's in the Image? A Deep-Dive into the Vision of Vision Language Models](whats_in_the_image_a_deep-dive_into_the_vision_of_vision_language_models.md)

</div>

<!-- RELATED:END -->
