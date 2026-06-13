---
title: >-
  [Paper Note] Topo-R1: Detecting Topological Anomalies via Vision-Language Models
description: >-
  [CVPR 2026][Multimodal VLM][Topological anomaly detection] Topo-R1 is proposed as the first framework to equip VLMs with topology-aware perception. Through an automated data construction pipeline combined with SFT and GR…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Topological anomaly detection"
  - "tubular structure segmentation"
  - "GRPO reinforcement learning"
  - "clDice"
  - "VLM fine-grained perception"
date: 2026-05-08
content_hash: 5464318730266d26
---

# Topo-R1: Detecting Topological Anomalies via Vision-Language Models

**Conference**: CVPR 2026
**arXiv**: [2603.13054](https://arxiv.org/abs/2603.13054)  
**Code**: To be confirmed  
**Area**: Multimodal VLM
**Keywords**: Topological anomaly detection, tubular structure segmentation, GRPO reinforcement learning, clDice, VLM fine-grained perception

## TL;DR
Topo-R1 is proposed as the first framework to equip VLMs with topology-aware perception. Through an automated data construction pipeline combined with SFT and GRPO reinforcement learning (incorporating a topology-aware composite reward), it enables annotation-free topological anomaly detection and classification in tubular structures.

## Background & Motivation
**Background**: Topological correctness in tubular structures (blood vessels, nerve fibers, road networks) is critical. Existing topology-preserving segmentation methods (persistent homology losses, clDice, etc.) rely on pixel-level annotations to constrain training losses.

**Limitations of Prior Work**: Topological annotation demands domain expertise and is extremely time-consuming; cross-domain transfer is difficult (retinal annotations do not apply to road networks); detecting topological errors in unannotated new domains at deployment time is infeasible.

**Key Challenge**: Topological anomalies are extremely sparse and localized — among thousands of correct pixels, a single missing pixel may sever a vascular connection. Detecting such "needle-in-a-haystack" errors requires a combination of global structural reasoning and local fine-grained perception, capabilities entirely absent in existing VLMs.

**Goal**: Enable VLMs to localize and classify topological errors in tubular structures without domain-specific annotations.

**Key Insight**: Topological anomaly detection is reformulated as a structured visual reasoning task — given an image and a segmentation mask, the model must output bounding boxes with type labels.

**Core Idea**: An automated data pipeline synthesizes topologically verified anomalies, and a GRPO reinforcement learning procedure incorporating type-aware Hungarian matching and a clDice reward is used to train the VLM.

## Method

### Overall Architecture
Two-stage training: Stage 1 uses synthetic data for SFT to bootstrap the VLM from a random initialization; Stage 2 applies GRPO reinforcement learning with a topology-aware composite reward for further optimization. The input consists of an image, a segmentation mask, and a detection prompt; the output is a structured error list of bounding boxes with error types.

### Key Designs

1. **Automated Data Construction Pipeline**:

    - **Function**: Aggregates cross-domain data and injects verifiable topological anomalies.
    - **Mechanism**: Aggregates data from three domains — road networks (60%), crack detection (20%), and retinal vessels (20%); injects four error types (broken connections, false connections, missing branches, spurious branches) onto mask skeletons; automatically verifies correctness via Betti number changes $(\beta_0, \beta_1)$.
    - **Design Motivation**: Manual topological annotation is prohibitively costly. Automated synthesis with Betti number verification ensures the topological correctness of generated data. The four error types exhaustively cover the connectivity and branching axes.

2. **Topology-Aware Composite Reward**:

    - **Function**: A multi-objective reward designed for GRPO: $R_{\text{total}} = 0.10 R_{\text{fmt}} + 0.85 R_{\text{acc}} + 0.05 R_{\text{topo}}$.
    - **Accuracy Reward $R_{\text{acc}}$**: Comprises a soft F1 detection reward (continuous IoU-based mapping $\phi$), a localization reward, and a type reward based on type-aware Hungarian matching.
    - **Topology Reward $R_{\text{topo}}$**: Computes $(1-\text{clDice})$ over matched pairs to quantify skeletal deviation, multiplied by an area penalty to discourage excessively large bounding boxes.
    - **Design Motivation**: IoU cannot capture topological significance. clDice measures connectivity differences via skeletal overlap, directly encoding the prior that topological errors are defined by connectivity changes.

3. **Type-Aware Hungarian Matching**:

    - **Function**: Performs optimal prediction–annotation matching independently within each error type.
    - **Mechanism**: For each error type $t$, an IoU affinity matrix is constructed and the linear assignment problem is solved for optimal one-to-one matching; TP/FP/FN statistics are aggregated across types.
    - **Design Motivation**: Guarantees globally optimal, order-invariant, one-to-one matching while naturally encoding type correctness.

### Loss & Training
Stage 1: Full-parameter SFT on approximately 12,900 samples via next-token prediction. Stage 2: GRPO on approximately 50,300 samples — for each query, $G$ candidate outputs are sampled; advantages are computed using within-group reward normalization, and the policy is optimized with PPO clipping and KL regularization.

## Key Experimental Results

### Main Results (Detection F1@IoU)

| Model | Method | F1@0.3 | F1@0.5 | F1@0.75 | aF1 |
|------|------|--------|--------|---------|-----|
| GPT-4o | Zero-shot | 0.5 | 0.3 | 0.0 | 0.1 |
| GPT-5.2 | Zero-shot | 0.4 | 0.2 | 0.0 | 0.1 |
| Qwen2.5-VL-3B | Zero-shot | 0.0 | 0.0 | 0.0 | 0.0 |
| Qwen2.5-VL-3B | SFT | ~15 | ~10 | ~3 | ~5 |
| Qwen2.5-VL-3B | **Topo-R1** | **32.5** | **22.8** | **8.1** | **12.4** |
| Qwen3-VL-8B | **Topo-R1** | **38.7** | **28.3** | **11.2** | **16.0** |

### Ablation Study

| Configuration | F1@0.5 | aF1 | Notes |
|------|--------|-----|------|
| SFT only | 10.2 | 5.1 | Supervised fine-tuning only |
| SFT + GRPO (w/o topo reward) | 18.5 | 9.3 | Without topology reward |
| SFT + GRPO (w/ topo reward) | **22.8** | **12.4** | Full Topo-R1 |
| w/o format reward | 20.1 | 10.8 | Increased formatting errors |

### Key Findings
- The strongest closed-source VLMs (GPT-5.2, Gemini-2.5-Flash) perform near-randomly on topological anomaly detection, confirming that existing VLMs lack topology-aware perception.
- SFT bootstraps from random initialization but yields limited gains; the exploratory capacity of GRPO is critical for discovering sparse anomalies.
- Despite a weight of only 0.05, the clDice topology reward contributes substantially, suggesting that reward design matters more than reward magnitude.
- Cross-domain training (road networks + cracks + vessels) yields better generalization than single-domain training.

## Highlights & Insights
- **Pioneering Contribution**: This is the first work to apply GRPO reinforcement learning to topological quality assessment, opening a new research direction in VLM topology-aware perception.
- **Elegant Reward Design**: clDice is repurposed from a loss function into an RL reward signal and conditioned on type-aware Hungarian matching, ensuring that only type-correct detections receive topology rewards — preventing misleadingly positive feedback for correct-location but wrong-type predictions.
- **Practical Value**: Topological quality assessment without target-domain annotations enables the framework to serve as a post-processing quality assurance tool for existing segmentation pipelines.

## Limitations & Future Work
- The current framework handles only 2D tubular structures; extension to 3D networks (e.g., cerebrovascular trees, neuronal connectomes) remains an open problem.
- Synthetic anomalies may not faithfully reflect the distribution of real post-processing errors (e.g., gradual boundaries from over- or under-segmentation).
- The fixed four-class error taxonomy may not cover all practical scenarios (e.g., false positives caused by partial occlusion).
- The 256×256 patch size limits the model's ability to perceive topological relationships at larger scales.

## Related Work & Insights
- **vs. AnomalyR1**: AnomalyR1 targets industrial anomaly detection; Topo-R1 focuses on topological anomalies, with fundamentally different reward design (clDice vs. IoU).
- **vs. clDice Loss**: clDice was originally used as a training loss to optimize segmentation; Topo-R1 repurposes it as an RL reward signal for detection and classification.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — First VLM topology-aware framework; both the problem formulation and methodology are pioneering.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-backbone and multi-domain evaluation, though real-world application validation is lacking.
- **Writing Quality**: ⭐⭐⭐⭐ — The method section is highly detailed with clear mathematical derivations.
- **Value**: ⭐⭐⭐⭐⭐ — Addresses a practical need for annotation-free topological quality assessment with broad application prospects.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Detecting Misbehaviors of Large Vision-Language Models by Evidential Uncertainty Quantification](../../ICLR2026/multimodal_vlm/detecting_misbehaviors_of_large_vision-language_models_by_evidential_uncertainty.md)
- [\[ICML 2026\] Debate with Images: Detecting Deceptive Behaviors in Multimodal Large Language Models](../../ICML2026/multimodal_vlm/debate_with_images_detecting_deceptive_behaviors_in_multimodal_large_language_mo.md)
- [\[CVPR 2026\] Understanding Task Transfer in Vision-Language Models](understanding_task_transfer_in_vision-language_models.md)
- [\[CVPR 2026\] Towards Calibrating Prompt Tuning of Vision-Language Models](towards_calibrating_prompt_tuning_of_vision-language_models.md)
- [\[CVPR 2026\] EvoPrompt: Evolving Prompt Adaptation for Vision-Language Models](evolving_prompt_adaptation_for_vision-language_models.md)

</div>

<!-- RELATED:END -->
