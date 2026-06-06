---
title: >-
  [Paper Note] Guideline-Consistent Segmentation via Multi-Agent Refinement
description: >-
  [AAAI 2026][Segmentation][guideline-consistent segmentation] A training-free multi-agent framework is proposed that achieves guideline-consistent semantic segmentation through an iterative Worker (segmentation execution)…
tags:
  - "AAAI 2026"
  - "Segmentation"
  - "guideline-consistent segmentation"
  - "multi-agent"
  - "VLM"
  - "reinforcement learning"
  - "training-free framework"
date: 2026-05-08
content_hash: 59afddbe911de030
---

# Guideline-Consistent Segmentation via Multi-Agent Refinement

**Conference**: AAAI 2026
**arXiv**: [2509.04687](https://arxiv.org/abs/2509.04687)  
**Code**: [Project Page](https://guideline-seg.github.io/)  
**Area**: Segmentation
**Keywords**: guideline-consistent segmentation, multi-agent, VLM, reinforcement learning, training-free framework

## TL;DR

A training-free multi-agent framework is proposed that achieves guideline-consistent semantic segmentation through an iterative Worker (segmentation execution) and Supervisor (guideline verification) loop coupled with an RL-based adaptive stopping strategy, surpassing prior SOTA by 8.61 and 5.5 gIoU on Waymo and ReasonSeg, respectively.

## Background & Motivation

Semantic segmentation requires not only precise pixel-level masks but also strict adherence to text-defined annotation guidelines. However:

**Guidelines are complex and lengthy**: Waymo's pedestrian guidelines contain more than ten rules—"a person riding a scooter counts as a pedestrian," "exclude mannequins/statues/reflections," "a pedestrian holding an object smaller than 2 m counts as a single label," etc.

**Existing methods fail on long-form text**: Open-vocabulary segmentation methods such as LISA and GroundedSAM perform well under phrase-level prompts but degrade sharply when faced with paragraph-level guidelines (LISA-7B drops from 43.42 to 23.41 gIoU).

**Human annotations are also inconsistent**: Waymo's official ground truth itself contains violations of its own guidelines (inconsistent annotation across different timestamps of the same scene).

**Retraining is impractical**: Annotation rules may be updated, making task-specific retraining infeasible.

**Core Goal**: Design a **training-free framework** capable of handling complex paragraph-level guidelines and ensuring guideline consistency through iterative verification.

## Method

### Overall Architecture

Pipeline: Input image + text guidelines → Context construction (filtering relevant rules) → Worker–Supervisor iterative loop (segment → verify → correct) → Adaptive stopping controller decides termination → Output guideline-consistent segmentation.

Core components:
- **Context Construction**: Enricher (scene description) + Retriever (guideline retrieval) + Smart Crop
- **Worker**: VLM-driven detection + SAM segmentation
- **Supervisor**: Dual-agent evaluation (Agent 1–evaluation / Agent 2–box generation) + SigLIP verification
- **AiRC**: RL-driven adaptive iterative refinement controller

### Key Designs

#### Context Construction

Different images require only a relevant subset of rules; feeding all rules causes information overload:

- **Enricher**: Uses the lightweight multimodal model Gemma3-4B to generate image captions, constructing a query $Q = \{P, \text{<caption>}, H \times W\}$ that combines the prompt and image resolution.
- **Retriever**: Encodes the query as a vector using SentenceTransformer and retrieves the top-$k=8$ most relevant guidelines from a FAISS index.
- **Smart Crop**: Splits the image into two regions based on object distribution, avoiding object bisection while balancing the number of objects on each side. A 0.8× downsampled image is first passed through OWLv2 to obtain coarse bounding boxes, after which the optimal split line is determined from the spatial distribution.

#### Worker–Supervisor Iterative Loop

**Worker**:
- *Initialization*: A VLM (Gemini-2.5-flash) detects bounding boxes for target categories, assigns unique IDs, and passes them to frozen SAM2.1 to generate masks.
- *Iteration*: Incorporates Supervisor feedback to correct outputs—adding missed objects, removing false detections, and adjusting imprecise boxes.

**Supervisor** (dual-agent design):
- **Agent 1 (Supervisor\_eval)**: Given the image, Worker output, and relevant guidelines, it identifies three types of issues: (i) missed objects, (ii) false positives violating exclusion rules, and (iii) masks requiring fine-grained adjustment. Outputs structured JSON.
- **Agent 2 (Supervisor\_boxgen)**: Generates candidate bounding boxes based on Agent 1's critique.
- **SigLIP Validator**: Crops candidate regions (with contextual buffer) and verifies them via SigLIP image–text matching; a candidate is accepted if the sigmoid probability $\geq 0.5$.

#### Adaptive Iterative Refinement Controller (AiRC)

Iterative control is formulated as a finite-horizon MDP solved with tabular Q-learning:

- **State space**: 6 states $s = 2d + v$, where $d \in \{0,1,2\}$ (scene density bucket) and $v \in \{0,1\}$ (whether unresolved violations exist).
- **Action space**: $\{\text{STOP},\ \text{CONTINUE}\}$
- **Issue count**: $I_t = I_{\text{miss}} + I_{\text{false}} + 0.1 \cdot I_{\text{ref}}$ (genuine errors count as 1; minor refinement suggestions count as 0.1).
- **Reward design**:

$$r(s,a,s') = \begin{cases}(I_t - I_{t+1}) - c + b[I_{t+1}=0], & a_t = \text{CONTINUE} \\ 0, & a_t = \text{STOP},\ I_t = 0 \\ -p, & a_t = \text{STOP},\ I_t > 0\end{cases}$$

  - Step cost $c = 0.02$, early-stop penalty $p = 2.0$, clean-scene bonus $b = 1.0$.
- Q-table is persisted across runs; $\epsilon$-greedy exploration with $\epsilon = 0.02$.
- Constraints: MIN\_ITERS = 2, MAX\_ITERS = 4.

### Loss & Training

**Training-free framework**: Both the VLM (Gemini-2.5-flash) and SAM2.1 are frozen; all adaptation is achieved through strategic prompting and context control.

- Worker temperature $T = 0.5$ (detection flexibility); Supervisor\_eval temperature $T = 0.3$ (deterministic reasoning).
- Average of 2.6 iterations per sample; per-sample cost approximately \$0.0088 (Gemini-2.5-flash API).
- To account for VLM non-determinism, each experiment is run 3 times with different random seeds and mean ± standard deviation is reported.

## Key Experimental Results

### Main Results

**Waymo Guideline-Consistent Dataset (101 manually verified samples, full-length guidelines)**:

| Method | gIoU | cIoU | mPr | mRec | mDice |
|--------|------|------|-----|------|-------|
| LISA-7B | 23.41 | 18.32 | 24.38 | 79.79 | 33.17 |
| GroundedSAM | 20.43 | 19.36 | 29.35 | 28.11 | 25.63 |
| Gemini-2.5 | 69.02 | 74.24 | 80.91 | 79.89 | 78.75 |
| SegZero | 71.96 | 73.72 | 86.34 | 78.45 | 80.88 |
| **Ours** | **80.57** | **86.70** | **91.06** | **84.78** | **87.20** |

**ReasonSeg val dataset**:

| Method | gIoU | cIoU |
|--------|------|------|
| Gemini-2.5 | 55.5 | 44.9 |
| SegZero | 62.6 | 62.0 |
| **Ours** | **68.1** | **66.4** |

Key observation: As guideline length increases (word → phrase → full text), competing methods suffer substantial performance degradation, whereas Gemini-2.5 and the proposed method actually benefit from complete guidelines.

### Ablation Study

**Component ablation (Waymo)**:

| Configuration | gIoU | cIoU | mPr | mRec |
|---------------|------|------|-----|------|
| Worker only | 69.02 | 74.24 | 80.91 | 79.89 |
| + Context Construction | 73.87 | 78.40 | 88.36 | 80.81 |
| + Context + Supervisor (Full) | **80.57** | **86.70** | **91.06** | **84.78** |

AiRC effect: Dynamic stopping resolves 110% more violations than fixed 2-iteration runs (0.61 vs. 0.29 violations/crop), requiring only 48% of crops to run additional iterations.

### Key Findings

- Methods such as LISA collapse to gIoU in the low 20s under full-length guidelines, as they cannot process complex rules embedded in long contexts.
- High recall (84.78) indicates successful recovery of missed objects (e.g., umbrellas, backpacks); high precision (91.06) indicates effective removal of false positives (e.g., cyclists, reflections).
- The adaptive stopping controller benefits dense scenes most, as violations are most likely to occur in crowded scenarios.

## Highlights & Insights

1. **Novel problem formulation**: The paper is the first to explicitly define "guideline-consistent segmentation," elevating annotation rule adherence to a first-order task objective, and reveals inconsistencies in existing ground-truth annotations.
2. **RL-based stopping strategy**: A compact 6-state Q-table realizes adaptive iterative control that is both more efficient and more effective than fixed-iteration alternatives.
3. **Dual-agent division of labor**: Separating evaluation and box generation into distinct agents proves more effective than assigning all responsibilities to a single agent.
4. **Scalability**: Both the VLM and SAM are frozen; guideline updates require no retraining—only modification of the text input.

## Limitations & Future Work

- Reliance on a closed-source VLM (Gemini-2.5 API) limits local deployment and reproducibility.
- Quantitative constraint interpretation is difficult: the VLM struggles to precisely judge "whether an object exceeds 2 m," which may cause misclassification.
- Bounding boxes serve as the primary SAM input, limiting fine-grained segmentation (e.g., fine structures such as eyelashes).
- Accuracy is prioritized over latency; multiple API calls make the framework unsuitable for real-time applications.
- The method supports only semantic segmentation and has not been extended to instance segmentation.

## Related Work & Insights

- **Multi-agent collaboration paradigm**: Inherits ideas from multi-agent frameworks such as MetaGPT and AutoGen—multiple specialized agents collaborating outperforms a single generalist agent.
- **VLM segmentation progress**: From single-pass inference in LISA to iterative refinement in this work, representing a paradigm shift for VLMs in segmentation from "answering" to "verifying."
- **Self-correction mechanism**: The Worker–Supervisor loop can be viewed as self-refinement for visual tasks, augmented with an explicit rule-verification step.
- Inspiration: The guideline-consistency paradigm is extensible to domains requiring strict rule adherence, such as medical image annotation and autonomous driving perception annotation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Novel problem formulation; unique training-free solution combining multi-agent collaboration and RL.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Validated on Waymo and ReasonSeg with a manually curated high-quality test set and sufficient ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with rich visualizations, though some content is redundant.
- Value: ⭐⭐⭐⭐ — Guideline consistency addresses a genuine pain point in practical annotation pipelines, offering high applied value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Towards Context-Aware Image Anonymization with Multi-Agent Reasoning](../../CVPR2026/segmentation/towards_context-aware_image_anonymization_with_multi-agent_reasoning.md)
- [\[CVPR 2026\] EReCu: Pseudo-label Evolution Fusion and Refinement with Multi-Cue Learning for Unsupervised Camouflage Detection](../../CVPR2026/segmentation/erecu_pseudolabel_evolution_unsupervised_camouflage.md)
- [\[CVPR 2026\] Learning Cross-View Object Correspondence via Cycle-Consistent Mask Prediction](../../CVPR2026/segmentation/learning_cross-view_object_correspondence_via_cycle-consistent_mask_prediction.md)
- [\[ICML 2026\] MVR-cache: Optimizing Semantic Caching via Multi-Vector Retrieval and Learned Prompt Segmentation](../../ICML2026/segmentation/mvr-cache_optimizing_semantic_caching_via_multi-vector_retrieval_and_learned_pro.md)
- [\[ICLR 2026\] RegionReasoner: Region-Grounded Multi-Round Visual Reasoning](../../ICLR2026/segmentation/regionreasoner_region-grounded_multi-round_visual_reasoning.md)

</div>

<!-- RELATED:END -->
