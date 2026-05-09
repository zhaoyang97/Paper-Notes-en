---
title: >-
  [Paper Note] WISER: Wider Search, Deeper Thinking, and Adaptive Fusion for Training-Free Zero-Shot Composed Image Retrieval
description: >-
  [CVPR 2026][Image Generation][Composed Image Retrieval] This paper proposes WISER, a training-free zero-shot composed image retrieval (ZS-CIR) framework that unifies T2I and I2I dual-path retrieval through an iterative "retrieve–verify–refine" loop. A VLM verifier explicitly models intent-awareness and uncertainty-awareness to enable adaptive fusion and structured self-reflective refinement. WISER achieves a relative improvement of 45% on CIRCO mAP@5 and 57% on CIRR Recall@1, surpassing many supervised methods.
tags:
  - CVPR 2026
  - Image Generation
  - Composed Image Retrieval
  - Zero-Shot
  - T2I+I2I Fusion
  - Self-Reflective Refinement
  - VLM Verification
  - CLIP
date: 2026-05-08
content_hash: d6a37a6eaca5f481
---

# WISER: Wider Search, Deeper Thinking, and Adaptive Fusion for Training-Free Zero-Shot Composed Image Retrieval

**Conference**: CVPR 2026  
**arXiv**: [2602.23029](https://arxiv.org/abs/2602.23029)  
**Code**: [https://github.com/Physicsmile/WISER](https://github.com/Physicsmile/WISER)  
**Area**: Image Generation  
**Keywords**: Composed Image Retrieval, Zero-Shot, T2I+I2I Fusion, Self-Reflective Refinement, VLM Verification, CLIP

## TL;DR

This paper proposes WISER, a training-free zero-shot composed image retrieval (ZS-CIR) framework that unifies T2I and I2I dual-path retrieval through an iterative "retrieve–verify–refine" loop. A VLM verifier explicitly models intent-awareness and uncertainty-awareness to enable adaptive fusion and structured self-reflective refinement. WISER achieves a relative improvement of 45% on CIRCO mAP@5 and 57% on CIRR Recall@1, surpassing many supervised methods.

## Background & Motivation

**Task Definition**: Composed Image Retrieval (CIR) retrieves a target image given a reference image $I_{\text{ref}}$ and a modification text $T_{\text{mod}}$. ZS-CIR requires no annotated triplets and relies on the generalization capability of pre-trained models.

**Two Paradigms**:
   - **T2I Paradigm** (Text-to-Image retrieval): Converts the reference image into a text caption, combines it with the modification text to generate an edited description, and retrieves using text embeddings. Excels at complex semantic modifications but loses fine-grained visual details.
   - **I2I Paradigm** (Image-to-Image retrieval): Directly edits the reference image using an image editing model and retrieves using image embeddings. Better preserves visual details but struggles with complex compositional edits or ambiguous intent.

**Key Challenge**: Real-world query intents are diverse; a single paradigm is insufficient. Existing fusion methods (e.g., CIG, IP-CIR) suffer from two critical deficiencies:
   - **Lack of intent-awareness**: Static fixed-weight fusion cannot adapt to varying query intents.
   - **Lack of uncertainty-awareness**: The reliability differences among candidate results from each branch are ignored.

**Goal**: Design an iterative "retrieve→verify→refine" closed loop that uses a VLM verifier to assess candidate quality, triggering structured self-reflective refinement for uncertain results and adaptive multi-level fusion for reliable results. The entire pipeline is training-free and modular.

## Method

### Overall Architecture

WISER adopts a three-stage iterative pipeline of "retrieve–verify–refine":

1. **Wider Search**: T2I and I2I retrieval paths are activated in parallel to expand the candidate pool.
2. **Adaptive Fusion**: A VLM verifier evaluates candidates; reliable results undergo multi-level fusion ranking, while unreliable ones trigger refinement.
3. **Deeper Thinking**: Uncertain retrievals undergo a three-step structured self-reflection to generate improvement feedback for the editor.

Formally, given reference image $I_{\text{ref}}$ and modification text $T_{\text{mod}}$, editor $\mathcal{F}$ generates edited caption $C_{\text{edit}}$ and edited image $I_{\text{edit}}$, encoded by CLIP's visual encoder $E_{\text{img}}$ and text encoder $E_{\text{txt}}$ into query vectors $q_v$ and $q_t$, and cosine similarity is computed over database $\mathcal{D}$ to perform retrieval.

### Key Design 1: Wider Search — Dual-Path Parallel Retrieval

**Core Idea**: Simultaneously generate an edited caption and an edited image for T2I and I2I retrieval respectively, and take the union to expand the candidate pool.

**T2I Path**: A pre-trained captioner (BLIP-2) converts the reference image into caption $C_{\text{ref}}$, which the editor combines with the modification text to produce an edited description:

$$C_{\text{edit}} = \mathcal{F}_{\text{txt}}(C_{\text{ref}}, T_{\text{mod}})$$

**I2I Path**: The editor directly edits the reference image:

$$I_{\text{edit}} = \mathcal{F}_{\text{img}}(I_{\text{ref}}, T_{\text{mod}})$$

**Dual-Path Retrieval**: Top-$K$ candidates are retrieved from each path:

$$\mathcal{R}_p = \{I_p^1, I_p^2, \ldots, I_p^K\}, \quad p \in \{\text{T2I}, \text{I2I}\}$$

Their union forms the expanded candidate pool:

$$\mathcal{R}_{\text{union}} = \mathcal{R}_{\text{T2I}} \cup \mathcal{R}_{\text{I2I}}$$

**Design Motivation**: T2I retrieval excels at semantic modification but loses visual details, while I2I preserves visual content but handles semantic modifications poorly. Dual-path parallelism captures both advantages and improves recall.

### Key Design 2: Adaptive Fusion — Verification-Guided Adaptive Fusion

This is the core module of WISER, consisting of two steps: verification scoring and multi-level fusion.

**Step 1: Verification Scoring**. For each candidate $I_p^k \in \mathcal{R}_{\text{union}}$, a triplet $(I_{\text{ref}}, T_{\text{mod}}, I_p^k)$ is fed into VLM verifier $\Phi$ (Qwen2.5-VL-7B), which answers a binary question of whether the candidate correctly reflects the modification of the reference image. Logits are extracted to compute confidence:

$$c_p^k = \frac{\exp(\ell_{p,k}^{(\text{yes})})}{\exp(\ell_{p,k}^{(\text{yes})}) + \exp(\ell_{p,k}^{(\text{no})})}$$

**Step 2: Multi-Level Fusion Strategy**, modeling uncertainty and intent-awareness at two levels simultaneously.

**Branch-Level Uncertainty-Awareness**: For each path, the highest-confidence candidate is selected as a pseudo-target $I_p^* = \arg\max_k c_p^k$ with reliability score $r_p = \max_k c_p^k$. If $\min(r_{\text{T2I}}, r_{\text{I2I}}) < \tau$ (either path is uncertain), Deeper Thinking refinement is triggered.

**Candidate-Level Intent-Awareness**: For reliable retrievals, a fused confidence score is computed:

$$c_{\text{fused}}^k = c_{\text{T2I}}^k + c_{\text{I2I}}^k$$

Candidates are ranked lexicographically by:

$$\Psi(I^k) = \left(-c_{\text{fused}}^k, \ -\max(c_{\text{T2I}}^k, c_{\text{I2I}}^k), \ -c_{\text{T2I}}^k\right)$$

**Ranking Logic**: The primary key is the fused score (overall intent alignment). T2I tends to yield higher confidence for semantics-dominant edits, I2I for visuals-dominant ones; the fused score naturally ranks candidates that excel in both at the top. Ties are broken by the single-path maximum and then by the T2I score.

### Key Design 3: Deeper Thinking — Structured Self-Reflective Refinement

For uncertain retrievals ($r_p < \tau$), an LLM refiner (GPT-4o) performs a three-step analysis:

**Step 1: Identify Modifications**. Given $C_{\text{ref}}$ and $T_{\text{mod}}$, the expected modifications are analyzed and structured modification phrases are generated — attribute changes (e.g., "red→blue") and entity additions/removals (e.g., "add hat").

**Step 2: Analyze Retrieval Results**. The caption of pseudo-target $I_p^*$ is obtained and compared item by item against the modification phrases from Step 1 to identify missing or incorrectly applied modifications.

**Step 3: Generate Refinement Suggestions**. Targeted suggestions are generated for unsatisfied modifications — enhancing the edited description for the T2I path and providing visual guidance for the I2I path. Suggestions are appended to the modification text and re-fed into the editor, iterating up to $N$ times.

**Design Motivation**: Simulates human self-reflection — "Where did the result go wrong? How can it be improved?" — without requiring any training data.

### Implementation Details

| Component | Specific Model |
|-----------|---------------|
| Editor $\mathcal{F}$ | BAGEL (unified text editing + image editing) |
| Verifier $\Phi$ | Qwen2.5-VL-7B |
| Refiner | GPT-4o |
| Captioner | BLIP-2 |
| Retrieval Model | CLIP ViT-B/32 / ViT-L/14 / ViT-G/14 |

- Candidate pool size $K=50$, reliability threshold $\tau=0.7$, refinement iterations $N=1$ (default single round)
- Single NVIDIA H20 GPU; approximately 0.5 GPU hours per 1% performance gain
- Deeper Thinking is triggered only for low-confidence samples; the refinement rate is below 30%

## Key Experimental Results

### Main Results — CIRCO and CIRR Benchmarks (Table 1)

| Method | Backbone | Free | CIRCO mAP@5 | CIRCO mAP@25 | CIRR R@1 | CIRR R@5 | CIRR R_sub@1 |
|--------|----------|------|-------------|-------------|----------|----------|-------------|
| CIReVL | ViT-L/14 | ✓ | 18.57 | 20.89 | 24.55 | 52.31 | 59.54 |
| LDRE | ViT-L/14 | ✓ | 23.35 | 26.44 | 26.53 | 55.57 | 60.43 |
| IP-CIR | ViT-L/14 | ✓ | 26.43 | 29.87 | 29.76 | 58.82 | 62.48 |
| CoTMR | ViT-L/14 | ✓ | 27.61 | 30.61 | 35.02 | 64.75 | 69.39 |
| **WISER** | **ViT-L/14** | **✓** | **35.10** | **38.46** | **49.23** | **76.72** | **77.81** |
| CoTMR | ViT-G/14 | ✓ | 32.23 | 35.60 | 36.36 | 67.52 | 71.19 |
| IP-CIR | ViT-G/14 | ✓ | 32.75 | 36.86 | 39.25 | 70.07 | 69.95 |
| **WISER** | **ViT-G/14** | **✓** | **36.53** | **40.46** | **49.54** | **77.40** | **78.10** |

### Main Results — Fashion-IQ Benchmark (Table 2, ViT-L/14)

| Method | Free | Shirt R@10 | Dress R@10 | Toptee R@10 | Avg R@10 | Avg R@50 |
|--------|------|-----------|-----------|------------|---------|---------|
| CIReVL | ✓ | 29.49 | 24.79 | 31.36 | 28.55 | 48.57 |
| LDRE | ✓ | 31.04 | 22.93 | 31.57 | 28.51 | 50.54 |
| CoTMR | ✓ | 35.43 | 31.18 | 38.55 | 35.05 | 57.09 |
| **WISER** | **✓** | **43.13** | **38.42** | **45.39** | **42.17** | **58.51** |

### Ablation Study — Core Component Effectiveness (Table 3, ViT-B/32)

| T2I | I2I | Deeper Thinking | Fusion | FIQ Avg R@10 | CIRCO mAP@5 |
|-----|-----|----------------|--------|-------------|-------------|
| ✗ | ✓ | ✗ | — | 22.65 | 7.00 |
| ✗ | ✓ | ✓ | — | 23.58 | 7.57 |
| ✓ | ✗ | ✗ | — | 28.59 | 17.28 |
| ✓ | ✗ | ✓ | — | 29.22 | 17.64 |
| ✓ | ✓ | ✗ | AVG (fixed weight) | 33.40 | 13.53 |
| ✓ | ✓ | ✗ | ADA (adaptive) | 40.83 | 31.32 |
| ✓ | ✓ | ✓ | ADA (full WISER) | **41.99** | **32.23** |

### Module Compatibility Ablation (Table 4, CIRCO, ViT-B/32)

| Editor | Verifier | Refiner | mAP@5 | mAP@25 |
|--------|----------|---------|-------|--------|
| BAGEL | Qwen2.5-VL-7B | Qwen-Turbo | 32.80 | 35.21 |
| BAGEL | Qwen2.5-VL-7B | GPT-3.5-Turbo | 32.57 | 35.13 |
| BAGEL | Qwen2.5-VL-7B | GPT-4o | 32.23 | 34.82 |
| BAGEL | Qwen2-VL-7B | GPT-4o | 25.50 | 28.41 |
| BAGEL | Qwen2.5-VL-3B | GPT-4o | 27.50 | 30.16 |
| BAGEL | Qwen2.5-VL-32B | GPT-4o | 31.69 | 34.22 |
| GPT4o+OmniGen2 | Qwen2.5-VL-7B | GPT-4o | 31.18 | 33.82 |
| GPT4o+Step1X-Edit | Qwen2.5-VL-7B | GPT-4o | 31.91 | 34.92 |

### Key Findings

1. **WISER substantially outperforms all training-free methods**: CIRCO mAP@5 improves from CoTMR's 27.61 to 35.10 (ViT-L/14, +27%); CIRR R@1 improves from 35.02 to 49.23 (+41%).
2. **Surpasses many supervised methods**: Training-required methods such as LinCIR, IP-CIR (trained), and AutoCIR are all outperformed by training-free WISER.
3. **Naïve fusion is harmful**: Fixed-weight AVG fusion achieves only 13.53 on CIRCO mAP@5, which is **lower than the T2I single-path baseline of 17.28**, demonstrating that noise interference causes naïve fusion to degrade.
4. **Adaptive fusion is the core contribution**: ADA jumps from AVG's 13.53 to 31.32 (+131%), confirming that verification-guided fusion is far superior to fixed-weight approaches.
5. **Deeper Thinking provides consistent gains**: The full model outperforms the variant without refinement by approximately 1 mAP point, and single-path variants also benefit.
6. **Strong module replaceability**: Swapping the refiner yields minimal performance variation (32.21–32.80), validating the plug-and-play nature of the framework.
7. **Larger verifier is not always better**: Qwen2.5-VL-32B slightly underperforms the 7B model (31.69 vs. 32.23), possibly due to overthinking.
8. **One refinement round suffices**: $N=1$ captures most of the gain; further iterations yield diminishing returns.
9. **Consistent advantage across backbones**: WISER achieves large margins from ViT-B/32 to ViT-G/14, demonstrating strong generalizability.

## Highlights & Insights

- **"Retrieve–Verify–Refine" iterative paradigm**: Upgrades composed image retrieval from a one-shot query to an iterative closed-loop process, mimicking the human cognitive pattern of "attempt–evaluate–improve."
- **Novel use of VLM verifier**: Rather than using the VLM solely for retrieval or generation, the framework exploits its logits for binary judgment to estimate candidate reliability, converting discriminative capability into an uncertainty signal.
- **Counter-intuitive finding reveals the core problem**: The observation that fixed-weight average fusion performs worse than a single path profoundly reveals that the T2I and I2I paths produce conflicting noise, making adaptive fusion not a luxury but a necessity.
- **Elegant multi-level fusion design**: Branch-level assessment of path reliability combined with candidate-level fused-score ranking addresses the orthogonal problems of uncertainty and intent alignment at two distinct granularities.
- **Fully training-free and modular**: All components can be replaced with stronger off-the-shelf models, endowing the framework with continuous self-improvement capability.

## Limitations & Future Work

1. **High inference latency**: Each query requires invoking captioner + editor + dual retrieval + VLM verifier + optional refiner, resulting in latency far exceeding single-path methods.
2. **Dependence on closed-source APIs**: The default refiner uses GPT-4o, increasing cost and deployment complexity (though Qwen-Turbo is also viable).
3. **Coarse binary verification granularity**: Yes/no logits may lack sensitivity to subtle visual differences.
4. **Refinement relies on captions rather than images**: The refiner analyzes captions of pseudo-targets rather than directly examining images; information loss in captioning may introduce bias in the refinement direction.
5. **Efficiency at large scale unverified**: The largest database evaluated contains approximately 120K images (CIRCO); efficiency at the million-image scale remains unknown.
6. **Manual threshold $\tau$ setting**: Although performance is stable in the range 0.5–0.7, the optimal value varies across datasets.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The iterative "retrieve–verify–refine" paradigm is novel and the multi-level fusion strategy is elegant; however, individual components are all existing models composed together.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three benchmarks × three backbones; ablations cover components, module selection, thresholds, and iteration counts; comprehensive comparison against supervised methods.
- **Writing Quality**: ⭐⭐⭐⭐ Architecture diagrams are clear; the complementarity analysis of T2I/I2I is logically consistent; quantitative results are well supported by qualitative case studies.
- **Value**: ⭐⭐⭐⭐ Establishes a strong new training-free ZS-CIR baseline; the "verification-guided fusion" paradigm is generalizable to other multimodal retrieval tasks; inference latency limits practical deployment.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Instance-Level Composed Image Retrieval](../../NeurIPS2025/image_generation/instance-level_composed_image_retrieval.md)
- [\[CVPR 2026\] RAISE: Requirement-Adaptive Evolutionary Refinement for Training-Free Text-to-Image Alignment](raise_requirement-adaptive_evolutionary_refinement_for_training-free_text-to-ima.md)
- [\[CVPR 2026\] CaReFlow: Cyclic Adaptive Rectified Flow for Multimodal Fusion](careflow_cyclic_adaptive_rectified_flow_for_multimodal_fusion.md)
- [\[CVPR 2026\] TAP: A Token-Adaptive Predictor Framework for Training-Free Diffusion Acceleration](tap_a_token-adaptive_predictor_framework_for_training-free_diffusion_acceleratio.md)
- [\[CVPR 2026\] Taming Video Models for 3D and 4D Generation via Zero-Shot Camera Control](taming_video_models_for_3d_and_4d_generation_via_zero-shot_camera_control.md)

<!-- RELATED:END -->
