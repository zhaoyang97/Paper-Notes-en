---
title: >-
  [Paper Note] RobustMerge: Parameter-Efficient Model Merging for MLLMs with Direction Robustness
description: >-
  [NeurIPS 2025][Multimodal VLM][Model Merging] From the perspective of low-rank decomposition, this paper identifies "direction robustness" as the key factor in parameter-efficient module merging (as opposed to sign conflicts in full-parameter merging), and proposes RobustMerge, which maintains singular value direction stability via complementary parameter adaptive scaling and cross-task normalization, achieving average improvements of 3.4% (seen tasks) and 4.5% (unseen tasks) on multimodal generation benchmarks.
tags:
  - NeurIPS 2025
  - Multimodal VLM
  - Model Merging
  - LoRA
  - Parameter-Efficient Fine-Tuning
  - Direction Robustness
  - Multi-Task Learning
date: 2026-05-08
content_hash: 20d95eac343fb251
---

# RobustMerge: Parameter-Efficient Model Merging for MLLMs with Direction Robustness

**Conference**: NeurIPS 2025  
**arXiv**: [2502.17159](https://arxiv.org/abs/2502.17159)  
**Code**: [GitHub](https://github.com/AuroraZengfh/RobustMerge)  
**Area**: Multimodal VLM  
**Keywords**: Model Merging, LoRA, Parameter-Efficient Fine-Tuning, Direction Robustness, Multi-Task Learning

## TL;DR
From the perspective of low-rank decomposition, this paper identifies "direction robustness" as the key factor in parameter-efficient module merging (as opposed to sign conflicts in full-parameter merging), and proposes RobustMerge, which maintains singular value direction stability via complementary parameter adaptive scaling and cross-task normalization, achieving average improvements of 3.4% (seen tasks) and 4.5% (unseen tasks) on multimodal generation benchmarks.

## Background & Motivation
In large model deployment, practitioners frequently need to merge multiple task-specific expert models into a single generalist model to acquire multi-task capabilities without data leakage. As model scale grows (e.g., MLLMs), parameter-efficient fine-tuning (PEFT, particularly LoRA) has become the standard approach for training expert models.

However, existing model merging methods (e.g., Task Arithmetic, TIES-merging, DARE) are primarily designed for **full fine-tuning (FFT)**, with their core strategies targeting **sign conflicts** among parameters. When these methods are directly applied to LoRA module merging, performance degrades severely—sometimes falling below the zero-shot baseline.

**Why do FFT merging methods fail for PEFT?** Through in-depth analysis, the authors identify the following critical distinctions:

**Distributional Differences**: FFT parameters have narrow, concentrated distributions (near the mean), where sign conflicts are genuinely the dominant issue; LoRA parameters exhibit substantially wider distributions, with the B matrix approximating a Gaussian and the A matrix approximating a uniform distribution.

**Singular Value Disparity**: LoRA modules exhibit a significant head-tail gap in singular values; directions corresponding to large singular values are inherently robust during merging, whereas directions corresponding to small singular values are highly susceptible to perturbation.

**Key Challenge**: For PEFT, the problem is not sign conflicts but **directional instability**—task-specific knowledge directions corresponding to small singular values are altered during merging, causing performance collapse.

Additionally, existing high-performing methods suffer from scalability issues: AdaMerging requires validation data, EMR-Merging requires additional storage, and LoraHub requires test-time optimization—none of which generalize to unseen tasks. RobustMerge targets a **training-free, data-free, storage-free** parameter-efficient merging algorithm that generalizes to unseen tasks.

## Method

### Overall Architecture
RobustMerge operates in two steps: (1) pruning and complementary parameter scaling, and (2) cross-task normalization. The processed LoRA modules are merged via a specific formula to obtain the final multi-task model.

### Key Designs

1. **Magnitude-Based Parameter Pruning**:

    - Parameters with small magnitudes (rather than random or sign-based selection) are set to zero: $\widetilde{A} = \mathcal{M}_A(k) \odot A$, $\widetilde{B} = \mathcal{M}_B(k) \odot B$
    - Here $k$ is the pruning rate, and $\mathcal{M}(\cdot)$ retains the top-k non-zero parameters by magnitude.
    - **Design Motivation**: Large-magnitude parameters better preserve directional stability in the low-rank space, consistent with the wide distribution property of LoRA; ablations confirm that sign-based pruning (as in TIES) performs worst in PEFT merging.

2. **Complementary Parameter Adaptive Scaling (Core Innovation)**:

    - Leverages the **asymmetric relationship** between the A and B matrices of LoRA (B is more critical; A approximates an orthogonal matrix) to directly adjust singular values on the original low-rank matrices, avoiding explicit SVD decomposition.
    - The scaling matrix $S$ is diagonal: $S^i = \frac{\sum_j |A_{[i,j]}|}{\sum_j |\mathcal{M}_A{[i,j]} \odot A_{[i,j]}|}$
    - **Key Effect**: This coefficient applies larger scaling to small singular values (since rows corresponding to small singular values suffer proportionally greater loss from pruning) while leaving large singular values largely unaffected, thereby **adaptively reducing the head-tail singular value gap**.
    - **Design Motivation**: Reducing the singular value gap mitigates directional instability during merging—large-value directions are naturally robust, and the task-specific knowledge encoded in small-value directions is what genuinely requires protection.

3. **Cross-Task Normalization**:

    - Scaling coefficients are normalized across all tasks: $\widetilde{S}_n^i = S_n^i / \sum_{n=1}^N S_n^i$
    - Final merging: $\Delta\widetilde{W} = \lambda \sum_{n=1}^N (\widetilde{B}_n \cdot \widetilde{S}_n) \cdot \sum_{n=1}^N \widetilde{A}_n$
    - **Design Motivation**: Imbalanced data volumes across tasks can cause overfitting to certain tasks; normalization balances coefficients across tasks, stabilizing performance on seen tasks and substantially enhancing generalization to unseen tasks.

### Loss & Training
RobustMerge is a **training-free** method; all operations are post-processing:
- **Input**: $N$ LoRA modules fine-tuned on different tasks $\{A_n, B_n\}_{n=1}^N$
- **Hyperparameters**: pruning rate $k$, scaling coefficient $\lambda$ (default: 2)
- Requires no validation data, additional storage, or training
- All merging experiments can be completed on a single NVIDIA A6000 GPU

## Key Experimental Results

### Main Results (MM-MergeBench, 8 Seen + 4 Unseen Tasks, LLaVA Base Model)

| Method | Seen Task Mean | Unseen Task Mean | Notes |
|--------|---------------|-----------------|-------|
| Zero-Shot | 43.37 | 25.22 | No merging baseline |
| Multi-Task | 63.62 | 36.06 | Joint multi-task training upper bound |
| Task Arithmetic | 53.93 | 33.31 | Simple additive merging |
| DARE | 53.84 | 33.15 | Random dropout + rescaling |
| TIES-merging | 53.09 | 33.14 | Sign voting strategy |
| PCB-merging | 53.70 | 33.53 | Competitive balance |
| **RobustMerge** | **57.33 (+3.4%)** | **37.99 (+4.5%)** | Surpasses multi-task learning |

### Ablation Study

| Component | Seen Task Mean | Gain | Notes |
|-----------|---------------|------|-------|
| Baseline (no adaptation) | 53.93 | — | Task Arithmetic |
| + Pruning & Scaling | 56.14 | +2.21 | Core contribution of direction robustness |
| + Pruning & Scaling + Normalization | **57.33** | **+3.40** | Cross-task balancing yields further improvement |

### Key Findings
- Existing FFT merging methods underperform even simple Task Arithmetic in PEFT settings, confirming that sign conflict strategies are ill-suited for LoRA.
- RobustMerge's gain on unseen tasks (4.5%) exceeds that on seen tasks (3.4%), with cross-task normalization identified as the key contributor.
- On general benchmarks (POPE/MME/MMBench), RobustMerge outperforms all baselines while preserving base model capabilities.
- On vision task experiments (CLIP-ViT-B-32 merged across 8 datasets): RobustMerge improves over zero-shot by 7.9% and over the best prior method by 4.4%.
- Direction robustness analysis: RobustMerge substantially improves directional similarity for eigenvectors corresponding to small singular values while enhancing their numerical ratio.

## Highlights & Insights
- **Theoretically Insightful**: Framing PEFT merging through the lens of SVD directional robustness, this work is the first to identify "directional instability"—rather than sign conflicts—as the core challenge in parameter-efficient merging.
- **Elegant and Efficient**: By exploiting the asymmetry between the A and B matrices, the method operates directly on the original matrices without explicit SVD decomposition, incurring minimal computational overhead.
- **Strong Generalization to Unseen Tasks**: This is the first merging method to consistently surpass multi-task learning on unseen tasks without requiring additional data or storage.

## Limitations & Future Work
- Validation has not been conducted on other PEFT architectures (e.g., Adapter, Prompt Tuning).
- Theoretical analysis is limited to a simplified two-model merging scenario; formal characterization of complex multi-model interactions remains an open problem.
- No dedicated algorithm operating directly on the decomposed matrices has been designed (due to efficiency considerations), which may represent a direction for further improvement.

## Related Work & Insights
- **vs. TIES-merging/DARE**: These methods are built on the sign conflict assumption, but PEFT parameters have wider distributions where magnitude matters more than sign for directional fidelity.
- **vs. AdaMerging/EMR-Merging**: These methods require validation data or additional storage; RobustMerge is fully training-free and generalizes more broadly.
- **vs. LoraHub**: LoraHub requires test-time adaptive coefficient optimization, whereas RobustMerge merges deterministically without any adaptation.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The directional robustness perspective is novel and fundamentally explains why FFT methods fail for PEFT; the complementary parameter scaling design is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Dual-track validation on multimodal and vision tasks, a 12-task benchmark, and comprehensive ablations (pruning rate, rank, components, scaling strategies).
- **Writing Quality**: ⭐⭐⭐⭐ Analysis logic is clear and visualizations are rich, though the dense notation requires careful reading.
- **Value**: ⭐⭐⭐⭐⭐ Provides both theoretical grounding and a practical solution for model merging in the PEFT era; the vast number of LoRA models on HuggingFace can directly benefit.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] FREE-Merging: Fourier Transform for Efficient Model Merging](../../ICCV2025/multimodal_vlm/free-merging_fourier_transform_for_efficient_model_merging.md)
- [\[CVPR 2026\] DC-Merge: Improving Model Merging with Directional Consistency](../../CVPR2026/multimodal_vlm/dc-merge_improving_model_merging_with_directional_consistency.md)
- [\[NeurIPS 2025\] MMPerspective: Do MLLMs Understand Perspective? A Comprehensive Benchmark for Perspective Perception, Reasoning, and Robustness](mmperspective_do_mllms_understand_perspective_a_comprehensive_benchmark_for_pers.md)
- [\[NeurIPS 2025\] HAWAII: Hierarchical Visual Knowledge Transfer for Efficient VLM](hawaii_hierarchical_visual_knowledge_transfer_for_efficient_vision-language_mode.md)
- [\[NeurIPS 2025\] AdaLRS: Loss-Guided Adaptive Learning Rate Search for Efficient Foundation Model Pretraining](adalrs_lossguided_adaptive_learning_rate_search_for_efficien.md)

</div>

<!-- RELATED:END -->
