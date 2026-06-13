---
title: >-
  [Paper Note] EVL-ECG: Efficient ECG Interpretation With Multi-Aspect Heterogeneous Knowledge Distillation
description: >-
  [ICML 2026][Model Compression][ECG Foundation Model] EVL-ECG addresses VLM distillation for ECG interpretation where the teacher and student are heterogeneous in visual token count, tokenizers…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "ECG Foundation Model"
  - "Cross-architecture Knowledge Distillation"
  - "Multi-head Cross-Attention"
  - "Optimal Transport"
  - "Geometric Relation Matching"
date: 2026-05-08
content_hash: 2d88a6a1da83e7b4
---

# EVL-ECG: Efficient ECG Interpretation With Multi-Aspect Heterogeneous Knowledge Distillation

**Conference**: ICML 2026  
**arXiv**: [2605.29977](https://arxiv.org/abs/2605.29977)  
**Code**: To be confirmed  
**Area**: Medical Imaging / VLM Distillation / ECG Interpretation  
**Keywords**: ECG Foundation Model, Cross-architecture Knowledge Distillation, Multi-head Cross-Attention, Optimal Transport, Geometric Relation Matching

## TL;DR
EVL-ECG addresses VLM distillation for ECG interpretation where the teacher and student are heterogeneous in visual token count, tokenizers, and sequence lengths. It introduces a cross-architecture distillation framework with three modules: "Multi-head Cross-Attention Alignment + Optimal Transport Visual Feature Matching + Geometric Intra-Architecture Relation Matching." This pushes a 2B student model to SOTA, achieving a 2.4% higher AUC and 1.1% higher clinical accuracy than existing KD methods.

## Background & Motivation

**Background**: VLM-based ECG interpretation has evolved from simple classification to generative clinical reporting (Khunte 2024, Liu 2024b, etc.). However, frontier VLMs are large and slow to infer, making bedside or edge deployment impractical. Knowledge Distillation (KD) is a standard compression solution, with existing work on cross-tokenizer/cross-modal KD (Boizard 2025, Feng 2025).

**Limitations of Prior Work**: Distilling a giant VLM teacher into a small LM student faces two independent yet entangled obstacles:
- **Heterogeneous Tokenizers**: The teacher and student use different vocabularies, making direct alignment of output probabilities impossible.
- **Unbalanced Visual Token Counts**: The teacher's visual encoder outputs dense tokens (e.g., ViT-L outputs 256 tokens), while the student utilizes a lighter encoder with only 64 tokens, resulting in misaligned sequence lengths.
- Existing KD methods treat these issues in **isolation**, limiting the potential of using modern efficient SLMs as backbones.

**Key Challenge**: ECG interpretation relies heavily on fine-grained morphology (P-QRS-T intervals, ST segments, axis direction). Hard point-to-point alignment between dense and sparse tokens is neither possible (length mismatch) nor correct (semantic mismatch). Simultaneously, the global spatial topology of the ECG (12-lead layout) must be preserved—pointwise alignment might confuse precordial leads (V1-V6) with limb leads (I-III).

**Goal**: (1) Resolve dual heterogeneity in tokenizers and visual tokens; (2) Preserve global spatial structure (lead layout, waveform topology); (3) Achieve SOTA performance within a $\leq$ 2B parameter budget.

**Key Insight**: Distillation is divided into three layers: pointwise (feature-level, using Multi-Head Cross-Attention for adaptive aggregation), distributional (visual feature-level, using Optimal Transport for soft distribution alignment), and relational (structural-level, matching geometric distances and angular relationships). These layers are complementary, targeting different diagnostic dimensions of ECG.

**Core Idea**: Use MHCA to let the student adaptively aggregate teacher information by projecting its queries onto the teacher's dense representations; use entropic OT on visual tokens for soft alignment to maintain global topology; use distance/angle relation matching to preserve the teacher's intrinsic diagnostic "geometric manifold."

## Method

### Overall Architecture

The total loss is defined as:
$$\mathcal{L}_{\text{total}} = (1-\alpha)\mathcal{L}_{\text{CE}} + \alpha(\lambda_m \mathcal{L}_{\text{mhca}} + \lambda_r \mathcal{L}_{\text{rel}} + \lambda_{\text{ot}}\mathcal{L}_{\text{ot}})$$

The three KD losses serve distinct purposes:
- $\mathcal{L}_{\text{mhca}}$: Aligns student hidden states with attention-weighted teacher contexts.
- $\mathcal{L}_{\text{ot}}$: Uses Sinkhorn soft alignment on visual tokens to preserve global topology.
- $\mathcal{L}_{\text{rel}}$: Matches distance and angle relations to preserve intrinsic geometric manifolds.

### Key Designs

1. **Multi-Head Cross-Attention (MHCA) Alignment**:
    - **Function**: Adaptively aggregates the teacher's dense $L_t$-token representation into the student's compact $L_s$-token representation.
    - **Mechanism**: Student hidden states $H_s$ serve as queries, while teacher hidden states $H_t$ serve as keys/values. $\hat H_t = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$, where each head calculates $\text{Softmax}((H_s W_i^Q)(H_t W_i^K)^\top/\sqrt{d_k})(H_t W_i^V)$. The alignment loss is $\mathcal{L}_{\text{mhca}} = \frac{1}{B \cdot L_s}\sum \|H_s - \hat H_t\|_2^2$.
    - **Design Motivation**: Point-to-point alignment is infeasible when $L_s \neq L_t$. MHCA allows the student to dynamically "extract" the most useful teacher information (e.g., local ectopic beats, subtle ST shifts). Theoretically, this is equivalent to OT barycentric projection under entropy regularization.

2. **Optimal Transport Visual Feature Matching (OT-VFM)**:
    - **Function**: Performs soft distribution alignment at the visual token layer to preserve the 12-lead global spatial topology.
    - **Mechanism**: Teacher and student visual tokens are treated as uniform empirical distributions $\mu, \nu$. The entropic OT problem $\mathbf{P}^\star_\varepsilon = \arg\min_\mathbf{P} \langle \mathbf{P}, C\rangle - \varepsilon \mathcal{H}(\mathbf{P})$ is solved via Sinkhorn iterations. The distillation loss is $\mathcal{L}_{\text{ot}} = \sum_{i,j} P^\star_{\varepsilon, ij} \|t_i - s_j\|_2^2$.
    - **Design Motivation**: In ECG images, position implies diagnostic meaning (e.g., V1 near the RA, V5/V6 reflecting the left ventricle). Pointwise matching might incorrectly map precordial leads to limb leads. OT provides soft alignment that accommodates differing token lengths while preserving global structure (the transport plan implicitly defines "which student token covers which teacher region").

3. **Geometric Intra-Architecture Relation Matching**:
    - **Function**: Calculates distance/angle relation matrices within each architecture and aligns them to preserve the teacher's intrinsic diagnostic logic.
    - **Mechanism**: Defines two relational potentials for hidden state sequences $H$: distance potential $\psi_D$ (mean-normalized pairwise Euclidean distance) and angle potential $\psi_A$ (pairwise cosine similarity). $\mathcal{L}_k = \frac{1}{B \cdot L_s^2} \sum \|\psi_k(\hat H_t^{(i)}, \hat H_t^{(j)}) - \psi_k(H_s^{(i)}, H_s^{(j)})\|^2$ for $k \in \{D, A\}$. Total relation loss is $\mathcal{L}_{\text{rel}} = \tfrac{1}{2}(\mathcal{L}_D + \mathcal{L}_A)$.
    - **Design Motivation**: ECG diagnosis fundamentally relies on **structural topology and temporal relations** (P to QRS intervals, ST orientation, segment ratios). Point-level alignment reconstructs tokens but loses global geometry. Relation potentials preserve the teacher's "diagnostic clustering geometry" (e.g., distance/angle structures of similar arrhythmias in the learned latent space).

## Key Experimental Results

### Main Results: Cross-ECG Benchmarks

| Dataset | Metric | Random | GPT-4o | Claude 3.5 | LLaVA-Med | **EVL-ECG (2B)** |
|------|------|-------|--------|----------|----------|--------------|
| PTB-XL-Super | AUC | 50.3 | 55.6 | 54.0 | 67.3 | **75.4** |
| PTB-XL-Super | F1 | 33.2 | 28.3 | 27.5 | 45.6 | **51.2** |
| CODE-15% | AUC | 48.8 | 59.9 | 58.3 | 70.1 | **78.6** |
| ECG-QA | Accuracy | 16.2 | 35.2 | 34.2 | 47.5 | **51.8** |
| MMMU-ECG | Accuracy | 24.2 | 43.5 | 42.0 | 51.3 | **55.8** |

EVL-ECG with 2B parameters outperforms general frontier VLMs like GPT-4o and Claude 3.5, as well as open-source medical VLMs like LLaVA-Med across all benchmarks.

### Ablation Study (PTB-XL-Super AUC)

| Configuration | AUC | Δ |
|------|------|---|
| Full EVL-ECG | 75.4 | – |
| − MHCA | 71.8 | −3.6 |
| − OT-VFM | 73.2 | −2.2 |
| − Geometric Relation | 73.6 | −1.8 |
| CE Only (Baseline Student) | 68.3 | −7.1 |

All three modules are indispensable; MHCA contributes the most (feature-level alignment is foundational), while OT and geometric matching provide critical refinements.

### Comparison with Prev. SOTA KD

| KD Method | AUC | Clinical Accuracy |
|---------|------|----------|
| Vanilla KL | 71.0 | 64.1 |
| TinyBERT | 72.2 | 64.7 |
| Cross-tokenizer KD (Boizard 2025) | 73.0 | 65.4 |
| **EVL-ECG** | **75.4** | **66.5** |

Ours achieves a +2.4 AUC gain and +1.1 clinical accuracy improvement over the strongest baseline.

### Key Findings
- **Three-layer KD Complementarity**: Feature-level MHCA, distribution-level OT, and relation-level Geometric matching each handle a different granularity; removing any leads to performance drops.
- **2B Students Outperform Frontier VLMs**: In specialized tasks like ECG, a small model with effective KD can surpass general large-scale models.
- **OT Preserves Global Topology**: Removing OT leads to lead position confusion (supported by qualitative cases in the paper).
- **Relation Matching Captures Diagnostic Reasoning**: Without it, recognition of complex arrhythmias (e.g., Atrial Fibrillation + LBBB) significantly decreases.

## Highlights & Insights
- **The Three-Layer KD is a Generalized Paradigm**: The combination of MHCA (pointwise), OT (distributional), and Relation (structural) can be transferred to any scenario involving teacher/student heterogeneity where multi-granularity structure preservation is required.
- **Elegance of OT for Visual Token Mismatch**: Unlike truncation or padding, which lose information, OT's soft transport allows full alignment between different sequence lengths.
- **Clear Diagnostic Significance of Geometric Matching**: Distance and angle directly correspond to clinical ECG features like "interval duration and electrical axis orientation," aligning theory with clinical practice.
- **Insight on MHCA as Barycentric Projection**: Reinterpreting attention as OT projection provides a unified mathematical perspective for KD design.

## Limitations & Future Work
- Only validated on 2D image representations of ECG; the effectiveness of KD for raw 1D signals and time-series backbones remains untested.
- The loss weights $\lambda$ were determined via grid search; adaptive balancing (e.g., GradNorm) might reduce tuning effort.
- The teacher model is a closed-source frontier VLM; different design choices might be optimal if the teacher were an open-source ECG-specific model.
- 2B students are still relatively large; failure modes when compressing more aggressively to < 1B have not been studied.
- Lack of transfer testing with other physiological signals (e.g., EEG) to verify generalizability.

## Related Work & Insights
- **vs. Traditional KD (Vanilla KL / TinyBERT)**: These work on homogeneous architectures and fail with heterogeneous tokenizers and visual tokens.
- **vs. Cross-tokenizer KD (Boizard 2025)**: Only addresses tokenizer heterogeneity without considering visual tokens; EVL-ECG handles dual heterogeneity.
- **vs. RKD (Relational KD, Park 2019)**: The geometric relation matching in this work is an instantiation of RKD for ECG, adding an angular potential specifically designed for electrical axis characteristics.
- **Insight**: Medical VLM deployment is often limited by frontier model sizes. The three-layer KD paradigm could be extended to other medical imaging VLMs in radiology, pathology, and dermatology.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of MHCA, OT, and Geometric Relation is novel, though individual modules have precedents.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 7 ECG benchmarks, comparison with frontier VLMs, full ablation studies, and comparisons with existing KD.
- Writing Quality: ⭐⭐⭐⭐ Clear framework diagrams, formula-supported module descriptions, and insightful theoretical connections.
- Value: ⭐⭐⭐⭐⭐ Bedside ECG interpretation is a high-value clinical scenario; a 2B SOTA model is directly deployable on edge devices.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Fuse Before Transfer: Knowledge Fusion for Heterogeneous Distillation](../../ICCV2025/model_compression/fuse_before_transfer_knowledge_fusion_for_heterogeneous_distillation.md)
- [\[ICCV 2025\] Perspective-Aware Teaching: Adapting Knowledge for Heterogeneous Distillation](../../ICCV2025/model_compression/perspective-aware_teaching_adapting_knowledge_for_heterogeneous_distillation.md)
- [\[AAAI 2026\] Error Correction in Radiology Reports: A Knowledge Distillation-Based Multi-Stage Framework](../../AAAI2026/model_compression/error_correction_in_radiology_reports_a_knowledge_distillation-based_multi-stage.md)
- [\[ICML 2026\] Towards Resource-Efficient LLMs: End-to-End Energy Accounting of Distillation Pipelines](towards_resource-efficient_llms_end-to-end_energy_accounting_of_distillation_pip.md)
- [\[ICML 2026\] Don't Ignore the Tail: Decoupling top-K Probabilities for Efficient Language Model Distillation](dont_ignore_the_tail_decoupling_top-k_probabilities_for_efficient_language_model.md)

</div>

<!-- RELATED:END -->
