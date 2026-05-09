---
title: >-
  [Paper Note] MoFu: Scale-Aware Modulation and Fourier Fusion for Multi-Subject Video Generation
description: >-
  [AAAI 2026][Video Generation][Multi-subject video generation] This paper proposes MoFu, which addresses two fundamental challenges in multi-subject video generation—**scale inconsistency** and **permutation sensitivity**—through two core modules: Scale-Aware Modulation (SMO, an LLM-guided scale-aware modulation mechanism) and Fourier Fusion (an FFT-based permutation-invariant feature fusion strategy). The work additionally introduces the MoFu-1M training dataset and the MoFu-Bench evaluation benchmark.
tags:
  - AAAI 2026
  - Video Generation
  - Multi-subject video generation
  - scale consistency
  - permutation invariance
  - Fourier fusion
  - DiT
date: 2026-05-08
content_hash: 10b030e272a5aa92
---

# MoFu: Scale-Aware Modulation and Fourier Fusion for Multi-Subject Video Generation

**Conference**: AAAI 2026
**arXiv**: [2512.22310](https://arxiv.org/abs/2512.22310)
**Code**: Unavailable
**Area**: Video Understanding / Video Generation
**Keywords**: Multi-subject video generation, scale consistency, permutation invariance, Fourier fusion, DiT

## TL;DR

This paper proposes MoFu, which addresses two fundamental challenges in multi-subject video generation—**scale inconsistency** and **permutation sensitivity**—through two core modules: Scale-Aware Modulation (SMO, an LLM-guided scale-aware modulation mechanism) and Fourier Fusion (an FFT-based permutation-invariant feature fusion strategy). The work additionally introduces the MoFu-1M training dataset and the MoFu-Bench evaluation benchmark.

## Background & Motivation

### State of the Field

Multi-subject video generation aims to synthesize videos from text prompts and multiple reference images while preserving the visual fidelity and natural scale of each subject. The quality of video generation has steadily improved with advances in large-scale models such as Wan2.1, HunyuanVideo, and Sora. Recent methods including Phantom, SkyReels-A2, and VACE have begun to explore multi-subject generation, yet fundamental challenges remain.

### Limitations of Prior Work

**Scale Inconsistency**: Reference images depict subjects at varying zoom levels—e.g., a close-up of a small bird versus a wide shot of an elephant—causing the generated video to render subjects at physically implausible sizes, such as a bowl larger than a table or a sparrow as large as an elephant.

**Permutation Sensitivity**: Existing methods process reference images sequentially (inserting them one by one along the frame or channel dimension), such that **changing the input order causes subject deformation, disappearance, or physically unreasonable interactions**. Furthermore, computational cost scales sharply with the number of reference images.

### Limitations of Prior Attempts

- **Addressing scale**: Phantom and SkyReels attempt to fuse text prompts into reference image representations, implicitly exploiting spatial relationships encoded in prompts (e.g., "a girl holding a dog"). However, these methods **lack an explicit mechanism** to interpret and enforce natural scale.
- **Addressing permutation**: CustomVideo and MAGREF concatenate multiple reference images onto a single canvas to form a unified input. This introduces **spatial bias**, where central or larger subjects may receive disproportionate attention.

### Root Cause

Multi-subject generation requires **simultaneously guaranteeing scale plausibility and input-order invariance**. The root cause of scale inconsistency is that scale differences across reference images are not explicitly considered; the root cause of permutation sensitivity is that sequential processing or spatial concatenation inevitably introduces positional bias. The two problems are intertwined yet require distinct technical solutions.

### Starting Point

The paper designs two orthogonal yet complementary modules: an LLM extracts implicit scale cues from text to **explicitly modulate** features (addressing scale), while FFT transforms reference features into the frequency domain for **symmetric summation** to achieve permutation-invariant fusion (addressing permutation).

## Method

### Overall Architecture

MoFu is built upon a DiT (Diffusion Transformer) backbone and integrates three components:
1. **Scale-Aware Modulation (SMO)**
2. **Fourier Fusion**
3. **Scale-Permutation Stability Loss (SPSL)**

### Key Designs

#### 1. Scale-Aware Modulation (SMO)

**Function**: Extracts relative scale relationships among subjects from text prompts and injects them into the generation process.

**Mechanism**:

**Step 1: LLM-based scale cue extraction**
- A frozen LLM (Qwen2.5) encodes the text prompt: $\mathbf{e}_p = \text{LLM}(p) \in \mathbb{R}^d$
- The extracted representation is the [CLS] embedding rather than text output, serving as a scale-aware semantic descriptor
- Carefully designed prompts elicit pairwise relative size reasoning among entities in the scene

**Step 2: Scale Control Adapter (SCA) predicts modulation parameters**
- A lightweight MLP maps the semantic embedding to a modulation triplet: $\gamma, \beta, \eta = f_{\text{SCA}}(\mathbf{e}_p)$
- $\gamma, \beta$: scale and shift factors
- $\eta$: gating factor controlling the residual connection

**Step 3: Adaptive feature modulation**

$$\hat{\mathbf{F}} = \gamma \odot \mathbf{F} + \beta, \quad \mathbf{F}_{out} = \mathbf{F} + \eta \cdot \text{Layer}(\hat{\mathbf{F}})$$

where Layer denotes an MHA or FFN module.

**Design Motivation**:
- Text prompts are a natural source of scale information—"a girl holding a kitten" implicitly conveys that the cat is much smaller than the person
- The commonsense reasoning capability of LLMs is leveraged to interpret these implicit scale relations
- Injection via DiT's existing modulation mechanism preserves the model architecture intact

#### 2. Fourier Fusion

**Function**: Fuses multiple reference images into a permutation-invariant unified representation.

**Mechanism**:

**Mathematical foundation**: A theorem from high-dimensional probability states that inner products between high-dimensional random vectors tend toward zero, i.e., they are approximately orthogonal. This implies that direct summation in the frequency domain does not cause severe information interference.

**Step 1: Reference image preprocessing**
- Grounded-SAM segments each reference image; subject regions are cropped
- A 3×3 CNN encoder extracts feature maps: $\mathbf{F}_i = \mathcal{E}(x_i) \in \mathbb{R}^{d \times H \times W}$

**Step 2: FFT transformation and frequency decomposition**
$$\mathcal{F}_i = \text{FFT}(\mathbf{F}_i)$$
$$\mathcal{F}_i^{\text{HF}} = M_{\text{freq}} \odot \mathcal{F}_i, \quad \mathcal{F}_i^{\text{LF}} = (1 - M_{\text{freq}}) \odot \mathcal{F}_i$$

**Step 3: Direct summation via approximate orthogonality**
$$\mathcal{F}^{\text{HF}} = \sum_{i=1}^{N} \mathcal{F}_i^{\text{HF}}, \quad \mathcal{F}^{\text{LF}} = \sum_{i=1}^{N} \mathcal{F}_i^{\text{LF}}$$

**Step 4: IFFT reconstruction**
$$\mathbf{F}_{\text{fused}} = \text{IFFT}(\mathcal{F}^{\text{HF}} + \mathcal{F}^{\text{LF}})$$

The fused representation is concatenated with video features to serve as a generation condition.

**Design Motivation**:
- Addition is inherently **commutative**; regardless of input order, the summation result is identical → permutation invariance
- Frequency decomposition preserves the distinction between high-frequency (texture details) and low-frequency (global structure) components
- Sequential concatenation's positional bias and canvas concatenation's spatial bias are both avoided
- Computational overhead is minimal: only FFT + element-wise summation + IFFT are required

#### 3. Scale-Permutation Stability Loss (SPSL)

**Function**: Jointly supervises scale consistency and permutation invariance.

**Scale Loss**: Adaptively weights the standard MSE loss by the area ratio of reference masks:

$$w_r = \frac{\exp(a_r)}{\sum_{r'}\exp(a_{r'})}, \quad \mathbf{M} = \sum_{r=1}^{R} w_r \cdot \text{Resize}(m_r)$$

$$\mathcal{L}_{\text{scale}} = \frac{\sum(\mathcal{L}_{\text{mse}} \odot \mathbf{M})}{\sum \mathbf{M} + \epsilon}$$

Subjects with larger area ratios receive higher loss weights.

**Permutation Loss**: Computes MSE loss across $P$ permutations in the frequency domain:

$$\mathcal{L}_{\text{perm}} = \frac{1}{P} \sum_{p=1}^{P} \|F(\mathcal{R}_p) - F(\mathcal{R}_{\text{ref}})\|_2^2$$

**Total loss**: $\mathcal{L}_{\text{SPSL}} = \mathcal{L}_{\text{scale}} + \mathcal{L}_{\text{perm}}$

**Design Motivation**: SMO and Fourier Fusion require supervision signals for scale and permutation information, respectively; SPSL provides these signals. Without SPSL, neither module can learn effectively.

### Loss & Training

- **MoFu-1M dataset**: A two-stage filtering pipeline reduces 15M clips to 2.5M high-quality clips, from which 1M are selected as the final training set
- AdamW optimizer, $\beta_1=0.9$, $\beta_2=0.999$, weight decay 0.01
- Initial learning rate 1e-5, cosine annealing with periodic restarts
- 16 × NVIDIA H800 GPUs, trained for 7 days
- Input resolution 480P, 81 frames

## Key Experimental Results

### Main Results

| Method | Aesthetics↑ | FaceSim↑ | GmeScore↑ | Motion↔ | ScaleScore↑ | SubjectSim↑ |
|--------|-------------|----------|-----------|---------|-------------|-------------|
| Phantom | 0.355 | 0.375 | 0.706 | 0.229 | 0.536 | 0.748 |
| SkyReels-A2 | 0.286 | 0.341 | 0.691 | 0.233 | 0.527 | 0.737 |
| VACE | 0.392 | 0.247 | 0.732 | 0.214 | 0.547 | 0.692 |
| MAGREF | 0.369 | 0.362 | 0.717 | 0.207 | 0.511 | 0.731 |
| **MoFu** | **0.401** | **0.396** | **0.745** | 0.221 | **0.585** | **0.755** |

- ScaleScore shows the largest improvement: 0.585 vs. the second-best VACE at 0.547 (+7.0%)
- MoFu achieves comprehensive superiority across nearly all metrics

### Ablation Study

| Configuration | Effect | Explanation |
|---------------|--------|-------------|
| w/o SMO | Sparrow rendered as large as elephant | Scale relationships entirely lost |
| w/ SMO | Natural body-size proportions preserved | LLM scale cues are effective |
| w/o Fourier Fusion | Changing reference order causes wooden crate to disappear | Permutation sensitive |
| w/ Fourier Fusion | All subjects generated completely regardless of order | Permutation invariant |

Note: The authors explain that SPSL is not ablated independently because SMO and Fourier Fusion depend on SPSL's scale and permutation supervision signals, respectively; retaining either module without SPSL is not meaningful.

### MoFu-Bench Statistics

- 1,000 subject–text pairs
- Reference image category distribution: person 26%, animal 17%, clothing 10%, object 30%, cartoon 4%, other 13%
- Prompt length predominantly 100–200 words
- Up to 3 reference images per case, with systematic scale variations and randomized permutations

### Key Findings

1. **Scale inconsistency is as much a semantic understanding problem as a visual one**: addressing it requires commonsense knowledge such as "an elephant is far larger than a sparrow"
2. **The theoretical grounding for frequency-domain summation is robust**: high-dimensional approximate orthogonality makes simple summation an effective permutation-invariant operation
3. **Using softmax-normalized area ratios as loss weights** in the scale loss elegantly converts spatial priors into supervision signals
4. **I2V methods (e.g., MAGREF) achieve strong FaceSim scores** but poor scale consistency, demonstrating that reference conditioning and scale reasoning are distinct capabilities

## Highlights & Insights

1. **Elegant integration of theory and practice**: High-dimensional approximate orthogonality provides rigorous theoretical justification for frequency-domain summation
2. **LLM as a scale reasoner**: LLM commonsense knowledge is creatively leveraged to interpret implicit scale relations encoded in text
3. **Two problems, two modules, one loss**: The symmetry and complementarity of the design reflect sound systems thinking
4. **MoFu-Bench fills a gap**: The first benchmark specifically designed to evaluate scale consistency and permutation invariance in multi-subject video generation
5. **Solid engineering**: Multi-stage data filtering from 15M to 1M clips; evaluation metrics span six dimensions

## Limitations & Future Work

1. **Emphasis on spatial reasoning may constrain temporal modeling**: Fine-grained temporal interactions in highly dynamic scenes may be handled less effectively
2. **Dependency on high-quality reference inputs**: Performance may degrade when reference images are sparse or noisy
3. **Cost of generality in a unified architecture**: Domain-specific priors (e.g., face-specific optimization) are absent
4. **ScaleScore relies on GPT-4o judgments**: The reliability of this metric is contingent on GPT-4o's own scale reasoning capability
5. **Subject count limitation**: Current demonstrations involve at most 3 subjects; performance with larger numbers remains to be validated
6. **High training cost**: 16×H800 GPUs for 7 days represents a substantial resource requirement

## Related Work & Insights

- **FFT in representation learning**: Fourier-domain operations as a tool for achieving specific mathematical properties (e.g., permutation invariance) merit further exploration in other settings
- **Flexibility of DiT's modulation mechanism**: The $\gamma \cdot \text{LayerNorm}(h) + \beta$ modulation paradigm in DiT provides a convenient injection point for diverse conditioning signals
- **LLM as an intermediate reasoner**: Extracting embeddings from LLMs for downstream tasks—rather than using their text outputs—is an increasingly prevalent pattern in multimodal systems
- **Systematic data construction pipeline**: The multi-stage pipeline of scene detection → aesthetic filtering → text alignment → subject extraction → face processing is broadly reusable

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The combination of Fourier fusion and LLM-guided scale-aware modulation is highly creative and theoretically well-grounded
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Quantitative and qualitative evaluations are comprehensive, though SPSL is not ablated independently
- **Writing Quality**: ⭐⭐⭐⭐ — Problem formulation is clear and method presentation is systematic
- **Value**: ⭐⭐⭐⭐ — The first systematic treatment of scale consistency and permutation invariance in multi-subject generation; both the benchmark and the method make meaningful contributions

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] BindWeave: Subject-Consistent Video Generation via Cross-Modal Integration](../../ICLR2026/video_generation/bindweave_subject-consistent_video_generation_via_cross-modal_integration.md)
- [\[CVPR 2026\] Rethinking Position Embedding as a Context Controller for Multi-Reference and Multi-Shot Video Generation](../../CVPR2026/video_generation/rethinking_position_embedding_as_a_context_controller_for_multi-reference_and_mu.md)
- [\[ICCV 2025\] DIVE: Taming DINO for Subject-Driven Video Editing](../../ICCV2025/video_generation/dive_taming_dino_for_subject-driven_video_editing.md)
- [\[AAAI 2026\] FilmWeaver: Weaving Consistent Multi-Shot Videos with Cache-Guided Autoregressive Diffusion](filmweaver_weaving_consistent_multi-shot_videos_with_cache-guided_autoregressive.md)
- [\[ICLR 2026\] Geometry-aware 4D Video Generation for Robot Manipulation](../../ICLR2026/video_generation/geometry-aware_4d_video_generation_for_robot_manipulation.md)

</div>

<!-- RELATED:END -->
