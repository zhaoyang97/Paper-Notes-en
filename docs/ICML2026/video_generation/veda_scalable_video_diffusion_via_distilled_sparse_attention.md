---
title: >-
  [Paper Note] VEDA: Scalable Video Diffusion via Distilled Sparse Attention
description: >-
  [ICML 2026][Video Generation][Sparse Attention] VEDA reformulates the sparse attention problem in video DiTs as "explicit distillation of the full-attention structure"—utilizing statistics-aware tile scoring + head-aware…
tags:
  - "ICML 2026"
  - "Video Generation"
  - "Sparse Attention"
  - "Video Diffusion Transformer"
  - "Distillation Learning"
  - "Hardware Optimization"
date: 2026-05-08
content_hash: 8733b6fc4d541fc1
---

# VEDA: Scalable Video Diffusion via Distilled Sparse Attention

**Conference**: ICML 2026  
**arXiv**: [2605.30325](https://arxiv.org/abs/2605.30325)  
**Code**: TBD  
**Area**: Video Generation / Diffusion Models / Model Acceleration  
**Keywords**: Sparse Attention, Video Diffusion Transformer, Distillation Learning, Hardware Optimization

## TL;DR
VEDA reformulates the sparse attention problem in video DiTs as "explicit distillation of the full-attention structure"—utilizing statistics-aware tile scoring + head-aware grouping search + hardware-efficient kernels to maintain generation quality at extreme 90-95% sparsity, delivering 5.1× end-to-end speedup and 10.5× attention acceleration for Waver-12B 720P 10-second videos.

## Background & Motivation

**Background**: Video Diffusion Transformers (DiT) have become the mainstream for high-fidelity video synthesis, but the $O(N^2)$ computation bottleneck of self-attention becomes extremely severe during high-resolution, long-duration generation.

**Limitations of Prior Work**: Existing sparse attention methods face two fundamental issues under high pruning regimes (≥ 90%):
- **Static methods** (SVG, STA) rely on predefined spatio-temporal masks and lack adaptability to head-specific attention geometries.
- **Dynamic methods** (VSA, VMOBA) rely on implicit learning without explicit supervision; using coarse statistics like mean pooling ignores critical signal peaks.

**Key Challenge**: Highly sparse pruning leads to structural artifacts such as "moire distortion / spatial warping / temporal flickering." However, experiments reveal this is **not caused by the sparsity ratio itself**, but by the insufficient tile-level structural alignment between the sparse mask and the full attention.

**Goal**: To achieve radical sparsification and actual acceleration of video DiTs while preserving generation quality.

**Key Insight**: A key observation is that "oracle" masks (obtained from full-attention Top-k) maintain high quality even at 90% sparsity. This inspires the use of explicit supervision for tile selection targets rather than relying on implicit learning via diffusion targets.

**Core Idea**: Reformulate sparse tile selection as explicit distillation of the full-attention structure, combined with head-aware grouping to address head heterogeneity and hardware-efficient kernels for real-world speedup.

## Method

### Overall Architecture
VEDA consists of three core modules:
- **Distilled Tile Scoring**: Learns to reconstruct tile-level scores of the full attention using a lightweight estimator, mapping token-level dense attention to sparse tile masks.
- **Head-aware Grouping Search**: Searches for the optimal tile grouping $(p_t, p_h, p_w)$ for each attention head.
- **Hardware-efficient Kernels**: Implements tile-skip attention kernels via ThunderKittens DSL and NVIDIA Hopper TMA, achieving 80% of the compute efficiency of FlashAttention-3.

### Key Designs

1. **Statistics-aware Tile Scoring Estimator (TripPool)**:

    - **Function**: Reconstructs tile-level scores of full attention through compressed tile representations to generate sparse masks.
    - **Mechanism**: Constructs a **TripPool descriptor** for each query/key tile—a concatenation of mean, max, and min values: $\text{TripPool}[\cdot] = \text{Avg}[\cdot] \oplus \text{Max}[\cdot] \oplus \text{Min}[\cdot]$. These are mapped to a shared latent space via head-specific MLP projectors $\phi_q, \phi_k$ to compute predicted scores $S_{ij}^{\text{pred}} = \frac{\phi_q(\text{TripPool}[\tilde{Q}_i]) \cdot \phi_k(\text{TripPool}[\tilde{K}_j])^\top}{\sqrt{d'}}$. Finally, KL divergence loss $\mathcal{L}_{\text{distill}} = \mathcal{D}_{KL}(A^{\text{tgt}} \| A^{\text{pred}})$ aligns predictions with full attention.
    - **Design Motivation**: Unlike mean pooling which ignores signal peaks, TripPool's max/min statistics preserve critical dependencies; explicit distillation avoids the drift of implicit learning; the **critical stop-gradient operation** decouples mask learning from feature learning, preventing perturbations to the pre-trained generation manifold.

2. **Head-aware Grouping Search**:

    - **Function**: Performs offline searches for the optimal spatio-temporal tile grouping configuration for each attention head in every layer.
    - **Mechanism**: Restricts tile configurations to the factorization of the hardware tile size $B$: $\Omega = \{(p_t, p_h, p_w) \in \mathbb{N}^3 \mid p_t p_h p_w = B\}$. For each candidate $\pi$, the sparse approximation error of the full attention output is minimized on a calibration set: $\pi^*_{l, h} = \arg\min_{\pi \in \Omega} \mathbb{E}_{x \sim \mathcal{D}_{\text{cal}}} \|O^{\text{fu}}_{l, h}(x) - O^{\text{sp}}_{l, h}(x; \pi)\|_F^2$.
    - **Design Motivation**: Attention heads exhibit significant heterogeneity in spatial/temporal dependencies; uniform grouping leads to a drop in tile recall at high sparsity; targeted configurations preserve critical information for different heads.

3. **Tile-skip Hardware Kernels + Two-stage Training**:

    - **Function**: Efficiently executes sparse masks on the GPU while avoiding convergence issues through stable two-stage training.
    - **Mechanism**: Stage 1 freezes the backbone and only trains projectors for 1000 steps to align sparse predictions; Stage 2 unfreezes all parameters for fine-tuning at the target sparsity. Utilizes asynchronous TMA + Warp-specialization: producer warps non-contiguously fetch selected key/value tiles from global memory to shared memory while consumer warps simultaneously execute Tensor Core operations, reaching ~80% FlashAttention-3 efficiency.
    - **Design Motivation**: Two-stage decoupling prevents backpropagation from damaging the pre-trained manifold; hardware optimization ensures algorithmic sparsity translates into actual end-to-end speedup rather than kernel overhead.

### Loss & Training
$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{diff}} + \lambda \mathcal{L}_{\text{distill}}$, using row-level KL divergence distillation + standard diffusion denoising. **Stop-gradient** ensures the backbone features do not receive gradient backpropagation from the mask estimator—experiments prove that allowing such backpropagation leads to significant degradation in generation quality.

## Key Experimental Results

### Main Results (Comparison of Full Attention vs. VSA on Waver-1B and Wan2.1-1.3B)

| Model | Method | Sparsity | Subject Consistency | Background Consistency | Motion Smoothness | Aesthetic Quality | End-to-End Latency |
|------|------|--------|---------|---------|--------|--------|----------|
| Waver-1B | Full Attn | 0% | 0.938 | 0.955 | 0.979 | 0.693 | 69.3s |
| Waver-1B | VSA | 87.5% | 0.933 | 0.949 | 0.978 | 0.692 | 34.3s |
| Waver-1B | **VEDA** | **90%** | **0.940** | **0.954** | **0.980** | **0.699** | **31.9s** |
| Waver-1B | **VEDA** | **95%** | 0.934 | 0.951 | 0.978 | 0.698 | **30.6s** |
| Wan2.1-1.3B | Full Attn | 0% | 0.940 | 0.969 | 0.977 | 0.670 | 58.5s |
| Wan2.1-1.3B | **VEDA** | **90%** | 0.887 | 0.941 | 0.972 | 0.663 | **37.6s** |

### Ablation Study

| Component | Configuration | Metric ↓ | Note |
|------|------|------|------|
| Tile Stats | Mean Pooling | 0.965 | Ignores peaks |
| Tile Stats | Max / Min | 0.982 | Misses mid-importance |
| Tile Stats | **TripPool** | **0.912** | Preserves key dependencies |
| Grouping | Static [8, 8, 2] | +3.2% Motion Loss | Spatial bias |
| Grouping | Static [4, 4, 8] | Baseline | Balanced config |
| Grouping | **Head-aware Dynamic** | +7.2% Motion / +9.6% Overall | Adapts to heterogeneity |

### Key Findings
- **Mask Accuracy Dominates Performance**: At a fixed 90% sparsity, the generation quality of the "oracle" mask is far superior to that of the mean-pooling mask—the root of the problem is alignment quality, not the sparsity ratio.
- **Significant Head Heterogeneity**: Spatial/temporal dependency patterns vary greatly across different layers and heads; uniform grouping fails at high sparsity.
- **Scalability**: Waver-12B 720P 10-second video generation achieves 5.1× end-to-end speedup + 10.5× attention speedup, reducing attention overhead from 92% to 50%; VEDA speedup increases with sequence length.

## Highlights & Insights
- **Fundamental Experimental Observation**: The "oracle mask" experiment precisely identifies structural alignment rather than sparsity ratio as the true bottleneck, overturning previous assumptions and laying the foundation for the method design.
- **Paradigm Shift to Explicit Supervision**: Instead of letting the diffusion target implicitly shape the sparse structure, explicit distillation directly supervises tile scores to avoid drift; the stop-gradient design cleverly protects the pre-trained generation manifold.
- **Fine-grained Head-aware Grouping**: Recognizing head heterogeneity and searching for specific spatio-temporal grouping configurations provides much finer granularity than concurrent static or global dynamic methods like VSA, making it transferable to other multi-head Transformer acceleration tasks.
- **Algorithm-Hardware Co-design**: From TMA asynchronous transfers to Warp-specialized kernels, the implementation translates theoretical FLOPs reduction into real end-to-end speedup, completing the engineering loop.

## Limitations & Future Work
- While two-stage training is stable, it requires manual design of learning rates and step counts, needing improved generalization.
- At 95%+ sparsity, further kernel fusion is required to enhance MFU.
- Head-aware grouping depends on an offline calibration set and may require re-searching for different data distributions.
- The robustness of TripPool to anomalous distributions is not fully discussed (max/min values are susceptible to outliers).

## Related Work & Insights
- **vs SVG / STA** (Static Sparsity): These rely on predefined patterns and lack adaptability; this work achieves content- and head-sensitive dynamic selection via explicit distillation.
- **vs VSA / VMOBA** (Dynamic Sparsity): These rely on implicit diffusion targets and coarse pooling; this work's explicit distillation and refined statistics capture the full-attention structure more accurately.
- **vs Other Acceleration** (Cache reuse PAB / TeaCache, Distillation CausVid): VEDA is orthogonal to these and can be used in combination.

## Rating
- Novelty: ⭐⭐⭐⭐⭐  First systematic introduction of explicit supervision + head-aware grouping for video DiT sparsification; findings on mask accuracy redefine the understanding of sparse attention bottlenecks.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐  Covers multiple model scales (1B / 12B), resolutions (480P / 720P), long sequences (34K-245K), human evaluation + VBench, and detailed ablations.
- Writing Quality: ⭐⭐⭐⭐⭐  Logical progression with strong evidence-driven findings; individual contributions of each module are clear.
- Value: ⭐⭐⭐⭐⭐  5.1× speedup is significant for industrial applications; the sparse attention design philosophy is also valuable for LLM acceleration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Light Forcing: Accelerating Autoregressive Video Diffusion via Sparse Attention](light_forcing_accelerating_autoregressive_video_diffusion_via_sparse_attention.md)
- [\[ICML 2026\] DFSAttn: Dynamic Fine-Grained Sparse Attention for Efficient Video Generation](dfsattn_dynamic_fine-grained_sparse_attention_for_efficient_video_generation.md)
- [\[ICML 2026\] Lightning Unified Video Editing via In-Context Sparse Attention](lightning_unified_video_editing_via_in-context_sparse_attention.md)
- [\[NeurIPS 2025\] VSA: Faster Video Diffusion with Trainable Sparse Attention](../../NeurIPS2025/video_generation/vsa_faster_video_diffusion_with_trainable_sparse_attention.md)
- [\[ICML 2026\] Attention Sparsity is Input-Stable: Training-Free Sparse Attention for Video Generation via Offline Sparsity Profiling and Online QK Co-Clustering](attention_sparsity_is_input-stable_training-free_sparse_attention_for_video_gene.md)

</div>

<!-- RELATED:END -->
