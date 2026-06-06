---
title: >-
  [Paper Note] Refining Context-Entangled Content Segmentation via Curriculum Selection and Anti-Curriculum Promotion
description: >-
  [ICML 2026][Segmentation][Camouflaged Object Segmentation] CurriSeg does not modify the segmentation network architecture but replaces the training schedule: it first uses a robust curriculum of "temporal loss statistics…
tags:
  - "ICML 2026"
  - "Segmentation"
  - "Camouflaged Object Segmentation"
  - "Curriculum Learning"
  - "Anti-Curriculum Learning"
  - "Frequency Domain Fine-tuning"
  - "Sample Reliability"
date: 2026-05-08
content_hash: a3798c74df0f8792
---

# Refining Context-Entangled Content Segmentation via Curriculum Selection and Anti-Curriculum Promotion

**Conference**: ICML 2026  
**arXiv**: [2602.01183](https://arxiv.org/abs/2602.01183)  
**Code**: https://github.com/ChunmingHe/CurriSeg  
**Area**: Image Segmentation / Camouflaged Object Detection / Curriculum Learning  
**Keywords**: Camouflaged Object Segmentation, Curriculum Learning, Anti-Curriculum Learning, Frequency Domain Fine-tuning, Sample Reliability

## TL;DR
CurriSeg does not modify the segmentation network architecture but replaces the training schedule: it first uses a robust curriculum of "temporal loss statistics + pixel entropy weighting" to stabilize the model, followed by an anti-curriculum "spectral-blindness" fine-tuning (removing high frequencies to force structural semantic understanding). This consistently improves FEDER / FSEL / RUN by 2–4% on camouflaged/polyp segmentation benchmarks like CHAMELEON / CAMO / COD10K / NC4K with zero additional parameters and shorter training times.

## Background & Motivation

**Background**: Context-entangled content segmentation (CECS) refers to segmentation tasks where the target and background highly overlap in texture, color, and shape, such as camouflaged object detection (COD), polyp segmentation, and medical lesion segmentation. Mainstream methods focus on network architectures: multi-scale aggregation (SINet, FEDER), attention refinement (Pang, FSEL), edge/uncertainty branches, Transformer context, and frequency domain modules.

**Limitations of Prior Work**: The training paradigm for almost all these methods follows a "shuffled-minibatch-standard supervision" approach, ignoring learning dynamics like "sample order, sample difficulty, and pixel difficulty." In CECS scenarios with extremely blurred features, this uniform treatment causes two issues: (1) "Easy samples" often contain spurious correlations—such as prominent background textures or clear target outlines—allowing the model to rely on these superficial shortcuts early on. (2) "Hard samples" mix truly informative hard samples with noisy or ambiguously labeled samples; assigning uniform weights allows dirty samples to degrade optimization.

**Key Challenge**: Standard curriculum learning (CL) aims to go "from easy to difficult," which paradoxically pushes the model into a "high-frequency texture bias" trap—since easy samples are exactly those with the most prominent high-frequency textures. Consequently, the model fails to learn when moving to hard samples. Conversely, completely ignoring curriculum leads to deviation caused by high-variance or label-noise samples. Furthermore, segmentation exhibits pixel-level heterogeneity—boundaries and low-contrast pixels are inherently difficult; treating them equal to homogeneous pixels allows gradients to be hijacked by unreliable regions.

**Goal**: (1) To design a truly robust curriculum for CECS that distinguishes hard-but-informative samples from noisy/ambiguous ones; (2) to manage pixel-level uncertainty so early training is not dominated by boundary noise; (3) to "increase difficulty" via anti-curriculum once the model stabilizes, forcing it to abandon high-frequency shortcuts in favor of low-frequency structures.

**Key Insight**: The authors draw an analogy to biological learning—predators solidify basic skills in simple scenarios before confronting complex environments. In machine learning, this translates to a "stabilize-then-perturb" two-stage approach: first, use a robust curriculum to converge the model on a clean subset for reliable basic features, then use anti-curriculum to actively deprive the model of high-frequency information, forcing it out of texture shortcuts.

**Core Idea**: Use temporal statistics (mean + variance) of sample losses to distinguish "hard-but-useful" from "noise/outliers," combined with pixel-entropy soft weighting for the first-stage robust curriculum. The second stage uses Spectral-Blindness Fine-Tuning (SBFT) to suppress high frequencies, forcing the model to use low-frequency structures and context for decision-making. The entire process requires no architectural changes or extra parameters.

## Method

### Overall Architecture
CurriSeg is a training scheduling framework wrapped around any CECS network, executed in two consecutive stages:

- **Stage I — Robust Curriculum Selection (RCS)**: A historical checkpoint $f_{\theta^{(k)}}$ is saved every $K=10$ epochs to calculate the difficulty score $d_i=1-\mathrm{IoU}(f_{\theta^{(k)}}(x_i), y_i)$ for all samples. The training subset is expanded by percentile (warm-up curriculum). Simultaneously, the mean and variance of difficulty over the last $K$ rounds provide sample-level weights $\omega_i$, while pixel entropy provides pixel-level soft weights $W_{h,w}(t)$.
- **Stage II — Anti-Curriculum Promotion (ACP)**: After the model stabilizes, Spectral-Blindness Fine-Tuning (SBFT) is performed. High-frequency components of the input images are suppressed based on a fixed cutoff frequency, forcing reliance on low-frequency structures and context.

The resulting segmentation model requires no architectural modifications and is evaluated directly on the original test sets.

### Key Designs

1. **Temporal Statistics Sample Weighting (TSSW) + Warm-up Curriculum**:
    - **Function**: Distinguishes "hard-but-informative" samples from "noise/outliers" at the sample level to construct a stable, noise-resistant training subset.
    - **Mechanism**: Difficulty $d_i=1-\mathrm{IoU}$ is calculated using historical checkpoints, maintaining a cyclic buffer of length $K$ for each sample to obtain mean $\mu_i=\frac{1}{K}\sum_k d_i^{(k)}$ and variance $\sigma_i^2=\frac{1}{K}\sum_k(d_i^{(k)}-\mu_i)^2$. After min-max normalization, two weights are derived: $\omega_i^\mu=1-\tilde\mu_i$ (prefers low mean loss/easier to learn) and $\omega_i^\sigma=\exp(-(\tilde\sigma_i^2-\sigma^*)^2/(2\gamma^2))$ (where $\sigma^*=0.5, \gamma=0.2$, using a bell-shaped function to prefer "moderate variance"). Low variance suggests a sample that is never learned (suspected label noise), while high variance suggests oscillation at the decision boundary (suspected ambiguous). The final weight $\omega_i=W^s_{\min}+(1-W^s_{\min})\cdot\omega_i^\mu\cdot\omega_i^\sigma\cdot(1-\tilde\mu_i(1-\tilde\sigma_i^2))$ specifically suppresses "high mean, low variance" outliers. The subset selection includes a warm-up: starting with only the easiest 60% subset $\mathcal{S}_t=\{i\mid d_i\le \mathrm{Per}(\{d_n\},p(t))\}$, with $p(t)$ growing linearly to 100% over $T_c$ epochs.
    - **Design Motivation**: Traditional CL uses static difficulty, which cannot differentiate if a sample is "hard but stable" or "noise." Using temporal statistics essentially treats the buffer as a lightweight "sample reliability probe." Stable high loss indicates true outliers, while oscillating high loss indicates samples worth learning.

2. **Pixel-wise Uncertainty Entropy (PUE)**:
    - **Function**: Suppresses gradients of uncertain pixels within a single image to prevent boundaries or low-contrast regions from hijacking optimization.
    - **Mechanism**: For each pixel prediction probability $p_{h,w}=\sigma(\hat y_{h,w})$, binary entropy is calculated as $H_{h,w}=-p\log_2 p-(1-p)\log_2(1-p)\in[0,1]$, maxed at $p=0.5$. The soft weight is defined as $W_{h,w}(t)=W_{\min}+(1-W_{\min})\cdot(1-\beta(t)\cdot H_{h,w})$, where $\beta(t)=1-t/T_c$ decays with the curriculum stage.
    - **Design Motivation**: Unlike prior work using uncertainty to "filter pseudo-labels," the goal here is not to hide uncertain regions but to prevent them from dominating early gradients. Retaining a minimum weight $W_{\min}$ ensures some signal remains to avoid completely ignoring boundary learning.

3. **Spectral-Blindness Fine-Tuning (SBFT) Anti-Curriculum**:
    - **Function**: Actively creates an "information scarcity" environment after stabilization, forcing the model to switch from high-frequency texture dependence to low-frequency structure and context dependency.
    - **Mechanism**: A 2D FFT is applied to the image, zeroing out high-frequency coefficients (or probabilistically attenuating them) before an IFFT back to the spatial domain. The model must segment based on the remaining low-frequency luminance/shape/context. This stage performs a short fine-tuning on the full set without RCS selection.
    - **Design Motivation**: In CECS, textures often overlap; high frequency can be interference. This is "anti-curriculum" in the literal sense—intentionally increasing difficulty to push the model out of its comfort zone.

### Loss & Training
- **Stage I**: The original segmentation loss (e.g., BCE+IoU) is multiplied by sample weight $\omega_i$ and pixel weight $W_{h,w}(t)$.
- **Stage II**: Standard segmentation loss is maintained, but the input is replaced with SBFT frequency-filtered images.
- No changes are made to the base network architecture, and no new trainable parameters are introduced. The only overhead is periodically saving checkpoints and the lightweight buffer for $\mu_i, \sigma_i^2$.

## Key Experimental Results

### Main Results
On four COD datasets across three backbones, CurriSeg applied to FEDER / FSEL / RUN yields:

| Baseline → + CurriSeg | Backbone | CHAMELEON $F_\beta\uparrow$ | CAMO $F_\beta\uparrow$ | COD10K $F_\beta\uparrow$ | NC4K $F_\beta\uparrow$ | Gain $\Delta$ |
|---|---|---|---|---|---|---|
| FEDER | ResNet50 | 0.850 | 0.775 | 0.715 | 0.808 | — |
| **FEDER+ (Ours)** | ResNet50 | **0.858** | **0.790** | **0.736** | **0.825** | **+2.46%** |
| FSEL | ResNet50 | 0.847 | 0.779 | 0.722 | 0.807 | — |
| **FSEL+ (Ours)** | ResNet50 | **0.856** | **0.792** | **0.742** | **0.823** | **+2.22%** |
| RUN | Res2Net50 | 0.879 | 0.815 | 0.764 | 0.830 | — |
| **RUN+ (Ours)** | Res2Net50 | **0.891** | **0.820** | **0.785** | **0.852** | **+2.23%** |
| RUN | PVT V2 | 0.877 | 0.861 | 0.810 | 0.868 | — |
| **RUN+ (Ours)** | PVT V2 | **0.893** | **0.879** | **0.828** | **0.889** | **+3.94%** |

In polyp segmentation (CVC-ColonDB, ETIS, PIS), CurriSeg consistently improves baselines like PolypPVT and CoInNet by approximately 2%.

### Ablation Study
Contributions of each module based on FEDER (ResNet50):

| Configuration | CHAMELEON $F_\beta\uparrow$ | COD10K $F_\beta\uparrow$ | Description |
|---|---|---|---|
| FEDER baseline | 0.850 | 0.715 | Standard training |
| + Vanilla CL | ~0.846 | ~0.711 | Slight drop; proves naive CL is harmful in CECS |
| + RCS (TSSW+PUE+warm-up) | 0.854 | 0.726 | Robust curriculum drives most gains |
| + ACP / SBFT (on top of RCS) | 0.858 | 0.736 | Anti-curriculum adds final boost |
| Full FEDER+ | **0.858** | **0.736** | Total +2.46% |

Training Cost (batch=2):

| Metric | FEDER | FEDER+ | FSEL | FSEL+ | RUN | RUN+ |
|---|---|---|---|---|---|---|
| Training Time (h) | 9.62 | **6.84** | 11.54 | **5.96** | 12.64 | **8.32** |
| GPU Mem (G) | 1.53 | 1.62 | 2.83 | 2.92 | 3.66 | 3.75 |

### Key Findings
- **Vanilla CL causes performance drops in CECS**: Naive "easy-to-hard" training leads to shortcuts via spurious textures.
- **RCS is the primary driver, ACP is the finishing touch**: RCS accounts for ~70% of total gains; SBFT extracts the final percentage.
- **Training time actually decreases**: Due to the warm-up skipping the hardest 40% of samples early on and faster convergence, training time for FEDER+ dropped from 9.62h to 6.84h.
- **Largest improvement on PVT V2 (+3.94%)**: Stronger backbones (Transformers) are more prone to high-frequency shortcuts, making anti-curriculum correction more effective.

## Highlights & Insights
- The method uses zero parameters and zero architectural changes; a pure training schedule modification yields a 2–4% gain, which is rare in a community obsessed with complex modules.
- It identifies the causal chain: "spurious texture correlation → lazy region → high-frequency reliance," and targets it with the ACP module.
- The TSSW "mean-variance joint outlier detection" can be applied to any supervised task with label noise or data ambiguity.
- The combination of pixel entropy and curriculum decay $\beta(t)$ creates a temporal "soft mask," preventing boundaries from being permanently ignored.

## Limitations & Future Work
- Verification is concentrated on CECS; performance on generic semantic segmentation (Cityscapes/ADE20K) remains unknown.
- SBFT cutoff frequency is a fixed hyperparameter; it lacks an adaptive mechanism for different backbones.
- The overhead of historical checkpoint evaluation grows linearly with data size, which may be costly for extremely large datasets (100K+).
- Explicit validation that the model "actually switches focus from high to low frequencies" (e.g., via Grad-CAM frequency analysis) would strengthen the narrative.

## Related Work & Insights
- **vs. Standard CL (Bengio et al. 2009)**: CurriSeg argues that in CECS, easy samples have spurious textures and hard samples have noise, requiring temporal statistics to distinguish them.
- **vs. Self-Paced Learning (Kumar 2010)**: SPL uses instantaneous snapshots; TSSW uses first- and second-order temporal statistics over $K$ rounds to identify oscillating ambiguous samples.
- **vs. CECS SOTA (FEDER, FSEL, RUN)**: While those methods modify architecture (frequency modules, edge branches), CurriSeg is orthogonal and serves as a complementary training-side enhancement.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First use of a two-stage "stabilize-then-perturb" curriculum with TSSW/PUE/SBFT in the CECS domain.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers various backbones and datasets; lacks generic segmentation tasks.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear narrative from observation ("Vanilla CL fails") to diagnosis to design.
- **Value**: ⭐⭐⭐⭐ Parameter-free, faster training, and highly applicable for industrial deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] JoDiffusion: Jointly Diffusing Image with Pixel-Level Annotations for Semantic Segmentation Promotion](../../AAAI2026/segmentation/jodiffusion_jointly_diffusing_image_with_pixel-level_annotations_for_semantic_se.md)
- [\[CVPR 2026\] INSID3: Training-Free In-Context Segmentation with DINOv3](../../CVPR2026/segmentation/insid3_training-free_in-context_segmentation_with_dinov3.md)
- [\[CVPR 2026\] RS-SSM: Refining Forgotten Specifics in State Space Model for Video Semantic Segmentation](../../CVPR2026/segmentation/rs-ssm_refining_forgotten_specifics_in_state_space_model_for_video_semantic_segm.md)
- [\[CVPR 2026\] Pointer-CAD: Unifying B-Rep and Command Sequences via Pointer-based Edges & Faces Selection](../../CVPR2026/segmentation/pointer-cad_unifying_b-rep_and_command_sequences_via_pointer-based_edges_faces_s.md)
- [\[ICCV 2025\] CAVIS: Context-Aware Video Instance Segmentation](../../ICCV2025/segmentation/cavis_context-aware_video_instance_segmentation.md)

</div>

<!-- RELATED:END -->
