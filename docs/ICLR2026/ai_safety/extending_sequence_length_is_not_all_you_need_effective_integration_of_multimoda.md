---
title: >-
  [Paper Note] Extending Sequence Length is Not All You Need: Effective Integration of Multimodal Signals for Gene Expression Prediction
description: >-
  [ICLR 2026][AI Safety][gene expression prediction] This paper challenges the prevailing "longer is better" paradigm in gene expression prediction, demonstrating that current SSM models fundamentally rely only on proximal information. It further identifies background chromatin signals (DNase-seq/Hi-C) as confounding variables that introduce spurious correlations, and proposes the Prism framework, which applies backdoor adjustment for deconfounding—achieving state-of-the-art performance with only 2k-length sequences, surpassing methods that use 200k-length sequences.
tags:
  - ICLR 2026
  - AI Safety
  - gene expression prediction
  - epigenomic signals
  - causal inference
  - backdoor adjustment
  - confounding variables
date: 2026-05-08
content_hash: a4fa592df4836eac
---

# Extending Sequence Length is Not All You Need: Effective Integration of Multimodal Signals for Gene Expression Prediction

**Conference**: ICLR 2026
**arXiv**: [2602.21550](https://arxiv.org/abs/2602.21550)
**Code**: [https://github.com/yangzhao1230/Prism](https://github.com/yangzhao1230/Prism)
**Area**: AI Safety
**Keywords**: gene expression prediction, epigenomic signals, causal inference, backdoor adjustment, confounding variables

## TL;DR
This paper challenges the prevailing "longer is better" paradigm in gene expression prediction, demonstrating that current SSM models fundamentally rely only on proximal information. It further identifies background chromatin signals (DNase-seq/Hi-C) as confounding variables that introduce spurious correlations, and proposes the Prism framework, which applies backdoor adjustment for deconfounding—achieving state-of-the-art performance with only 2k-length sequences, surpassing methods that use 200k-length sequences.

## Background & Motivation

**Background**: Gene expression prediction aims to predict mRNA expression levels (CAGE values) from DNA sequences. Mainstream approaches focus on extending input sequence length to capture distal enhancers (potentially hundreds of thousands of base pairs away), commonly using SSMs (e.g., Caduceus, Mamba) for linear-complexity long-sequence modeling. Increasingly, methods also incorporate multimodal epigenomic signals (H3K27ac, DNase-seq, Hi-C) to provide cell-type-specific information.

**Limitations of Prior Work**: (a) The fixed-size hidden states of SSMs struggle to retain information from ultra-long sequences and exhibit a recency bias; experiments show that Caduceus performance degrades continuously beyond 2k sequence length, and Seq2Exp (trained on 200k) shows almost no performance change when truncated to 2.5k at test time. (b) Existing methods naively concatenate multimodal epigenomic signals, ignoring the distinct biological roles of different signal types.

**Key Challenge**: Different epigenomic signals play fundamentally different roles—H3K27ac directly marks active regulatory elements ("foreground signals"), while DNase-seq/Hi-C reflect background chromatin accessibility ("background signals"). Models develop excessive dependence on background signals during training (performance drops sharply upon removal), yet these signals contribute little independently. This asymmetry indicates that models learn spurious correlations: open chromatin regions co-occur with high expression, but gene expression can occur independently in low-accessibility regions.

**Goal**: (a) Demonstrate that long-sequence modeling is not effective with current tools; (b) identify and eliminate confounding effects introduced by background chromatin signals; (c) achieve state-of-the-art performance using short sequences combined with proper multimodal signal integration.

**Key Insight**: The paper frames multimodal signal fusion as a causal inference problem—modeling background chromatin state as a confounding variable $C$, and applying backdoor adjustment to block the path $H \leftarrow C \rightarrow Y$, retaining only the direct causal effect $H \rightarrow Y$.

**Core Idea**: Rather than blindly extending sequence length, the key is to correctly integrate proximal epigenomic signals by accounting for the distinct roles of different signals through causal deconfounding.

## Method

### Overall Architecture
Input: a 2k bp DNA sequence $X$ centered on the TSS + multimodal epigenomic signals $S$ (H3K27ac, DNase-seq, Hi-C). A signal encoder $g_\theta$ maps $S$ to high-dimensional features $H$; a confounder encoder $g_\omega$ learns $n$ weight vectors from $S$ representing different background chromatin states; a predictor $h_\phi$ (based on Caduceus) integrates $X$ and the deconfounded $H$ to predict expression level $Y$.

### Key Designs

1. **Empirical Analysis of Long-Sequence Ineffectiveness**:

    - Function: Demonstrates through controlled experiments that current SSMs cannot genuinely exploit long-sequence information.
    - Mechanism: (a) Caduceus performance degrades continuously beyond 2k; (b) Seq2Exp, despite training on 200k sequences, shows almost no performance change when truncated to 2.5k at test time—indicating it fundamentally relies only on proximal information.
    - Design Motivation: Challenges the inertia of the "longer is better" paradigm and redirects attention to the underexplored problem of signal integration.

2. **Structural Causal Model (SCM)**:

    - Function: Formalizes the confounding problem in epigenomic signals using a causal graph.
    - Mechanism: Defines three variables—high-dimensional epigenomic features $H$, gene expression $Y$, and background chromatin state $C$. Causal relationships: $H \rightarrow Y$ (direct regulatory effect) and $H \leftarrow C \rightarrow Y$ (confounding path). Standard training optimizes $P(Y|H)$, which conflates both paths.
    - Design Motivation: Recasts the signal fusion problem as a causal inference problem, providing a theoretical framework for deconfounding.

3. **Confounder Encoder + Backdoor Adjustment**:

    - Function: Learns multiple representations of background chromatin states and removes confounding effects via backdoor adjustment.
    - Mechanism: The confounder encoder $g_\omega$ generates $n$ weight vectors $\{a_1, \dots, a_n\}$ from raw signals $S$, each representing a background state. Prediction applies intervention: $P(Y|X, do(H)) = \frac{1}{n} \sum_{i=1}^{n} h_\phi(X, H \odot a_i)$, i.e., predicting independently under each background state and averaging.
    - Design Motivation: Avoids overly simplified biological priors (e.g., directly removing certain signals), instead learning diverse representations of background states in a data-driven manner.

4. **Diversity Constraint**:

    - Function: Prevents weight vectors from collapsing to a single mode.
    - Mechanism: A uniform loss $\mathcal{L}_3 = \log(\sum_{i,j} \exp(2t \cdot \tilde{a}_i^T \tilde{a}_j - 2t))$ penalizes similarity among weight vectors.
    - Design Motivation: Ensures diverse background state representations are learned, with each emphasizing different signal combinations (e.g., one emphasizing chromatin accessibility, another emphasizing 3D organization).

### Loss & Training

Total loss $\mathcal{L} = \mathcal{L}_1 + \alpha \mathcal{L}_2 + \beta \mathcal{L}_3$:
- $\mathcal{L}_1$: Standard prediction loss (Huber loss), predicting $Y$ directly from $H$.
- $\mathcal{L}_2$: Intervention regularization, Huber loss between the average prediction across background states and $Y$.
- $\mathcal{L}_3$: Diversity loss, ensuring weight vectors remain distinct.

The signal encoder $g_\theta$ is a simple linear layer; the confounder encoder $g_\omega$ is a lightweight 1D-CNN, adding only 11K parameters.

## Key Experimental Results

### Main Results

CAGE value prediction is evaluated on two cell lines, K562 and GM12878, against 9 baseline methods:

| Method | K562 MSE↓ | K562 Pearson↑ | GM12878 MSE↓ | GM12878 Pearson↑ |
|------|------|------|------|------|
| Enformer (200k) | 0.2920 | 0.7961 | 0.2889 | 0.8327 |
| Caduceus (200k) | 0.2197 | 0.8475 | 0.2124 | 0.8819 |
| Seq2Exp-soft (200k) | 0.1856 | 0.8723 | 0.1873 | 0.8951 |
| **Prism (2k)** | **0.1789** | **0.8751** | **0.1759** | **0.9016** |

Prism with 2k sequences outperforms all methods using 200k sequences.

### Ablation Study

| Configuration | K562 MSE↓ | Note |
|------|---------|------|
| $n=0$ (no deconfounding) | 0.1863 | Degenerates to standard training |
| $n=1$ | 0.1891 | Single background state insufficient |
| $n=2$ (default) | 0.1789 | Balance of performance and efficiency |
| $n=4$ (optimal) | **0.1762** | More states improve with diminishing returns |
| $\alpha=0$ (no intervention loss) | Degrades | Validates necessity of intervention regularization |
| $\beta$ sensitivity | Stable across 0.1–1.0 | Diversity constraint is robust |

### Key Findings
- The "emperor's new clothes" of long-sequence modeling: Seq2Exp trains on 200k but uses only proximal 2.5k information; long sequences are not only unhelpful but may be harmful.
- H3K27ac (foreground signal) contributes the most and nearly matches full-signal performance alone; however, naive concatenation of background signals introduces spurious correlations.
- Learned weight vectors exhibit meaningful biological patterns: structural similarity across genes (e.g., "active" vs. "repressed" states) with diversity within genes.
- Prism adds only 11K parameters (vs. 500K+ for Seq2Exp), making it extremely lightweight.

## Highlights & Insights
- **Counterintuitive core finding**: In a field where everyone pursues longer sequences, this work rigorously demonstrates that shorter sequences with better signal integration are more effective. This "less is more" perspective is highly valuable.
- **Transfer of causal inference framework**: The deconfounding approach (backdoor adjustment from Qiang et al. 2022) is transferred from computer vision to genomics, bridging two seemingly unrelated fields. The backdoor adjustment methodology is generalizable to any multimodal fusion scenario exhibiting foreground/background confounding.
- **Minimal parameter overhead**: Only 11K additional parameters achieve state-of-the-art performance, indicating that the key lies in the modeling approach rather than model capacity—an insight with broad implications for the field.
- **Visualization validates causal hypothesis**: Visualization of weight vectors clearly reveals complementary "active/repressed" patterns, providing intuitive support for the causal framework.

## Limitations & Future Work
- Validation is limited to two cell lines (K562, GM12878), with no cross-tissue or cross-species generalization assessment.
- The definition of confounding variables as "background chromatin states" is abstract, and direct biological validation is lacking—do the learned weights truly correspond to known chromatin states (e.g., the 15 states in ChromHMM)?
- The hypothesis that proximal epigenomic signals reflect distal regulation via chromatin loops is plausible but lacks direct experimental evidence.
- The number of background states $n$ still requires manual tuning, and the optimal value (4) differs from the default (2).
- Whether these conclusions remain valid when stronger long-sequence models emerge (e.g., future SSM improvements) is an open question.

## Related Work & Insights
- **vs. Seq2Exp**: Seq2Exp uses learnable masks on 200k sequences to identify important regions with naive signal concatenation. This work demonstrates that Seq2Exp effectively uses only proximal information and that signal concatenation introduces confounding. Prism is more concise and effective with short sequences and deconfounding.
- **vs. EPInformer**: EPInformer, based on the ABC model, uses DNase-seq peaks to locate candidate regulatory regions. This paper argues that DNase-seq is a background signal and its direct use may introduce confounding.
- **vs. Enformer**: Enformer's 128× downsampling discards single-nucleotide resolution, making it inferior to single-base-resolution methods on specialized gene expression tasks.
- **Generalizability of the causal approach**: The backdoor adjustment deconfounding strategy is transferable to other multimodal fusion scenarios—whenever different modalities exhibit foreground/background role differences, analogous confounding problems may arise.

## Rating
- Novelty: ⭐⭐⭐⭐ The application of causal deconfounding to genomics is novel, though the methodology originates from computer vision.
- Experimental Thoroughness: ⭐⭐⭐⭐ Ablation and sensitivity analyses are thorough, but only two cell lines are evaluated.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is developed in a well-structured, layered manner; experimental design is elegant; the counterintuitive core finding is rigorously argued.
- Value: ⭐⭐⭐⭐ Offers important insights to the gene expression prediction field; the method is concise and practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Not All Deepfakes Are Created Equal: Triaging Audio Forgeries for Robust Deepfake Singer Identification](../../NeurIPS2025/ai_safety/not_all_deepfakes_are_created_equal_triaging_audio_forgeries_for_robust_deepfake.md)
- [\[CVPR 2026\] FedAFD: Multimodal Federated Learning via Adversarial Fusion and Distillation](../../CVPR2026/ai_safety/fedafd_multimodal_federated_learning_via_adversarial_fusion_and_distillation.md)
- [\[ICLR 2026\] Time Is All It Takes: Spike-Retiming Attacks on Event-Driven Spiking Neural Networks](time_is_all_it_takes_spike-retiming_attacks_on_event-driven_spiking_neural_netwo.md)
- [\[CVPR 2026\] Computation and Communication Efficient Federated Unlearning via On-server Gradient Conflict Mitigation and Expression](../../CVPR2026/ai_safety/computation_and_communication_efficient_federated_unlearning_via_on-server_gradi.md)
- [\[AAAI 2026\] Breaking the Dyadic Barrier: Rethinking Fairness in Link Prediction Beyond Demographic Parity](../../AAAI2026/ai_safety/breaking_the_dyadic_barrier_rethinking_fairness_in_link_prediction_beyond_demogr.md)

</div>

<!-- RELATED:END -->
