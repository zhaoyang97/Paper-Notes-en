---
title: >-
  [Paper Note] CrossNovo: Bidirectional Representations Augmented Autoregressive Biological Sequence Generation
description: >-
  [NeurIPS 2025][Computational Biology][de novo peptide sequencing] CrossNovo integrates autoregressive (AR) and non-autoregressive (NAR) decoders through a shared spectrum encoder, importance annealing…
tags:
  - "NeurIPS 2025"
  - "Computational Biology"
  - "de novo peptide sequencing"
  - "autoregressive"
  - "non-autoregressive"
  - "CTC"
  - "knowledge distillation"
date: 2026-05-08
content_hash: 78f4276d135acadb
---

# CrossNovo: Bidirectional Representations Augmented Autoregressive Biological Sequence Generation

**Conference**: NeurIPS 2025
**arXiv**: [2510.08169](https://arxiv.org/abs/2510.08169)  
**Code**: To be confirmed  
**Area**: Biological Sequence Generation / Proteomics
**Keywords**: de novo peptide sequencing, autoregressive, non-autoregressive, CTC, knowledge distillation

## TL;DR
CrossNovo integrates autoregressive (AR) and non-autoregressive (NAR) decoders through a shared spectrum encoder, importance annealing, and gradient-blocked knowledge distillation, enabling the bidirectional global understanding of NAR to augment AR sequence generation. On the 9-Species benchmark, it achieves amino acid accuracy of 0.811 (+2.6%) and peptide recall of 0.654 (+5.3%).

## Background & Motivation

**Background**: De novo peptide sequencing from mass spectrometry data is a core task in proteomics. AR models (e.g., ContraNovo) generate sequences step-by-step but lack a global view; NAR models (e.g., PrimeNovo) generate in parallel but suffer from training instability.

**Limitations of Prior Work**: AR models are prone to errors when distinguishing amino acids with similar masses (e.g., I/L, K/Q), as each step only attends to local history; NAR models have bidirectional attention but produce incoherent sequences and exhibit unstable training.

**Key Challenge**: The causal dependency of AR ensures generation coherence but limits access to global information; the strong bidirectional understanding of NAR cannot guarantee sequence consistency.

**Goal**: To introduce bidirectional representations from NAR to enhance the global understanding of AR, while preserving AR generation quality.

**Key Insight**: A shared encoder allows both decoders to learn complementary representations → NAR representations serve as additional context distilled into AR → gradient blocking prevents AR loss from corrupting NAR features.

**Core Idea**: The NAR decoder provides bidirectional global representations, which are distilled into the AR decoder via gradient-blocked cross-decoder attention, combined with a shared mass spectrum encoder for hybrid sequence generation.

## Method

### Overall Architecture
Input spectrum → **Shared Spectrum Encoder** (Transformer with domain-adapted sinusoidal encoding for m/z-intensity pairs) → encoded features $E^{(b)}$ → **AR Decoder** (causal self-attention + cross-attention to $E^{(b)}$ + prefix/suffix mass constraints) + **NAR Decoder** (non-causal self-attention + cross-attention to $E^{(b)}$, CTC loss) → Two-stage training: multi-task learning (importance annealing) → knowledge distillation (gradient blocking).

### Key Designs

1. **Shared Spectrum Encoder + Mass Constraints**:

    - Function: Encodes mass spectrum data into shared features and injects biochemical priors into AR.
    - Mechanism: m/z-intensity pairs are encoded using domain-adapted sinusoidal encoding (adapted to the value range), followed by $b$-layer Transformer self-attention. At each AR step, prefix/suffix mass constraints (cumulative mass of generated tokens and remaining mass) are additionally provided.
    - Design Motivation: The shared encoder ensures both decoders observe a consistent spectral representation; mass constraints inject biochemical knowledge into AR.

2. **Importance Annealing Multi-task Training (Stage 1)**:

    - Function: Jointly trains AR (CE loss) and NAR (CTC loss) with dynamically adjusted weights.
    - Mechanism: $\mathcal{L} = \lambda_{AT} \mathcal{L}_{AT} + (1-\lambda_{AT}) \mathcal{L}_{NAT}$, where $\lambda_{AT}(i) = i/T_{total}$ — early training emphasizes NAR (learning global patterns), while later training emphasizes AR (fine-grained sequence generation guidance).
    - Design Motivation: NAR learns global bidirectional understanding faster (parallel generation); fine-grained AR guidance requires NAR to have already learned good representations.

3. **Gradient-Blocked Knowledge Distillation (Stage 2)**:

    - Function: Distills bidirectional NAR representations into AR.
    - Mechanism: The encoder and NAR are frozen; only AR is fine-tuned. The AR cross-attention attends jointly to $[\mathbb{GB}(V^{(L')}) \oplus E^{(b)}]$ — the last-layer NAR features (with gradient blocking $\mathbb{GB}$ = detach) concatenated with encoder features.
    - Design Motivation: Gradient blocking is critical — ablation studies show that removing it causes gradient explosion. NAR's bidirectional features provide AR with a global perspective without being corrupted by AR loss in return.

### Loss & Training
- AR: Cross-entropy $\mathcal{L}_{AT} = -\sum_t \log p(a_t | a_{<t}, \mathcal{S})$
- NAR: CTC loss (handling variable-length alignment)
- 8×A100, AdamW lr=5e-4, cosine annealing, optimal beam size=5

## Key Experimental Results

### Main Results

| Benchmark | Metric | CrossNovo | ContraNovo | PrimeNovo (NAR) |
|-----------|--------|-----------|------------|-----------------|
| 9-Species-v1 | AA Accuracy | **0.811** | 0.785 | 0.788 |
| 9-Species-v1 | Peptide Recall | **0.654** | 0.621 | 0.638 |
| 9-Species-v2 | AA Accuracy | **0.906** | 0.882 | 0.891 |
| 9-Species-v2 | Peptide Recall | **0.786** | 0.752 | 0.777 |

### Ablation Study

| Configuration | AA Accuracy | Peptide Recall | Note |
|---------------|-------------|----------------|------|
| w/o gradient blocking | — | — | Gradient explosion |
| w/o shared encoder | 0.698 | 0.546 | Severe degradation |
| Full model | **0.811** | **0.654** | Best |

### Key Findings
- Consistently outperforms all baselines across all 9 species, with +9% improvement on species where AR excels (Human, Mouse).
- Achieves consistently superior disambiguation accuracy for amino acids with similar masses (I/L, K/Q).
- Zero-shot antibody sequencing yields +5% peptide recall, demonstrating generalization capability.
- Gradient blocking is essential for training stability — its removal causes gradient explosion.

## Highlights & Insights
- **The AR+NAR fusion knowledge distillation paradigm is broadly applicable**: The idea of NAR providing global representations distilled into AR via gradient blocking is transferable to other sequence generation tasks (e.g., speech recognition, machine translation).
- **Importance annealing is a simple yet effective curriculum strategy**: Allowing global understanding to mature early and refining sequence generation later avoids complex training schedules.

## Limitations & Future Work
- Two-stage training increases pipeline complexity.
- Beam search introduces inference latency (beam=5 is optimal but slower).
- Validation is limited to peptide sequencing; other biological sequences (e.g., RNA) remain untested.
- The NAR decoder is unused at inference time (used only for distillation), resulting in low resource utilization.

## Related Work & Insights
- **vs ContraNovo**: A purely AR model lacking a global view; CrossNovo compensates via NAR distillation.
- **vs PrimeNovo**: A purely NAR model with poor generation coherence; CrossNovo preserves AR sequence consistency.
- **vs CTC-based methods**: CTC is used for NAR training rather than inference, avoiding the decoding degradation issues associated with CTC.

## Rating
- Novelty: ⭐⭐⭐⭐ The AR+NAR fusion with gradient-blocked distillation is a novel design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple benchmarks + ablation studies + downstream antibody task.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear and well-structured.
- Value: ⭐⭐⭐⭐ Provides a generalizable paradigm for AR/NAR fusion in sequence generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Controllable Sequence Editing for Biological and Clinical Trajectories](../../ICLR2026/computational_biology/controllable_sequence_editing_for_biological_and_clinical_trajectories.md)
- [\[ICML 2026\] Protein Autoregressive Modeling via Multiscale Structure Generation](../../ICML2026/computational_biology/protein_autoregressive_modeling_via_multiscale_structure_generation.md)
- [\[NeurIPS 2025\] Learning Repetition-Invariant Representations for Polymer Informatics](learning_repetition-invariant_representations_for_polymer_informatics.md)
- [\[NeurIPS 2025\] Diffusion Generative Modeling on Lie Group Representations](diffusion_generative_modeling_on_lie_group_representations.md)
- [\[NeurIPS 2025\] GraphFLA: Augmenting Biological Fitness Prediction Benchmarks with Landscape Features](augmenting_biological_fitness_prediction_benchmarks_with_landscapes_features_fro.md)

</div>

<!-- RELATED:END -->
