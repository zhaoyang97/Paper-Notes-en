---
title: >-
  [Paper Note] NeuroGaze-Distill: Brain-informed Distillation and Depression-Inspired Geometric Priors for Robust Facial Emotion Recognition
description: >-
  [ICLR 2026][Human Understanding][facial emotion recognition] This paper proposes NeuroGaze-Distill, a cross-modal distillation framework that extracts static Valence-Arousal prototypes from an EEG-trained teacher model and injects them into a purely visual student model via Proto-KD and depression-inspired geometric priors (D-Geo), improving cross-dataset robustness for facial expression recognition without requiring paired EEG-face data.
tags:
  - ICLR 2026
  - Human Understanding
  - facial emotion recognition
  - knowledge distillation
  - EEG prototypes
  - depression-inspired prior
  - cross-dataset robustness
date: 2026-05-08
content_hash: 3b0e405a20b6046a
---

# NeuroGaze-Distill: Brain-informed Distillation and Depression-Inspired Geometric Priors for Robust Facial Emotion Recognition

**Conference**: ICLR 2026
**arXiv**: [2509.11916](https://arxiv.org/abs/2509.11916)
**Code**: [GitHub](https://github.com/Lixeeone/NeuroGaze-Distill) (minimal reproduction repository)
**Area**: Human Understanding
**Keywords**: facial emotion recognition, knowledge distillation, EEG prototypes, depression-inspired prior, cross-dataset robustness

## TL;DR

This paper proposes NeuroGaze-Distill, a cross-modal distillation framework that extracts static Valence-Arousal prototypes from an EEG-trained teacher model and injects them into a purely visual student model via Proto-KD and depression-inspired geometric priors (D-Geo), improving cross-dataset robustness for facial expression recognition without requiring paired EEG-face data.

## Background & Motivation

1. Appearance-based facial expression recognition (FER) models perform well within-domain but generalize poorly across datasets — differences in demographics, capture conditions, and annotation conventions induce severe distribution shift.
2. Facial appearance is an indirect and biased proxy for emotion, whereas physiological signals (e.g., EEG) encode emotional dynamics that are decoupled from appearance.
3. Collecting large-scale paired EEG-face data is impractical and incompatible with deploying purely visual systems.
4. Core idea: learn static neuro-informed prototypes in the continuous Valence-Arousal (V/A) space and distill them into a pure image-based student model.
5. Inspired by affective neuroscience: depression-related research has observed anhedonia — attenuated emotional responses in high-valence regions.
6. This observation is encoded as a lightweight geometric regularization term (D-Geo) that softly shapes the geometry of the embedding space.

## Method

### Overall Architecture

Four-stage pipeline:
1. Train a teacher network on EEG data (DREAMER + MAHNOB-HCI) to regress V/A values.
2. Aggregate teacher validation-set embeddings into a 5×5 V/A prototype grid (25 static prototypes), then freeze and reuse.
3. Train a ResNet-18/50 student model on FERPlus with a joint objective: CE + KD + Proto-KD + D-Geo.
4. At inference, only the visual model is required — no EEG or other non-visual signals are needed.

### Key Designs

**Design 1: Static Neuro-informed Prototypes**
- **Function**: Construct 25 V/A prototypes from the EEG teacher's validation-set embeddings.
- **Mechanism**: Discretize the V/A space into a 5×5 grid (bin centers from −0.8 to 0.8) and average L2-normalized penultimate-layer teacher features within each bin. Empty bins are filled with the mean of the nearest non-empty bin.
- **Design Motivation**: The 5×5 grid balances coverage and statistical stability — denser grids (e.g., 7×7) result in sparse bins and collapse. Prototypes are constructed once, frozen, and reused, requiring no paired EEG-face data.

**Design 2: Prototype Knowledge Distillation (Proto-KD, Cosine)**
- **Function**: Align student features with the static prototypes.
- **Mechanism**: Compute cosine similarities $s_k = \cos(f(x), p_k)$ between the student feature $f(x)$ and all prototypes $p_k$, construct a soft distribution $q^{stu} = \text{softmax}(s/\tau)$ ($\tau=0.90$), and minimize $D_{KL}(q^{pro} \| q^{stu})$.
- **Design Motivation**: Enables the visual student to implicitly inherit the affective spatial structure captured by the EEG teacher without requiring paired training data.

**Design 3: Depression-Inspired Geometric Prior (D-Geo)**
- **Function**: Regularize the geometry of the embedding space.
- **Mechanism**: (1) Apply an intra-class variance upper-bound cap to high-valence categories (happiness, surprise); (2) globally encourage inter-class margins to maintain separability. A cosine ramp delays activation (epochs 20→60), and the regularization weight is kept small.
- **Design Motivation**: Motivated by anhedonia findings in depression research — more compact representations of high-valence regions may improve robustness. t-SNE visualizations confirm that D-Geo yields more compact high-valence clusters while preserving separation of low-valence classes.

### Loss & Training

Overall loss:
$$\mathcal{L} = \underbrace{\mathcal{L}_{CE}}_{\text{label smoothing 0.055 + class weights}} + \lambda_{kd} \underbrace{\mathcal{L}_{KD}}_{\text{MSE/KL, T=5.0}} + \lambda_{proto} \underbrace{D_{KL}(q^{pro} \| q^{stu})}_{\text{Proto-KD, } \tau=0.90} + \lambda_{geo} \underbrace{\mathcal{L}_{D\text{-}Geo}}_{\text{delayed activation}}$$

Training details: AdamW optimizer, cosine learning rate decay, base LR $2 \times 10^{-4}$, weight decay 0.05, batch size 128, mixed precision (AMP), gradient clipping 1.0. Student EMA is disabled (Mean-Teacher-style EMA degrades performance on this task).

## Key Experimental Results

### Main Results

FERPlus validation set ablation (8-way):

| Variant | Acc (%) | Macro-F1 (%) | bACC (%) |
|---------|---------|--------------|----------|
| B0: CE only | 78.22 | 51.29 | 49.77 |
| B1: +KD | 82.31 | 63.56 | 59.28 |
| B2: +KD+Proto | 81.48 | 64.21 | 60.30 |
| **B3: Full (+D-Geo)** | **83.06** | **64.74** | **59.90** |
| Full (T=1) | 83.66 | 65.39 | 61.32 |

Cross-dataset evaluation (A3_full, present-only):

| Dataset | Acc (%) | Macro-F1 (%) | bACC (%) |
|---------|---------|--------------|----------|
| FERPlus (valid) | 83.06 | 64.74 | 59.90 |
| AffectNet-mini | 76.30 | 75.60 | 75.77 |
| CK+ | 64.93 | 49.33 | 52.46 |

### Ablation Study

Component contribution analysis:

| Component | Key Effect |
|-----------|-----------|
| KD (B0→B1) | Large Macro-F1 gain (+12.27); accelerates early convergence |
| Proto-KD (B1→B2) | Improves class-balanced bACC (+1.02); stabilizes prototype alignment |
| D-Geo (B2→B3) | Maintains best Macro-F1; shapes more compact high-valence clusters |

Hyperparameter sensitivity:

| Parameter | Optimal Value | Notes |
|-----------|--------------|-------|
| $\lambda_{proto}$ | 0.12 | Stable in range 0.10–0.15 |
| D-Geo activation | epoch 20→60 | Starting from epoch 0 harms early separability |
| V/A grid size | 5×5 | 7×7 leads to sparse bin collapse |

### Key Findings

1. KD is the single largest contributing factor (Macro-F1 +12.27); Proto-KD and D-Geo provide complementary late-stage improvements.
2. Proto-KD reduces anger/sadness confusion (on AffectNet-mini); D-Geo further improves high-valence cluster purity.
3. t-SNE visualizations show progressive improvement from B0 to B3: increasing inter-class margins and more compact high-valence clusters.
4. The present-only Macro-F1 on AffectNet-mini (75.60%) substantially outperforms the 8-way result on CK+ (39.23%), highlighting label-set mismatch as a critical issue in cross-dataset evaluation.
5. Student EMA (Mean-Teacher style) degrades performance on this task and is disabled.
6. All artifacts (prototype bank, checkpoints, metrics JSON) are verified with SHA-256 hashes, ensuring strong reproducibility.

## Highlights & Insights

1. **Novel cross-modal distillation pathway**: EEG → V/A prototypes → visual student, elegantly circumventing the need for paired data.
2. **D-Geo, the depression-inspired geometric prior**, offers a highly original regularization strategy — encoding neuroscientific insights as embedding-space constraints.
3. **The 5×5 prototype design is simple and effective**: constructed once, frozen, and reused with no additional inference overhead.
4. **The present-only evaluation protocol** fairly addresses label-set mismatch and is worth broader adoption in the FER community.
5. The responsible research statement is thorough: it explicitly clarifies that D-Geo is a non-diagnostic, non-clinical geometric bias, and provides a reuse checklist.

## Limitations & Future Work

1. **The FERPlus validation accuracy (83.06%) is not state-of-the-art in FER**, serving primarily as a proof of concept for the framework's generalizability.
2. The gains from D-Geo are modest (Macro-F1 from 64.21 to 64.74), potentially constrained by the small regularization weight.
3. The teacher network relies on limited EEG data (DREAMER + MAHNOB-HCI); prototype quality is bounded by the scale and diversity of the EEG corpus.
4. 8-way performance on CK+ remains low (Acc 55.86%), indicating that extreme domain-shift scenarios remain challenging.
5. Only ResNet-18/50 backbones are evaluated; stronger visual architectures (e.g., ViT, ConvNeXt) are not explored.
6. The "Gaze" component of "NeuroGaze" is disabled in the final experiments, creating a slight disconnect between the name and the actual method.

## Related Work & Insights

- **Knowledge Distillation** (Hinton et al., 2015): foundational logit-based soft-target distillation method.
- **Prototype Learning** (Snell et al., 2017; Li et al., 2020): prototypes represent class/region structure; Proto-KD extends this paradigm to the V/A space.
- **EEG-based Emotion Recognition**: most prior work is EEG-only or requires paired multimodal training; this paper is the first to realize unidirectional distillation (EEG → V/A → vision).
- Inspiration: the cross-modal prototype distillation paradigm may generalize to other "expensive signal → cheap inference" scenarios (e.g., fMRI → behavioral prediction).

## Rating

- **Novelty**: ⭐⭐⭐⭐ The EEG → V/A prototype distillation concept is novel, and D-Geo offers a distinctive, cross-disciplinary perspective.
- **Experimental Thoroughness**: ⭐⭐⭐ Ablations are sufficient, but comprehensive comparison with FER state-of-the-art methods is lacking, and data scale is limited.
- **Writing Quality**: ⭐⭐⭐⭐ Algorithmic pseudocode is clear; the responsible research statement is thorough and professional, though some descriptions are redundant.
- **Value**: ⭐⭐⭐ The framework concept is thought-provoking, but practical FER performance gains are limited; the effectiveness of D-Geo requires further validation.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Facial-R1: Aligning Reasoning and Recognition for Facial Emotion Analysis](../../AAAI2026/human_understanding/facial-r1_aligning_reasoning_and_recognition_for_facial_emotion_analysis.md)
- [\[CVPR 2026\] A Two-Stage Dual-Modality Model for Facial Expression Recognition](../../CVPR2026/human_understanding/a_two_stage_dual_modality_model_for_facial_expression_recognition.md)
- [\[ICLR 2026\] BAH Dataset for Ambivalence/Hesitancy Recognition in Videos for Digital Behaviour Analysis](bah_dataset_for_ambivalencehesitancy_recognition_in_videos_for_digital_behaviour.md)
- [\[ICCV 2025\] SynFER: Towards Boosting Facial Expression Recognition with Synthetic Data](../../ICCV2025/human_understanding/synfer_towards_boosting_facial_expression_recognition_with_synthetic_data.md)
- [\[ICLR 2026\] GaitSnippet: Gait Recognition Beyond Unordered Sets and Ordered Sequences](gaitsnippet_gait_recognition_beyond_unordered_sets_and_ordered_sequences.md)

<!-- RELATED:END -->
