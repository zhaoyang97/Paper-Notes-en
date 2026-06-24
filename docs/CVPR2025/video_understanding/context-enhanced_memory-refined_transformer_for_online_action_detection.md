---
title: >-
  [Paper Note] Context-Enhanced Memory-Refined Transformer for Online Action Detection
description: >-
  [CVPR 2025][Video Understanding][Online Action Detection] This paper reveals the training-inference inconsistency problem in existing online action detection (OAD) methods—where unbalanced context exposure of short-term memory frames and non-causal information leakage introduced by pseudo-futures bias learning toward intermediate frames—and proposes CMeRT to address this issue through a near-past context-enhanced encoder and a near-future-based memory refinement decoder…
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Online Action Detection"
  - "Action Anticipation"
  - "Transformer"
  - "Memory Mechanism"
  - "Training-Inference Discrepancy"
date: 2026-05-08
content_hash: 36abbf8a7050d1a8
---

# Context-Enhanced Memory-Refined Transformer for Online Action Detection

**Conference**: CVPR 2025  
**arXiv**: [2503.18359](https://arxiv.org/abs/2503.18359)  
**Code**: [GitHub](https://github.com/pangzhan27/CMeRT)  
**Area**: Video Understanding  
**Keywords**: Online Action Detection, Action Anticipation, Transformer, Memory Mechanism, Training-Inference Discrepancy

## TL;DR
This paper reveals the training-inference inconsistency problem in existing online action detection (OAD) methods—where unbalanced context exposure of short-term memory frames and non-causal information leakage introduced by pseudo-futures bias learning toward intermediate frames—and proposes CMeRT to address this issue through a near-past context-enhanced encoder and a near-future-based memory refinement decoder, achieving state-of-the-art performance on THUMOS'14, CrossTask, and EK100.

## Background & Motivation
Online action detection (OAD) requires real-time action recognition in video streams based solely on past observations, serving as a foundation for applications like autonomous driving, surveillance, and AR assistants. State-of-the-art OAD methods divide historical frames into long-term and short-term memory, and compensate for missing future context by predicting a pseudo-future. During training, a causal mask is used to make all frames in the short-term memory serve as training samples, whereas only the latest frame is used during inference.

The key challenge lies in the **training-inference discrepancy**, which leads to two types of biases:

**Unbalanced Context Exposure**: The causal mask leaves early frames in short-term memory (e.g., $t_s$) with virtually no immediate context, whereas the latest frame ($t$) enjoys full context. This leads to poor representation quality (high loss) for early frames; however, incorporating these low-quality samples during training undermines the classifier's capability to predict the latest frame.

**Non-Causal Leakage**: Methods like MAT generate a "pseudo-future" based on the complete short-term memory to enhance detection, but this allows intermediate frames to indirectly access subsequent frames via "future $\rightarrow$ short-term $\rightarrow$ future" pathways, violating causality. This biases training toward intermediate frames (exhibiting a valley-shaped loss curve), detrimental to the learning of the latest frame.

This paper proposes CMeRT, which (1) complements early frames with immediate information via near-past context, and (2) generates the near-future solely from long-term memory (rather than short-term memory) to avoid non-causal leakage.

## Method

### Overall Architecture
CMeRT adopts an encoder-decoder architecture operating on five context partitions: long-term memory $M_L$, short-term memory $M_S$, anticipation query $Q_A$, near-past $M_C$, and near-future $M_F$. The encoder compresses long-term memory and enhances short-term memory encoding using the near-past context; the decoder generates the near-future from compressed long-term memory and refines the short-term memory. All modules are built based on a unified Transformer Decoder Unit (TDU).

### Key Designs
1. **Context-Enhanced Encoder**:

    - **Function**: Complements early frames in short-term memory with immediate past context to alleviate the unbalanced exposure issue.
    - **Mechanism**: Extract the near-past memory $M_C = \{f_i\}_{i=t_s-T_c}^{t_s-1}$ (length $T_c \ll T_l$), concatenate it before the short-term memory, and encode them together with the anticipation query via a causally masked TDU: $M_{SA} = \text{TDU}(M_C \| M_S \| Q_A, \hat{M}_L \| M_S \| Q_A, \hat{M}_L \| M_S \| Q_A, G)_{[T_c:T_c+T_s+T_a]}$.
    - **Design Motivation**: Although the near-past $M_C$ overlaps with the long-term memory $M_L$, the long-term memory loses fine-grained details during compression, whereas $M_C$ preserves these details for early frames; after encoding, $M_C$ is discarded, leaving only the enhanced short-term memory and anticipation.

2. **Near-Future Generator**:

    - **Function**: Generates near-future context from compressed long-term memory to provide future information for all short-term frames.
    - **Mechanism**: $M_F = \text{TDU}(Q_F, \hat{M}_L, \hat{M}_L, \text{None})$, retrieving useful information from $\hat{M}_L$ using a learnable query $Q_F$ (length $T_f$).
    - **Design Motivation**: The key improvement is avoiding the use of short-term memory to generate the near-future (unlike MAT), relying solely on the compressed long-term memory instead, which fundamentally eliminates the non-causal leakage problem.

3. **Memory Refinement**:

    - **Function**: Refines the encoded short-term memory with near-future context to boost detection and anticipation performance.
    - **Mechanism**: $\hat{M}_{SA} = \text{TDU}(M_{SA}, \hat{M}_L \| M_{SA} \| M_F, \hat{M}_L \| M_{SA} \| M_F, G)$.
    - **Design Motivation**: Near-future information helps disambiguate ongoing actions; meanwhile, because $M_F$ originates from compressed long-term memory rather than short-term memory, no causal leakage is introduced.

### Loss & Training
The training loss is defined as: $\mathcal{L} = \mathcal{L}_{SA}^1 + \lambda_1 \mathcal{L}_{SA}^0 + \lambda_2 \mathcal{L}_F$

where $\mathcal{L}_{SA}^0$ and $\mathcal{L}_{SA}^1$ denote the cross-entropy losses for the encoder output and the refined output respectively, and $\mathcal{L}_F$ is the cross-entropy loss for the near-future generation. A shared classifier is utilized. The trade-off coefficients are set to $\lambda_1 = 0.2$ and $\lambda_2 = 0.5$. The optimization employs the Adam optimizer with cosine annealing and warmup. Training sampling strategies include sliding-window (THUMOS and CrossTask) and event-centric sampling (EK100). Inference utilizes a sliding window with step size 1 to simulate online streaming scenarios.

## Key Experimental Results

### Main Results

| Dataset | Metric | CMeRT | MAT (Prev. SOTA) | Gain |
|--------|------|------|----------|------|
| THUMOS'14 | mAP (Detection) | 73.2 | 71.6 | +1.6 |
| CrossTask | mAP (Detection) | 35.9 | 33.9 | +2.0 |
| EK100 | Top-5 Recall (Action) | 27.6 | 26.3 | +1.3 |
| THUMOS'14 | mAP (Anticipation Avg) | 59.5 | 58.2 | +1.3 |
| EK100 | Top-5 Recall (Action Anticip.) | 19.8 | 19.5 | +0.3 |

### Ablation Study

| Configuration | TH'14 mAP | CrossTask mAP | EK100 Action | Description |
|------|---------|------|------|------|
| W/o CE, w/o MR | 71.5 | 33.4 | 26.3 | Baseline |
| +MR (Near-Future Refinement) | 73.0 | 34.8 | 27.1 | +1.5 / +1.4 / +0.8 |
| +CE (Near-Past Enhancement) | 71.9 | 33.9 | 26.6 | +0.4 / +0.5 / +0.3 |
| +CE+MR (Full CMeRT) | 73.2 | 35.9 | 27.6 | Optimal combination |

| Near-Past Length (s) | CrossTask | Near-Past Length (s) | TH'14 | EK100 Action |
|------|---------|------|------|------|
| 5 | 35.1 | 0.5 | **73.2** | 27.2 |
| **10** | **35.9** | 1 | 72.8 | 27.3 |
| 15 | 35.6 | 2 | 72.7 | **27.6** |

### Key Findings
- Memory refinement (MR) contributes more than context enhancement (CE), with the former bringing gains of +1.5%, +1.4%, and +0.8% across the three datasets, respectively.
- The optimal length of near-past context varies with dataset complexity: the simpler THUMOS requires only 0.5s, while the more complex CrossTask and EK100 need longer durations.
- Naive solutions (e.g., MAT-rw weighting the latest frame, or MAT-stream training solely with the latest frame) yield limited improvements or even show severe performance degradation.
- Replacing traditional features with DinoV2 earns CMeRT a 76.4% mAP on THUMOS, further validating its compatibility with stronger features.
- Efficiency surpasses MAT: fewer parameters (94.5M vs 107.4M) and a higher FPS (126.6 vs 102.0).

## Highlights & Insights
- **Diagnosis of Training-Inference Inconsistency**: Precising the sources of the two types of biases through frame-level loss curve visualization—a diagnostic methodology that possesses transfer value.
- **Ingenious Introduction of Near-Past Context**: Rather than simply extending short-term memory, it supplements immediate context for early frames during training without increasing inference costs.
- **Leakage-Free Near-Future Generation**: Generates the near-future from compressed long-term memory instead of short-term memory, systematically eliminating non-causal leakage at its source.
- **Unified Detection and Anticipation**: Handles both online detection and action anticipation within a single framework, achieving mutual benefits through a shared classifier and joint training.
- **New Evaluation Protocol**: Standardizes the OAD field by introducing stronger features (DinoV2), event-centric metrics, and a new benchmark (CrossTask).

## Limitations & Future Work
- It still relies on pre-extracted frame features (e.g., ResNet-50, I3D), and end-to-end training remains unexplored.
- The lengths of near-past and near-future contexts must be manually tuned for each dataset.
- The gain in anticipation on EK100 is relatively incremental (+0.3%), likely due to the extremely long-tailed distribution of fine-grained actions in this dataset.
- The near-future generated from compressed long-term memory may discard certain temporal details, suffering from an information loss compared to predictions based on short-term memory.
- Learnable frame sampling weight strategies to replace the uniform processing of all short-term frames have not been explored.

## Related Work & Insights
- **vs LSTR**: LSTR pioneered the long/short-term memory framework; CMeRT introduces near-past/near-future contexts on top of it, yielding significant performance gains.
- **vs TeSTra**: TeSTra enhances streaming efficiency but fails to solve the unbalanced context issue; CMeRT resolves this limitation by supplying the near-past.
- **vs MAT**: MAT introduces conditional recurrent interactions to unify detection and anticipation, but its CCI causes non-causal leakage; CMeRT's memory refinement successfully avoids this issue.
- **vs JOAAD**: JOAAD is a recent state-of-the-art (72.6% on TH'14); CMeRT outperforms it at 73.2% while maintaining a simpler methodology.

## Rating
- Novelty: ⭐⭐⭐⭐ The diagnosis of the training-inference discrepancy is profound and novel, and the designs for near-past/near-future are systematic, though the overall framework is an incremental improvement over existing memory-based methods.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covering three datasets, detection and anticipation tasks, comprehensive ablations (lengths, distances, feature types, efficiency), along with new benchmarks and protocols.
- Writing Quality: ⭐⭐⭐⭐⭐ Excellent visual analysis in the problem diagnosis section, rigorous logical derivation, and a natural transition from observations to the proposed method.
- Value: ⭐⭐⭐⭐ Provides a systematic solution to the training-inference consistency issue in the OAD domain and advances the update of evaluation protocols.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Online Temporal Action Localization with Memory-Augmented Transformer](../../ECCV2024/video_understanding/online_temporal_action_localization_with_memory-augmented_transformer.md)
- [\[ECCV 2024\] HAT: History-Augmented Anchor Transformer for Online Temporal Action Localization](../../ECCV2024/video_understanding/hat_history-augmented_anchor_transformer_for_online_temporal_action_localization.md)
- [\[ECCV 2024\] Bayesian Evidential Deep Learning for Online Action Detection](../../ECCV2024/video_understanding/bayesian_evidential_deep_learning_for_online_action_detection.md)
- [\[CVPR 2025\] Object-Shot Enhanced Grounding Network for Egocentric Video](object-shot_enhanced_grounding_network_for_egocentric_video.md)
- [\[CVPR 2025\] HierarQ: Task-Aware Hierarchical Q-Former for Enhanced Video Understanding](hierarq_task-aware_hierarchical_q-former_for_enhanced_video_understanding.md)

</div>

<!-- RELATED:END -->
