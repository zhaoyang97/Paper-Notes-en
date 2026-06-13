---
title: >-
  [Paper Note] XLSR-MamBo: Scaling the Hybrid Mamba-Attention Backbone for Audio Deepfake Detection
description: >-
  [ACL 2026][Audio & Speech][Audio Deepfake Detection] The XLSR-MamBo framework is proposed to systematically explore four topology designs and various SSM variants (Mamba2, Hydra…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "Audio Deepfake Detection"
  - "Mamba"
  - "Hybrid Architecture"
  - "State Space Models"
  - "XLSR"
date: 2026-05-08
content_hash: 6869e39c2e2e270a
---

# XLSR-MamBo: Scaling the Hybrid Mamba-Attention Backbone for Audio Deepfake Detection

**Conference**: ACL 2026 Findings  
**arXiv**: [2601.02944](https://arxiv.org/abs/2601.02944)  
**Code**: [GitHub](https://github.com/saki-ciallo/XLSR-MamBo)  
**Area**: AI Security / Audio Deepfake Detection  
**Keywords**: Audio Deepfake Detection, Mamba, Hybrid Architecture, State Space Models, XLSR

## TL;DR
The XLSR-MamBo framework is proposed to systematically explore four topology designs and various SSM variants (Mamba2, Hydra, GDN) in Mamba-Attention hybrid architectures for audio deepfake detection. MamBo-3-Hydra achieves competitive performance across multiple benchmarks by leveraging Hydra’s native bidirectional modeling, while increasing backbone depth effectively mitigates performance instability found in shallow models.

## Background & Motivation

**Background**: Audio Deepfake Detection (ADD) has transitioned from handcrafted features to end-to-end architectures. The mainstream approach utilizes XLSR as a front-end feature extractor paired with attention-based classifiers like Conformer. Recently, State Space Models (SSMs) such as Mamba have gained attention due to their linear complexity.

**Limitations of Prior Work**: Purely causal SSMs are unidirectional and struggle to capture the content-based retrieval capabilities required for identifying global frequency-domain forgery traces. Existing bidirectional Mamba extensions rely on manually designed dual-branch strategies (e.g., forward-backward concatenation), which introduce structural redundancy. Furthermore, the quadratic complexity of Transformers limits efficiency for long sequences.

**Key Challenge**: SSMs excel at efficient temporal compression and capturing local high-frequency artifacts, while Attention is proficient in global correlation and content retrieval. Deepfake signals exhibit both local high-frequency artifacts and global spectral inconsistencies, making a single mechanism insufficient.

**Goal**: To systematically explore the optimal topological combination of SSM-Attention hybrid architectures in ADD and evaluate the impact of depth scaling on performance stability.

**Key Insight**: Inspired by hybrid LLM architectures like Jamba and Zamba, this work explores customized designs for the ADD task, specifically introducing Hydra (a native bidirectional SSM) to replace heuristic bidirectional strategies.

**Core Idea**: The complementarity between SSMs and Attention (temporal compression vs. content retrieval) is crucial in ADD. Hydra’s native bidirectional parameterization is more elegant than dual-branch strategies, and stacking SSM depth $N$ mitigates performance instability.

## Method

### Overall Architecture
Input raw audio features are extracted via XLSR ($X \in \mathbb{R}^{T \times 1024}$), followed by RMSNorm and linear projection to a hidden dimension $D=128$. The representation is encoded through $L=5$ MamBo hybrid layers, aggregated into an utterance-level representation via gated attention pooling, and finally passed through a linear layer to output binary classification logits.

### Key Designs

1.  **Four MamBo Topology Designs**:
    *   **Function**: Systematically explores different combinations of SSM and Attention.
    *   **Mechanism**: MamBo-1 (pure SSM replacing MHA), MamBo-2 (Mamer, SSM followed by MHA replacing FFN), MamBo-3 (interleaved Mamba and Transformer layers), and MamBo-4 (interleaved Mamba and Mamer layers). Each topology can be paired with different SSM variants (Mamba, Mamba2, Hydra, GDN).
    *   **Design Motivation**: MamBo-1/2 explore intra-layer SSM-Attention hybridization, while MamBo-3/4 explore inter-layer interleaving; different forgery trace types may require different processing patterns.

2.  **Hydra Native Bidirectional SSM**:
    *   **Function**: Captures non-causal global dependencies without dual-branch heuristics.
    *   **Mechanism**: Hydra parameterizes forward and backward scans as quasi-separable matrices involving lower-triangular (past information) and upper-triangular (future information) structures. The formula is:
        $$\text{shift}(SS(X)) + \text{flip}(\text{shift}(SS(\text{flip}(X)))) + DX$$
        This achieves native bidirectional processing within linear complexity.
    *   **Design Motivation**: Deepfake detection requires non-causal context as artifacts may be distributed throughout the audio. Hydra is more elegant and lacks the structural redundancy of manual bidirectional strategies.

3.  **Depth Scaling (Stacking N)**:
    *   **Function**: Enhances performance stability by increasing the number of stacked SSM layers.
    *   **Mechanism**: Introduces a stacking hyperparameter $N$, allowing $N$ consecutive SSM blocks within a single unit. Experiments found that $N=3$ yields optimal performance and stability, whereas shallow models ($N=1$) exhibit high performance variance.
    *   **Design Motivation**: Shallow SSMs lacks sufficient representational depth to consistently capture complex forgery traces.

### Loss & Training
FocalLoss is used to handle class imbalance. The AdamW optimizer is employed ($lr=10^{-5}$) with a 10% linear warmup and cosine decay. Training uses mixed precision (BF16/FP32) for up to 20 epochs with early stopping (patience=7). Models are trained on the ASVspoof 2019 LA training set and evaluated for cross-dataset generalization.

## Key Experimental Results

### Main Results

| Model | ASV21LA EER↓ | ASV21DF EER↓ | ITW EER↓ |
| :--- | :--- | :--- | :--- |
| XLSR-Conformer (Baseline) | ~1.0 | ~2.5 | ~5.0 |
| MamBo-1-Mamba (N=1) | 1.19 | 2.08 | 4.65 |
| MamBo-3-Hydra (N=3) | **Best** | **Competitive** | **Competitive** |
| RawBMamba | - | - | - |

### Ablation Study

| Configuration | ASV21LA | Description |
| :--- | :--- | :--- |
| MamBo-1 (Pure SSM) | Baseline | SSM replaces Attention |
| MamBo-2 (Mamer) | Slightly Better | Intra-layer hybrid is helpful |
| MamBo-3 (Interleaved) | **Best** | Inter-layer interleaving works best |
| N=1 vs N=3 | Variance↓ | Depth scaling significantly improves stability |

### Key Findings
*   MamBo-3 (interleaved Mamba-Transformer) performs best across most benchmarks, proving inter-layer interleaving is superior to intra-layer hybridization.
*   Hydra performs best within the MamBo-3 setup, as its native bidirectional modeling is more effective than Mamba's heuristic dual-branch approach.
*   Increasing SSM stacking depth $N$ from 1 to 3 significantly reduces performance variance; the instability of shallow models is a risk for practical deployment.
*   Robustness is maintained against Diffusion and Flow Matching synthesis methods on the DFADD dataset, demonstrating generalization capability.
*   GDN's delta rule memory management also performs well in certain scenarios.

## Highlights & Insights
*   The systematic topological exploration (4 designs × 4 SSM variants × different depths) provides a comprehensive design guide for SSM-Attention hybrid architectures in speech tasks. This methodology is transferable to other audio domains.
*   The advantages of Hydra’s native bidirectionality in ADD validate the hypothesis that "causal consistency violation" serves as a clue for forgery detection.
*   The insight "depth scaling mitigates shallow instability" is a practical engineering observation with direct implications for deployment.

## Limitations & Future Work
*   Training was conducted only on the ASVspoof 2019 LA training set, limiting training data diversity.
*   The model scale is relatively small ($D=128$, $L=5$); the performance of larger-scale models remains unexplored.
*   Performance on the ITW dataset still has room for improvement.
*   End-to-end training (unfreezing XLSR parameters) was not explored.
*   Future work could investigate more hybrid topologies and cross-lingual generalization.

## Related Work & Insights
*   **vs XLSR-Conformer**: A pure attention architecture; the proposed hybrid SSM improves both efficiency and performance.
*   **vs RawBMamba**: Employs manual bidirectional Mamba strategies; this work uses the more elegant Hydra native bidirectionality.
*   **vs Jamba/Samba**: Hybrid architectures in the LLM field; this work is the first to systematically apply this paradigm to ADD.

## Rating
*   Novelty: ⭐⭐⭐⭐ Systematic exploration of SSM-Attention hybrids in ADD; Hydra introduction is innovative.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across four topologies, four variants, multiple depths, and multiple datasets.
*   Writing Quality: ⭐⭐⭐⭐ Detailed background and clear experimental organization.
*   Value: ⭐⭐⭐⭐ Provides a systematic reference for architectural selection in the ADD field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] HCFD: A Benchmark for Audio Deepfake Detection in Healthcare](hcfd_a_benchmark_for_audio_deepfake_detection_in_healthcare.md)
- [\[ACL 2026\] An Exploration of Mamba for Speech Self-Supervised Models](an_exploration_of_mamba_for_speech_self-supervised_models.md)
- [\[ACL 2026\] Analyzing Reasoning Shifts in Audio Deepfake Detection under Adversarial Attacks: The Reasoning Tax versus Shield Bifurcation](analyzing_reasoning_shifts_in_audio_deepfake_detection_under_adversarial_attacks.md)
- [\[ACL 2026\] RTCFake: Speech Deepfake Detection in Real-Time Communication](rtcfake_speech_deepfake_detection_in_real-time_communication.md)
- [\[ACL 2026\] Semi-Supervised Diseased Detection from Speech Dialogues with Multi-Level Data Modeling](semi-supervised_diseased_detection_from_speech_dialogues_with_multi-level_data_m.md)

</div>

<!-- RELATED:END -->
