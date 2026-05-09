---
title: >-
  [Paper Note] XLSR-MamBo: Scaling the Hybrid Mamba-Attention Backbone for Audio Deepfake Detection
description: >-
  [ACL 2026][AI Safety][Audio Deepfake Detection] This paper proposes the XLSR-MamBo framework, systematically exploring four topology designs and multiple SSM variants (Mamba2, Hydra, GDN) for Mamba-Attention hybrid architectures in audio deepfake detection, where MamBo-3-Hydra achieves competitive performance across multiple benchmarks through Hydra's native bidirectional modeling, and increasing backbone depth effectively mitigates shallow model instability.
tags:
  - ACL 2026
  - AI Safety
  - Audio Deepfake Detection
  - Mamba
  - Hybrid Architecture
  - State Space Models
  - XLSR
date: 2025-05-08
content_hash: f2267d71659a4ae3
---

# XLSR-MamBo: Scaling the Hybrid Mamba-Attention Backbone for Audio Deepfake Detection

**Conference**: ACL 2026  
**arXiv**: [2601.02944](https://arxiv.org/abs/2601.02944)  
**Code**: [GitHub](https://github.com/saki-ciallo/XLSR-MamBo)  
**Area**: AI Safety / Audio Deepfake Detection  
**Keywords**: Audio Deepfake Detection, Mamba, Hybrid Architecture, State Space Models, XLSR

## TL;DR

This paper proposes the XLSR-MamBo framework, systematically exploring four topology designs and multiple SSM variants (Mamba2, Hydra, GDN) for Mamba-Attention hybrid architectures in audio deepfake detection, where MamBo-3-Hydra achieves competitive performance across multiple benchmarks through Hydra's native bidirectional modeling, and increasing backbone depth effectively mitigates shallow model instability.

## Background & Motivation

**Background**: Audio deepfake detection (ADD) has evolved from hand-crafted features to end-to-end architectures. XLSR as a frontend feature extractor paired with attention-based classifiers like Conformer is the mainstream approach. Recently, state space models (SSMs) such as Mamba have attracted interest due to their linear complexity.

**Limitations of Prior Work**: Pure causal SSMs are unidirectional, making it difficult to capture the content-based retrieval capability needed for global frequency-domain forgery traces. Existing bidirectional Mamba extensions rely on heuristic dual-branch strategies (e.g., forward-backward concatenation), introducing structural redundancy. Transformer's quadratic complexity limits efficiency on long sequences.

**Key Challenge**: SSMs excel at efficient temporal compression and local high-frequency artifact capture, while Attention excels at global association and content retrieval — deepfake signals manifest simultaneously as local high-frequency artifacts and global spectral inconsistencies, and neither mechanism alone suffices.

**Goal**: Systematically explore the optimal topology combinations of SSM-Attention hybrid architectures in ADD and evaluate the impact of depth scaling on performance stability.

**Key Insight**: Inspired by hybrid architectures in LLMs such as Jamba and Zamba, but with customized exploration for the ADD task, particularly introducing Hydra (a native bidirectional SSM) to replace heuristic bidirectional strategies.

**Core Idea**: The complementarity of SSM and Attention (temporal compression vs. content retrieval) is particularly important in ADD; Hydra's native bidirectional parameterization is more elegant than dual-branch strategies; and increasing SSM stacking depth N mitigates performance instability.

## Method

### Overall Architecture

Raw audio input is processed by XLSR to extract features ($X \in \mathbb{R}^{T \times 1024}$), followed by RMSNorm + linear projection to hidden dimension D=128, encoded through L=5 MamBo hybrid layers, aggregated into utterance-level representation via gated attention pooling, and a linear layer outputs binary classification logits.

### Key Designs

1. **Four MamBo Topology Designs**:

    - Function: Systematically explore different combination patterns of SSM and Attention
    - Mechanism: MamBo-1 (pure SSM replacing MHA), MamBo-2 (Mamer: SSM followed by MHA replacing FFN), MamBo-3 (alternating Mamba layers and Transformer layers), MamBo-4 (alternating Mamba layers and Mamer layers). Each topology can be paired with different SSM variants (Mamba, Mamba2, Hydra, GDN)
    - Design Motivation: MamBo-1/2 explore intra-layer SSM-Attention mixing; MamBo-3/4 explore inter-layer alternation; different forgery trace types may require different processing approaches

2. **Hydra Native Bidirectional SSM**:

    - Function: Capture non-causal global dependencies without heuristic dual-branch design
    - Mechanism: Hydra parameterizes forward and backward scans as quasi-separable matrices, containing lower-triangular (past information) and upper-triangular (future information) structures. The formula is $\text{shift}(SS(X)) + \text{flip}(\text{shift}(SS(\text{flip}(X)))) + DX$, achieving native bidirectional processing within linear complexity
    - Design Motivation: Deepfake detection requires non-causal context (artifacts may be distributed throughout the audio); Hydra is more elegant than heuristic bidirectional strategies with no structural redundancy

3. **Depth Scaling (Stacking N)**:

    - Function: Improve performance stability by increasing SSM stacking depth
    - Mechanism: Introduces a stacking hyperparameter N, allowing consecutive stacking of N SSM blocks within a single unit. Experiments show that N=3 achieves optimal performance and stability, while shallow models (N=1) have high performance variance
    - Design Motivation: Shallow SSMs lack sufficient representational depth to consistently capture complex forgery traces

### Loss & Training

FocalLoss is used to handle class imbalance. AdamW optimizer ($lr=10^{-5}$) with 10% linear warmup + cosine decay. Mixed precision training (BF16/FP32), up to 20 epochs with early stopping patience=7. Trained on ASVspoof 2019 LA training set with cross-dataset evaluation for generalization.

## Key Experimental Results

### Main Results

| Model | ASV21LA EER↓ | ASV21DF EER↓ | ITW EER↓ |
|-------|-------------|-------------|----------|
| XLSR-Conformer (baseline) | ~1.0 | ~2.5 | ~5.0 |
| MamBo-1-Mamba (N=1) | 1.19 | 2.08 | 4.65 |
| MamBo-3-Hydra (N=3) | Best | Competitive | Competitive |
| RawBMamba | - | - | - |

### Ablation Study

| Config | ASV21LA | Note |
|--------|---------|------|
| MamBo-1 (pure SSM) | Baseline | SSM replaces Attention |
| MamBo-2 (Mamer) | Slightly better | Intra-layer mixing helps |
| MamBo-3 (alternating) | Best | Inter-layer alternation works best |
| N=1 vs N=3 | Variance↓ | Depth scaling significantly improves stability |

### Key Findings

- MamBo-3 (Mamba-Transformer alternation) performs best on most benchmarks, demonstrating that inter-layer alternation outperforms intra-layer mixing
- Hydra performs best in MamBo-3, as its native bidirectional modeling is more effective than Mamba's heuristic dual-branch approach
- Increasing SSM stacking depth N from 1 to 3 significantly reduces performance variance; shallow model instability is a practical deployment concern
- Robustness is maintained on the DFADD dataset against diffusion and flow-matching synthesis methods, demonstrating generalization ability
- GDN's delta rule memory management also performs well in certain scenarios

## Highlights & Insights

- The systematic topology exploration (4 designs × 4 SSM variants × different depths) provides a comprehensive design guide for SSM-Attention hybrid architectures in speech tasks. This methodology is transferable to other speech tasks
- Hydra's native bidirectional capability validates the hypothesis of "causal consistency violation" as a forgery detection cue in ADD
- "Depth scaling mitigates shallow instability" is a practical engineering insight with direct guidance for real-world deployment

## Limitations & Future Work

- Trained only on ASVspoof 2019 LA training set, limiting training data diversity
- Model scale is relatively small (D=128, L=5); larger-scale model performance is unexplored
- Performance on the ITW dataset still has room for improvement
- End-to-end training was not explored (XLSR parameters are frozen)
- Future work could explore more hybrid topologies and cross-lingual generalization

## Related Work & Insights

- **vs XLSR-Conformer**: Pure attention architecture; this paper's hybrid SSM improves both efficiency and performance
- **vs RawBMamba**: Heuristic bidirectional Mamba strategy; this paper's Hydra-based native bidirectionality is more elegant
- **vs Jamba/Samba**: Hybrid architectures from the LLM domain; this paper is the first to systematically apply this paradigm to ADD

## Rating

- Novelty: ⭐⭐⭐⭐ Systematic exploration of SSM-Attention hybrid for ADD; Hydra introduction is novel
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four topologies × four variants × multiple depths × multiple datasets — very comprehensive
- Writing Quality: ⭐⭐⭐⭐ Thorough background knowledge, clear experimental organization
- Recommendation: ⭐⭐⭐⭐ Provides systematic architecture selection guidance for the ADD field

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Detect All-Type Deepfake Audio: Wavelet Prompt Tuning for Enhanced Auditory Perception](../../AAAI2026/ai_safety/detect_all-type_deepfake_audio_wavelet_prompt_tuning_for_enhanced_auditory_perce.md)
- [\[CVPR 2026\] Tutor-Student Reinforcement Learning: A Dynamic Curriculum for Robust Deepfake Detection](../../CVPR2026/ai_safety/tutor-student_reinforcement_learning_a_dynamic_curriculum_for_robust_deepfake_de.md)
- [\[ICCV 2025\] Vulnerability-Aware Spatio-Temporal Learning for Generalizable Deepfake Video Detection](../../ICCV2025/ai_safety/vulnerability-aware_spatio-temporal_learning_for_generalizable_deepfake_video_de.md)
- [\[NeurIPS 2025\] Not All Deepfakes Are Created Equal: Triaging Audio Forgeries for Robust Deepfake Singer Identification](../../NeurIPS2025/ai_safety/not_all_deepfakes_are_created_equal_triaging_audio_forgeries_for_robust_deepfake.md)
- [\[AAAI 2026\] Reference Recommendation based Membership Inference Attack against Hybrid-based Recommender Systems](../../AAAI2026/ai_safety/reference_recommendation_based_membership_inference_attack_against_hybrid-based_.md)

</div>

<!-- RELATED:END -->
