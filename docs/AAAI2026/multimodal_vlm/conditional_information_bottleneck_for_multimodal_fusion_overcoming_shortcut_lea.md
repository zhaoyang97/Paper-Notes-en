---
title: >-
  [Paper Note] Conditional Information Bottleneck for Multimodal Fusion: Overcoming Shortcut Learning in Sarcasm Detection
description: >-
  [AAAI 2026][Multimodal VLM][Multimodal Fusion] This paper identifies three types of shortcut learning in multimodal sarcasm detection (character label bias, canned laughter label leakage, and sentiment inconsistency shortcuts), reconstructs a shortcut-free benchmark MUStARD++R, and proposes MCIB, a multimodal fusion framework based on the Conditional Information Bottleneck. MCIB achieves effective fusion by compressing redundancy in the primary modality while preserving complementary information from auxiliary modalities.
tags:
  - AAAI 2026
  - Multimodal VLM
  - Multimodal Fusion
  - Information Bottleneck
  - Sarcasm Detection
  - Shortcut Learning
  - Mutual Information
date: 2026-05-08
content_hash: 98156143097a114d
---

# Conditional Information Bottleneck for Multimodal Fusion: Overcoming Shortcut Learning in Sarcasm Detection

**Conference**: AAAI 2026
**arXiv**: [2508.10644](https://arxiv.org/abs/2508.10644)
**Code**: [GitHub](https://github.com/sljgkjhwe/MCIB.git)
**Area**: Multimodal VLM
**Keywords**: Multimodal Fusion, Information Bottleneck, Sarcasm Detection, Shortcut Learning, Mutual Information

## TL;DR
This paper identifies three types of shortcut learning in multimodal sarcasm detection (character label bias, canned laughter label leakage, and sentiment inconsistency shortcuts), reconstructs a shortcut-free benchmark MUStARD++R, and proposes MCIB, a multimodal fusion framework based on the Conditional Information Bottleneck. MCIB achieves effective fusion by compressing redundancy in the primary modality while preserving complementary information from auxiliary modalities.

## Background & Motivation

**Background**: Multimodal sarcasm detection requires integrating text, audio, and video to identify the discrepancy between surface meaning and true intent. Existing methods commonly leverage sentiment polarity contrast, external knowledge, and speaker characteristics as auxiliary signals.

**Limitations of Prior Work**: The authors identify three critical shortcut learning problems:
- **Character label bias**: Certain characters (e.g., Sheldon) are naturally inclined toward sarcastic expression, causing models that incorporate character labels to learn "who said it" rather than "what was said."
- **Canned laughter label leakage**: In sitcoms, sarcastic utterances are frequently followed by canned laughter, leading models to learn "laughter implies sarcasm." Removing laughter causes F1 to plummet from 73.47 to 43.59.
- **Sentiment inconsistency shortcut**: 99% of sarcastic samples exhibit inconsistent explicit/implicit sentiment, yet such labels are unavailable in real-world settings.

**Key Challenge**: Model performance in sarcasm detection largely stems from shortcuts rather than genuine sarcasm understanding. Furthermore, existing modality fusion methods do not yield significant information gain—adding extra modalities sometimes even degrades performance.

**Goal**: (1) Remove shortcut signals and reconstruct a fairer benchmark; (2) Design a fusion method that truly extracts cross-modal complementary information.

**Key Insight**: Apply the Conditional Information Bottleneck (CIB) to multimodal fusion by distinguishing primary and auxiliary modalities, compressing redundancy in the primary modality while preserving complementary information from the auxiliary modality.

**Core Idea**: Simultaneously compress primary modality redundancy $I(x_p; b)$ and preserve auxiliary modality complementarity $I(b; y|x_a)$ via the CIB, realizing a fusion paradigm of "removing redundancy while retaining complementarity."

## Method

### Overall Architecture
Three modalities (audio, video, text) are paired pairwise, forming three parallel CIB structures. Each CIB produces a latent state $b$ that captures task-relevant complementary information with redundancy removed. The three $b$ representations are concatenated for final prediction.

### Key Designs

1. **Conditional Information Bottleneck (CIB)**:

    - **Function**: For each modality pair, compress a latent state $b$ from the primary modality that retains task-relevant information.
    - **Mechanism**: $\min_{p(b|x_p, x_a)} I(x_p; b) - \lambda I(b; y | x_a)$. The first term compresses primary modality redundancy; the second preserves complementary information provided by the auxiliary modality.
    - The compression term is bounded by a variational upper bound: $I(x_p; b) \leq \mathbb{E}_{p(x_p)}[D_{KL}(q(b|x_p) \| r(b))]$, where $r(b) = \mathcal{N}(0, I)$.
    - The retention term is bounded by an ELBO lower bound: $I(b;y|x_a) \geq \mathbb{E}[\log q(y|b, x_a)]$.
    - **Design Motivation**: Conventional IB operates on two information sources; the primary-auxiliary design naturally extends it to the multimodal setting. The conditional mutual information term ensures that only information complementary to the auxiliary modality—absent from the primary—is retained.

2. **Pairwise Three-Modality Fusion**:

    - The three modalities are alternately assigned as primary or auxiliary, producing $b_0, b_1, b_2$.
    - Independent weights $\lambda_0, \lambda_1, \lambda_2$ control the compression–retention trade-off for each pair.

3. **MUStARD++R Dataset Reconstruction**:

    - All shortcut labels (character labels, sentiment labels) are removed.
    - Video clips are trimmed using utterance timestamps to eliminate canned laughter.

### Loss & Training
$\mathcal{L}_{total} = \alpha_0 \mathcal{L}_0 + \alpha_1 \mathcal{L}_1 + \alpha_2 \mathcal{L}_2 + \beta \mathcal{L}_{pred}$, where $\mathcal{L}_i = \mathcal{L}_{IB_i} + \lambda_i \mathcal{L}_{CIB_j}$. Feature extraction: DeBERTa (text), MFCC + OpenSMILE (audio), ResNet-152 (video). Trained on an A100 GPU.

## Key Experimental Results

### Main Results

| Method | Shortcut Labels | Precision | Recall | F1 |
|---|---|---|---|---|
| ABCA-IMI | ○◇ | 76.20 | 74.20 | 75.20 |
| SpeechPrompt v2 | ○ | 78.33 | 58.06 | 73.47 |
| SpeechPrompt v2 (w/o shortcuts) | — | 63.03 | 27.87 | 43.59 |
| GPT-4o | ◇ | 62.66 | 83.90 | 71.74 |
| GPT-4o (w/o shortcuts) | — | 67.11 | 85.47 | 75.19 |
| Gemini 2.5 | ◇ | 62.89 | 84.75 | 72.20 |
| **MCIB** | ○ | **77.18** | **76.30** | **76.85** |
| **MCIB (w/o shortcuts)** | — | **76.14** | **75.83** | **75.64** |

**Key observation**: MCIB degrades by only 1.21% after shortcut removal (F1: 76.85→75.64), whereas SpeechPrompt v2 drops by 29.88%, demonstrating that MCIB does not rely on shortcuts.

### Ablation Study

| Configuration | F1 |
|---|---|
| Text only $x_t$ | 70.98 |
| Audio only $x_a$ | 68.97 |
| Visual only $x_v$ | 69.99 |
| Text + Visual $x_{tv}$ | 73.77 |
| Audio + Text $x_{at}$ | 73.69 |
| Optimal three-modality combination $x_{va}+x_{at}+x_{tv}$ | **75.64** |
| w/o Transformer | 74.32 |
| w/o Fine-Grained | 71.19 |

1. **Modality combination**: Text performs best as the primary modality; the combination of visual-assisted audio, audio-assisted text, and text-assisted visual yields the best result.
2. **Transformer vs. MLP**: The Transformer encoder improves performance by 1.32% over MLP.
3. **Fine-grained vs. coarse-grained features**: Word-level alignment outperforms coarse-grained features by 4.45%.

### Key Findings
1. **Shortcut learning is severely harmful**: Performance drops substantially for most methods after shortcut removal, indicating that prior "SOTA" results largely relied on spurious data correlations.
2. **GPT-4o/Gemini 2.5 improve after shortcut removal**: Character information acts as noise for LLMs, interestingly suggesting that LLMs and specialized models exhibit different sensitivities to shortcuts.
3. **MCIB's fusion optimization is effective**: Venn diagram visualizations show a reduced redundancy region and an enlarged complementarity region, validating the information-theoretic objectives of CIB.

## Highlights & Insights
- The **systematic analysis of shortcut learning** is exemplary: quantitatively revealing three types of shortcuts via chi-square tests, Phi coefficients, and before/after comparisons is far more convincing than merely asserting "bias exists."
- The **"compress redundancy, retain complementarity" fusion paradigm** is elegant: formalizing multimodal fusion as a compression–retention optimization via CIB is transferable to any multimodal task.
- The **MUStARD++R dataset** is an independent contribution to the community: a clean benchmark free of canned laughter, character labels, and sentiment annotations enables fairer evaluation of fusion methods.
- The **pairwise fusion strategy** cleverly decomposes the three-modality problem into three two-modality subproblems, avoiding the difficulty of directly handling the high-dimensional joint distribution.

## Limitations & Future Work
1. **Evaluation is limited to MUStARD++ and its variants**: MSD datasets are small (only a few hundred samples), and the generalizability of findings requires broader validation.
2. **Large number of hyperparameters** ($\lambda_0, \lambda_1, \lambda_2, \alpha_0, \alpha_1, \alpha_2, \beta$, totaling 7), leading to high tuning cost.
3. **Modality pairing strategy is selected manually**: Whether optimal modality pairing can be learned automatically remains an open question.
4. **Lacks in-depth comparison with the latest multimodal large language models** (e.g., GPT-4o with chain-of-thought).
5. **The CIB framework is generalizable** to broader multimodal sentiment analysis tasks such as deception detection and humor recognition.

## Related Work & Insights
- **IB-based methods**: SIB (single modality–target), DIB (modality pairs), ITHP (two-level IB); MCIB is the first to introduce conditional mutual information for selective inter-modality information transfer.
- **Conditional Mutual Information (CIB)**: Gondek et al. proposed maximizing conditional mutual information to avoid redundancy in clustering; Li et al. demonstrated that modalities with high complementarity exhibit lower robustness.
- **MUStARD/MUStARD++**: Mainstream MSD datasets; this paper exposes their shortcut problems.
- **Insight**: The information bottleneck perspective can be applied to analyze the redundancy–complementarity trade-off in any multimodal fusion task.

## Rating

| Dimension | Score (1–5) |
|---|---|
| Novelty | 4.0 |
| Technical Depth | 4.0 |
| Experimental Thoroughness | 3.5 |
| Writing Quality | 3.5 |
| Practical Value | 3.5 |
| **Overall** | **3.7** |

## Related Papers

- [\[ACL 2026\] From Verbatim to Gist: Distilling Pyramidal Multimodal Memory via Semantic Information Bottleneck](../../ACL2026/multimodal_vlm/from_verbatim_to_gist_distilling_pyramidal_multimodal_memory_via_semantic_inform.md)
- [\[ICML 2025\] Learning Optimal Multimodal Information Bottleneck Representations](../../ICML2025/multimodal_vlm/learning_optimal_multimodal_information_bottleneck_representations.md)
- [\[AAAI 2026\] Learning to Tell Apart: Weakly Supervised Video Anomaly Detection via Disentangled Semantic Alignment](learning_to_tell_apart_weakly_supervised_video_anomaly_detection_via_disentangle.md)
- [\[AAAI 2026\] Exploring LLMs for Scientific Information Extraction using the SciEx Framework](exploring_llms_for_scientific_information_extraction_using_the_sciex_framework.md)
- [\[AAAI 2026\] BOFA: Bridge-Layer Orthogonal Low-Rank Fusion for CLIP-Based Class-Incremental Learning](bofa_bridge-layer_orthogonal_low-rank_fusion_for_clip-based_.md)

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] From Verbatim to Gist: Distilling Pyramidal Multimodal Memory via Semantic Information Bottleneck](../../ACL2026/multimodal_vlm/from_verbatim_to_gist_distilling_pyramidal_multimodal_memory_via_semantic_inform.md)
- [\[AAAI 2026\] Information Theoretic Optimal Surveillance for Epidemic Prevalence in Networks](information_theoretic_optimal_surveillance_for_epidemic_prevalence_in_networks.md)
- [\[AAAI 2026\] BOFA: Bridge-Layer Orthogonal Low-Rank Fusion for CLIP-Based Class-Incremental Learning](bofa_bridge-layer_orthogonal_low-rank_fusion_for_clip-based_.md)
- [\[NeurIPS 2025\] Structure-Aware Fusion with Progressive Injection for Multimodal Molecular Representation Learning](../../NeurIPS2025/multimodal_vlm/structure-aware_fusion_with_progressive_injection_for_multimodal_molecular_repre.md)
- [\[AAAI 2026\] Learning to Tell Apart: Weakly Supervised Video Anomaly Detection via Disentangled Semantic Alignment](learning_to_tell_apart_weakly_supervised_video_anomaly_detection_via_disentangle.md)

<!-- RELATED:END -->
