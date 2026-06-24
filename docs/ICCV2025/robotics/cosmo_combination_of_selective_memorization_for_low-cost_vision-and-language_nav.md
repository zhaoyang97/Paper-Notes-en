---
title: >-
  [Paper Note] COSMO: Combination of Selective Memorization for Low-cost Vision-and-Language Navigation
description: >-
  [ICCV 2025][Robotics][Vision-and-Language Navigation] This paper proposes COSMO, a low-cost VLN architecture combining selective memorization, which replaces the computationally expensive attention mechanisms in Transformers with two customized selective state space modules—Round Selective Scan (RSS, capturing global context in a single scan pass) and Cross-modal Selective State Space Module (CS3, dual-stream cross-modal interaction)—achieving navigation performance surpassin…
tags:
  - "ICCV 2025"
  - "Robotics"
  - "Vision-and-Language Navigation"
  - "State Space Model"
  - "Mamba"
  - "Hybrid Architecture"
  - "Low-cost Navigation"
date: 2026-05-08
content_hash: bf55b8403469d39b
---

# COSMO: Combination of Selective Memorization for Low-cost Vision-and-Language Navigation

**Conference**: ICCV 2025
**arXiv**: [2503.24065](https://arxiv.org/abs/2503.24065)  
**Code**: N/A  
**Area**: Robotics
**Keywords**: Vision-and-Language Navigation, State Space Model, Mamba, Hybrid Architecture, Low-cost Navigation

## TL;DR

This paper proposes COSMO, a low-cost VLN architecture combining selective memorization, which replaces the computationally expensive attention mechanisms in Transformers with two customized selective state space modules—Round Selective Scan (RSS, capturing global context in a single scan pass) and Cross-modal Selective State Space Module (CS3, dual-stream cross-modal interaction)—achieving navigation performance surpassing the baseline DUET with only 15.5% of its parameters and 9.3% of its FLOPs.

## Background & Motivation

Vision-and-Language Navigation (VLN) requires an agent to navigate unseen 3D environments following natural language instructions, and is a core task in embodied intelligence. Current VLN research faces a dual dilemma:

**The cost of performance gains**: State-of-the-art methods (e.g., BEVBert, GridMM, KERM) improve performance by incorporating external knowledge bases, map information, and depth cues, causing model parameters to balloon from DUET's 181M to over 222M, with FLOPs reaching 15–18G. NaviLLM even reaches 6,633M parameters and 1,011G FLOPs.

**Performance degradation on long instructions**: As navigation instruction length increases and path complexity grows, Transformer-based methods suffer significant performance drops. The $O(L^2)$ attention complexity of Transformers makes them inefficient when processing long historical sequences.

**The potential and challenges of Mamba/SSMs**: State Space Models (SSMs) offer linear computational complexity $O(L)$ and strong long-sequence modeling capability, making them promising alternatives to Transformers. However, directly applying Mamba to VLN faces two fundamental problems:

**Insufficient spatial relationship modeling**: SSMs are inherently causal 1D sequence models and cannot learn inter-viewpoint spatial relationships as attention mechanisms do.

**Insufficient input selection capability**: Although SSMs outperform Transformers on generation tasks, they perform poorly on tasks requiring selection from inputs—VLN being a typical example.

Empirical validation: directly replacing the Transformer components in DUET with Mamba causes SR to plummet from 46.98% to 32.25% (a drop of 14.73%).

## Method

### Overall Architecture

COSMO adopts a hybrid architecture philosophy: "SSMs handle selective memorization; Transformers handle precise decision-making":

1. **Node Encoder**: Constructs a topological map and encodes panoramic observations at each node.
2. **Global Cross-modal Encoder**: Performs cross-modal fusion of the topological map and instructions (CS3 + GASA + cross-attention).
3. **Local Cross-modal Encoder**: Performs fine-grained fusion of current-node viewpoints and instructions (CS3 + RSS + cross-attention + self-attention).
4. **Dynamic Action Fusion**: Integrates predictions from both scales to produce the final action.

### Key Designs

1. **Round Selective Scan (RSS)**:

    - **Function**: Captures global relationships among all tokens in a single scan pass, replacing the high cost and causality constraints of bidirectional scanning.
    - **Mechanism**: The input sequence $x'$ is flipped and concatenated to form $x = [x' | \text{flip}(x')]$, with the class token appearing at both ends. A single forward scan is performed; by the time the second half of the sequence is processed, the state space already contains information from all tokens. The output is split into two equal halves, the second half is flipped, and both halves are summed to produce the final result.
    - **Design Motivation**: Bidirectional scanning (Bi-Mamba) requires two passes and remains causal (each token can only attend to tokens preceding it in its scan direction). RSS's elegant "flip-and-concatenate" design allows every token to access global information through the state space in a single pass; thanks to hardware-aware parallel algorithms, doubling the sequence length has negligible impact on training and inference time.

2. **Cross-modal Selective State Space Module (CS3)**:

    - **Function**: Adapts SSMs into a dual-stream cross-modal interaction architecture for deep fusion of visual and textual features.
    - **Mechanism**: Taking the update of $x$ (visual) as an example: $y$ (text) is used to construct $\mathbf{B}$ and $\mathbf{\Delta}$ (controlling how information is written into the state space), while the class token of $x$ is used to construct $\mathbf{C}$ (controlling how information is read from the state space into the target modality). $y$ first undergoes RSS-style flip-and-concatenate processing; after scanning, the output of $y$'s class token serves as a gate to selectively filter relevant information from $x$. The reverse operation updates $y$.
    - **Design Motivation**: Existing multimodal SSM methods (e.g., VL-mamba) naively concatenate the two modality sequences into a single stream, but in VLN the visual and textual sequences are severely length-imbalanced, requiring fine-grained modality alignment and comprehensive cross-modal interaction. CS3 achieves this by using one modality to control state updates and the other to control output readout.

3. **Hybrid Architecture Design**:

    - **Function**: Combines the efficient memory filtering of SSMs with the precise contextual selection of Transformers.
    - **Mechanism**: Within the cross-modal encoder, CS3 first performs semantic alignment and selective memorization (filtering visual information irrelevant to the instruction), RSS then broadcasts contextual information, and finally cross-attention and self-attention perform token-level precise localization and action decision-making.
    - **Design Motivation**: Ablation experiments (Table 5) clearly demonstrate: a pure SSM architecture (SSM+SSM) achieves only 41.92% SR, a pure Transformer (Trans+Trans) achieves 49.47%, while **SSM→Trans (COSMO) achieves the optimal 50.81%**, validating that the "filter-then-select" hybrid strategy outperforms any single-paradigm architecture.

### Loss & Training

- Follows DUET's training strategy, using identical input features and hyperparameters.
- The text encoder uses TinyBert (hidden size 312, intermediate size 1200).
- The state space dimension for both RSS and CS3 is set to 16.
- The optimal checkpoint is selected based on SR+SPL on the validation unseen split.

## Key Experimental Results

### Main Results

**REVERIE Dataset:**

| Method | Val OSR↑ | Val SR↑ | Val SPL↑ | Test SR↑ | Test SPL↑ | Param(M)↓ | FLOPs(G)↓ |
|--------|---------|--------|---------|---------|----------|----------|----------|
| DUET | 51.07 | 46.98 | 33.73 | 52.51 | 36.06 | 181 | 4.95 |
| KERM | 55.21 | 50.44 | 35.38 | 52.43 | 39.21 | 222 | 15.24 |
| NaviLLM | 52.27 | 42.15 | 35.68 | 39.80 | 32.33 | 6633 | 1011 |
| **COSMO** | **56.09** | **50.81** | **35.93** | **52.53** | **36.12** | **28** | **0.46** |

**R2R Dataset:**

| Method | Val NE↓ | Val SR↑ | Val SPL↑ | Test NE↓ | Test SR↑ |
|--------|--------|--------|---------|---------|---------|
| DUET | 3.31 | 72 | 60 | 3.65 | 69 |
| **COSMO** | **3.15** | **73** | **61** | **3.43** | **71** |

**R2R-CE Dataset (Test):**

| Method | SR↑ | SPL↑ |
|--------|-----|------|
| DUET | 42 | 36 |
| **COSMO** | **47** | **40** |

### Ablation Study

**RSS vs. Alternative Scanning Strategies (REVERIE Val Unseen):**

| Configuration | SR↑ | SPL↑ | Inference Time (s)↓ | Note |
|--------------|-----|------|-------------------|------|
| Mamba + CS3 (#1) | 47.20 | 32.04 | 10.46 | Standard Mamba replacing RSS |
| Bi-Mamba + CS3 (#2) | 50.75 | 34.77 | 11.38 | Bidirectional Mamba replacing RSS |
| RSS + Bi-Mamba (#3) | 46.95 | 31.40 | 10.80 | Bi-Mamba replacing CS3 |
| **RSS + CS3 (#4)** | **50.81** | **35.93** | **10.64** | Full COSMO |

**Architecture Design Ablation:**

| Architecture | SR↑ | SPL↑ | Note |
|-------------|-----|------|------|
| SSM + SSM | 41.92 | 27.61 | Pure SSM (worst) |
| Trans + SSM | 46.58 | 31.38 | Transformer first → SSM second |
| Trans + Trans | 49.47 | 31.10 | Scaled-down pure Transformer |
| **SSM + Trans** | **50.81** | **35.93** | COSMO: SSM first → Trans second (best) |

**Performance with Direct Mamba Substitution (Failure Cases):**

| Method | SR↑ | Note |
|--------|-----|------|
| Mamba (single-stream) | 32.25 | 14.73% below DUET; confirms direct SSM replacement is infeasible |
| Bi-Mamba | 35.61 | Only +3.26%; still far insufficient |

### Key Findings

1. **Dramatic reduction in parameters and computation**: COSMO requires only 28M parameters (15.5% of DUET) and 0.46G FLOPs (9.3% of DUET) while achieving equal or better navigation performance.
2. **The SSM→Trans hybrid order is critical**: SSMs first perform selective memory filtering, followed by Transformers for precise action decision-making; reversing this order degrades performance.
3. **RSS strictly outperforms Bi-Mamba**: The single-scan approach is not only more efficient (0.74s faster inference) but also achieves 1.16% higher SPL.
4. **CS3 markedly outperforms simple sequence concatenation**: SR gap of 3.86% and SPL gap of 4.53%, validating the necessity of the dual-stream architecture for VLN.
5. **COSMO shows the largest gains on R2R-CE**: SR +5%, SPL +4%, indicating a more pronounced advantage in continuous environments with longer decision sequences.

## Highlights & Insights

- **The concept of "selective memorization" precisely addresses the core requirement of VLN**: Navigation accumulates large quantities of visual observations, but only those relevant to the instruction need to be retained—the selective mechanism of SSMs is naturally suited to this need.
- **The flip-and-concatenate design of RSS is particularly elegant**: It achieves global context awareness with minimal implementation complexity, avoiding the overhead of multi-directional scanning.
- **The dual-stream design philosophy of CS3**: Using the source modality to control state updates and the target modality to control output readout provides a principled solution for cross-modal SSM interaction.
- **6.5× parameter reduction + 10.7× FLOPs reduction**: This has significant practical implications for deployment on edge devices.
- **Node encoder improvement**: Relocating topological map construction to the node encoding stage reduces the parameter requirements of the cross-modal encoder.

## Limitations & Future Work

1. The state space dimension of RSS and CS3 ($N=16$) is manually specified; larger state spaces may yield further improvements.
2. COSMO's SPL on the REVERIE test split is nearly identical to DUET (36.12 vs. 36.06), indicating limited improvement in navigation efficiency.
3. No fair comparison is made with methods that leverage additional data or knowledge (e.g., ScaleVLN, BEVBert), as their performance gains partly stem from extra information.
4. The gating mechanism in CS3 relies on the class token output; when sequences are very long, the class token may fail to effectively compress all relevant information.
5. Newer SSM variants (e.g., Mamba2) are not explored as potential avenues for further performance improvement.

## Related Work & Insights

- Compared with multimodal SSM works such as VL-mamba and Cobra, CS3's dual-stream design is better suited to scenarios with imbalanced sequence lengths between modalities (in VLN, visual sequences are far longer than textual ones).
- The success of the hybrid architecture (SSM + Transformer) provides a design reference for other tasks requiring "memorization + decision-making" (e.g., dialogue systems, long-document reasoning).
- The flip-and-concatenate technique in RSS can be generalized to other vision tasks requiring non-causal SSMs.
- COSMO's success demonstrates that architectural innovation may be more cost-effective than simply scaling model size or incorporating external knowledge.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ RSS and CS3 are entirely novel designs; the hybrid architecture philosophy is conceptually profound.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three VLN datasets with comprehensive ablation and architecture design validation.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured, though the description of CS3 could be made more intuitive.
- **Value**: ⭐⭐⭐⭐⭐ A 10× reduction in computation with equivalent or improved performance carries major practical significance for real-world deployment of embodied intelligence systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] NavMorph: A Self-Evolving World Model for Vision-and-Language Navigation in Continuous Environments](navmorph_a_self-evolving_world_model_for_vision-and-language_navigation_in_conti.md)
- [\[ICCV 2025\] Selective Contrastive Learning for Weakly Supervised Affordance Grounding](selective_contrastive_learning_for_weakly_supervised_affordance_grounding.md)
- [\[ICML 2026\] Neural Low-Discrepancy Sequences](../../ICML2026/robotics/neural_low-discrepancy_sequences.md)
- [\[ECCV 2024\] LLM as Copilot for Coarse-Grained Vision-and-Language Navigation](../../ECCV2024/robotics/llm_as_copilot_for_coarse-grained_vision-and-language_navigation.md)
- [\[CVPR 2026\] TRM-VLA: Temporal-Aware Chain-of-Thought Reasoning and Memorization for Vision-Language-Action Models](../../CVPR2026/robotics/trm-vla_temporal-aware_chain-of-thought_reasoning_and_memorization_for_vision-la.md)

</div>

<!-- RELATED:END -->
