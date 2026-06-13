---
title: >-
  [Paper Note] LaMoGen: Language to Motion Generation Through LLM-Guided Symbolic Inference
description: >-
  [CVPR 2026][Human Understanding][Text-driven motion generation] This paper proposes LabanLite, a symbolic motion representation, and the LaMoGen framework…
tags:
  - "CVPR 2026"
  - "Human Understanding"
  - "Text-driven motion generation"
  - "Labanotation"
  - "symbolic reasoning"
  - "LLM Agent"
  - "interpretable motion synthesis"
date: 2026-05-08
content_hash: ec7c6c51558c54e4
---

# LaMoGen: Language to Motion Generation Through LLM-Guided Symbolic Inference

**Conference**: CVPR 2026
**arXiv**: [2603.11605](https://arxiv.org/abs/2603.11605)  
**Code**: Available ([Project Page](https://jjkislele.github.io/LaMoGen/))  
**Area**: Human Understanding
**Keywords**: Text-driven motion generation, Labanotation, symbolic reasoning, LLM Agent, interpretable motion synthesis

## TL;DR

This paper proposes LabanLite, a symbolic motion representation, and the LaMoGen framework, which for the first time enables LLMs to autonomously compose motion sequences through interpretable Laban symbol reasoning, surpassing conventional text-motion joint embedding methods in temporal precision and controllability.

## Background & Motivation

**Background**: Text-driven human motion generation (Text-to-Motion) has made significant progress in recent years. Mainstream methods rely on text-motion joint embedding spaces and generate motion sequences via diffusion models or autoregressive Transformers. Representative works include MDM, ReMoDiff, MoDiff, CoMo, and MotionGPT.

**Limitations of Prior Work**: Joint-embedding-based methods perform poorly on **temporal precision** and **fine-grained semantics**. For example, given the instruction "Walk forward in 5 steps and then walk backward in 3 steps," existing methods typically generate a generic "walking forward" motion without accurately reflecting step counts or temporal ordering. Furthermore, these methods lack **interpretability**—generated results are black-box outputs that users cannot inspect or edit at intermediate stages.

**Key Challenge**: There is a fundamental gap between the high-level semantic structure of language descriptions (which explicitly encode body parts, directions, temporal ordering, and repetition counts) and the continuity and opacity of motion embedding spaces. Prior attempts to decompose text into body-part-level tokens (e.g., CoMo's Posescript) encode only static poses and lack the expressiveness needed to represent motion transitions and temporal dynamics.

**Goal**: To establish an interpretable and editable intermediate symbolic representation that enables LLMs to autonomously compose motion sequences through symbolic reasoning, while ensuring precision in temporal structure, body-part coordination, and language alignment.

**Key Insight**: Inspiration is drawn from the Labanotation dance notation system, which encodes movement attributes—including body parts, directions, levels, and durations—in symbolic form, and is inherently interpretable and structured. The authors accordingly design LabanLite as a "symbolic bridge" connecting language and motion.

**Core Idea**: Complex motions are decomposed into sequences of Laban symbols; the LLM reasons and plans in the symbolic space; a decoder then reconstructs continuous motion trajectories from the symbols.

## Method

### Overall Architecture

LaMoGen is a two-stage generation framework following the pipeline **Text → LabanLite → Motion**:

- **Stage 1 (High-level Semantic Planning)**: An LLM converts textual instructions into conceptual Laban symbol sequences via Retrieval-Augmented Prompting.
- **Stage 2 (Low-level Motion Synthesis)**: A Kinematic Detail Augmentor autoregressively completes the conceptual symbols into full LabanLite encodings, which are then decoded into continuous motion by the Laban-Motion Decoder.

The framework consists of two core modules: the **Laban-Motion Encoder-Decoder** (bidirectional conversion between motion and symbols) and **LLM-Guided Text-Laban-Motion Generation** (LLM-driven symbolic composition and motion generation).

### Key Designs

#### 1. LabanLite Motion Representation

- **Function**: Reformulates traditional Labanotation into a frame-level symbolic representation suitable for machine learning and LLM reasoning.
- **Mechanism**: Three key enhancements are made to Labanotation: (1) symbols are divided into **conceptual symbols** (primary motion structure, e.g., direction and level changes) and **detail symbols** (fine-grained attributes, e.g., bending angle); (2) event-level annotations are replaced with **frame-level annotations**, with each frame corresponding to one Laban instance; (3) each conceptual symbol is paired with a formatted **conceptual description** (`<body-part group> <moving semantic> in <time> seconds`).
- **Design Motivation**: The conceptual/detail separation allows the LLM to operate exclusively on high-level conceptual symbols (which align well with natural language), while details are automatically completed by a dedicated Augmentor. Frame-level annotation improves compatibility with ML models, and the standardized conceptual description format ensures unambiguous conversion between symbols and text.

#### 2. Laban Codebook and Encoder-Decoder

- **Function**: Establishes a learnable embedding space for Laban symbols, supporting bidirectional conversion between motion and symbols.
- **Mechanism**: All unique frame-level Laban instances are assigned Laban codes, forming a codebook $C=\{c_n\}_{n=1}^{N}$. The encoder activates corresponding codebook entries via a binary indicator vector $v_t$, and sums their embeddings to obtain the frame-level latent $z_t=\sum_n v_t^n c_n$. A Transformer-based decoder reconstructs the motion from the latent.
- **Design Motivation**: Unlike VQ-VAE, this design uses **additive composition** rather than single-token selection, approximating continuous variation through linear combination and enabling complex motions to be built from simple symbols.

#### 3. Automatic Laban Symbol Detection Pipeline

- **Function**: Automatically extracts Laban symbol sequences from continuous motion data.
- **Mechanism**: A three-step pipeline: (1) **Dynamic interval segmentation**: classifying motion frames as dynamic or static and partitioning by atomic actions; (2) **Frame-level symbol extraction**: computing direction/level from 3D displacements of end-effectors relative to the pelvis, computing orientation/bending from Euler angles, and quantifying effort from pelvis velocity; (3) **Interval-level symbol aggregation**: assigning representative symbol combinations to each temporal interval.
- **Design Motivation**: Provides high-quality training data for the Conceptual Description Database and Laban Benchmark. Discretization rules strictly follow threshold standards accepted by professionals in the Labanotation literature.

#### 4. LLM-Guided Motion Concept Composition

- **Function**: Leverages an LLM to convert user text instructions into a conceptual Laban symbol plan.
- **Mechanism**: A **Conceptual Description Database** (a key-value store mapping motion captions to conceptual descriptions) is maintained. At inference time, CLIP computes semantic similarity between the user query and database captions to retrieve Top-K entries as in-context examples for the LLM. The LLM then infers correspondences between textual instructions and symbolic motion patterns, editing or composing new conceptual descriptions.
- **Design Motivation**: The RAG strategy allows the LLM to understand the Laban symbol system without fine-tuning. The standardized format of conceptual descriptions ensures that LLM outputs can be unambiguously mapped back to Laban symbols.

#### 5. Kinematic Detail Augmentor

- **Function**: Completes the conceptual symbols produced by the LLM into full LabanLite encodings.
- **Mechanism**: Conditioned on the text $m$ and masked conceptual vectors $\hat{v}_{1:t-1}$, the Augmentor autoregressively predicts the complete binary indicator vector $v_t$ for each frame, activating both conceptual and detail attribute entries in the codebook. During training, random masking is applied to the conceptual vectors (optimal masking ratio = 0.3) to improve generalization.
- **Design Motivation**: LLMs excel at high-level planning but lack temporal modeling precision; detail symbols (orientation, bending, effort) require a dedicated temporal model. This stage adds approximately 60% of the information to the symbol sequence.

### Loss & Training

- **Codebook Training**: Jointly optimizes decoder parameters $\theta$ and codebook $C$ by minimizing the reconstruction loss $\mathcal{L}_{rec} = \|X - \hat{X}\|_1 + \lambda\|\dot{X} - \dot{\hat{X}}\|_1$ (pose L1 + velocity L1).
- **Augmentor Training**: Binary cross-entropy loss $\mathcal{L}_{gen} = -\sum_{t,n}[v_t^n \log p_t^n + (1-v_t^n)\log(1-p_t^n)]$, predicting the activation probability of each codebook entry per frame.
- **End-of-sequence**: An `<EOS>` token is appended and the codebook is extended to $N+1$ entries to mark motion termination.

## Key Experimental Results

### Main Results

**Table 1: Quantitative Comparison on Laban Benchmark** (Labanotation-based metrics + R@3 / FID)

| Method | avg.SMT↑ | avg.TMP↑ | avg.HMN↑ | R@3↑ | FID↓ |
|--------|----------|----------|----------|------|------|
| MDM | 0.338 | 0.298 | 0.201 | 0.180 | 22.81 |
| ReMoDiff | 0.441 | 0.365 | 0.265 | 0.192 | 7.121 |
| MoDiff | 0.466 | 0.366 | 0.274 | 0.196 | 5.701 |
| CoMo | 0.393 | 0.239 | 0.251 | 0.176 | 21.94 |
| MotionGPT | 0.461 | 0.347 | 0.307 | 0.195 | 2.072 |
| **LaMoGen (GPT4.1)** | **0.534** | **0.502** | **0.393** | **0.208** | **1.861** |
| LaMoGen (Human) | 0.626 | 0.628 | 0.462 | 0.211 | 1.769 |

**Table 2: Comparison on the Standard HumanML3D Benchmark**

| Method | R@1↑ | R@3↑ | FID↓ | MM-Dist↓ | Diversity→ |
|--------|------|------|------|----------|-----------|
| Real data | 0.511 | 0.797 | 0.002 | 2.974 | 9.503 |
| ReMoDiff | 0.510 | 0.795 | **0.103** | **2.974** | 9.018 |
| CoMo | 0.502 | 0.790 | 0.262 | 3.032 | 9.936 |
| MotionGPT | 0.492 | 0.778 | 0.232 | 3.096 | 9.528 |
| **LaMoGen (GPT4.1)** | 0.491 | **0.796** | 0.252 | 3.087 | 9.124 |
| LaMoGen (Human) | **0.513** | **0.813** | 0.206 | 2.993 | 9.635 |

### Ablation Study

**Effect of LLM Capability**: Stronger LLMs yield better generation quality. GPT-4.1 > DeepSeek-V3 > Qwen3 > GPT-4.1mini > None (no LLM), with consistent improvements observed across Laban metrics and FID.

**Number of Retrieved Examples**: Top-K retrieval is evaluated on HumanML3D with GPT-4.1. Performance improves steadily from K=1 to K=3 (the LLM requires sufficient examples to imitate); K=5 or K=7 yields no further gain (overly long context causes the LLM to lose track of key cues). Top-3 is used by default.

**Masking Ratio**: Experiments on the random masking ratio applied to conceptual symbols during Augmentor training. A ratio of 0.3 is optimal—too low leads to over-reliance on conceptual cues (poor generalization), while too high provides insufficient conceptual guidance.

**Laban Symbol Detection Accuracy** (Table 3):

| Method | avg.SMT↑ | avg.TMP↑ | avg.HMN↑ |
|--------|----------|----------|----------|
| Ikeuchi et al. | 0.751 | 0.632 | 0.611 |
| **Ours** | **0.871** | **0.852** | **0.786** |

### Key Findings

1. **Symbolic reasoning substantially outperforms joint embedding**: On the Laban Benchmark, LaMoGen (GPT4.1) comprehensively surpasses all joint-embedding-based methods across SMT/TMP/HMN metrics, demonstrating the advantage of symbolic reasoning in temporal precision and body-part coordination.
2. **Unexpected performance of MotionGPT on structured instruction understanding**: Although MotionGPT performs modestly on conventional benchmarks, it outperforms CoMo on the Laban Benchmark, suggesting that traditional metrics fail to effectively differentiate the true capabilities of motion generation methods.
3. **Limitations of FID**: LaMoGen's FID is slightly inferior to certain baselines, because LabanLite's high-level abstraction assigns the same symbol to low-level variations under the same semantics (e.g., individual differences in arm-raising speed), which is an inherent expressiveness limitation of symbolic representations.
4. **Upper bound of the human composer**: Using ground-truth annotated conceptual symbols (Human) yields better results than the LLM composer, indicating that the LLM's symbolic composition capability still has room for improvement.

## Highlights & Insights

- **First framework for LLM-autonomous motion generation**: LaMoGen is the first framework enabling an LLM to autonomously compose motions through symbolic reasoning without fine-tuning, establishing a new paradigm for agent-based motion generation.
- **Dual advantage of symbolic representation**: LabanLite allows LLMs to "understand" motion (via conceptual descriptions) while also enabling human experts to directly inspect and edit intermediate results.
- **Evaluation contribution**: The proposed SMT/TMP/HMN Laban metrics fill a gap in existing evaluation protocols with respect to temporal precision and multi-part coordination.
- **Hierarchical design philosophy**: The two-stage architecture—conceptual/detail separation combined with a division of labor between the LLM and the Augmentor—represents an elegant "separation of concerns" design.

## Limitations & Future Work

1. **Expressiveness ceiling of LabanLite**: Discretization of symbols inevitably loses low-level motion detail, resulting in elevated FID. Future work could consider introducing continuous attribute fields or residual compensation mechanisms.
2. **LLM dependency**: Framework performance is bounded by LLM capability (weaker LLMs noticeably degrade symbolic composition quality), and inference requires API calls, adding latency and cost.
3. **Dataset limitations**: The Laban Benchmark is predominantly composed of walking-type motions, providing insufficient evaluation coverage for more complex whole-body actions such as dance or gymnastics.
4. **Fixed Laban symbol set**: Adherence to the traditional Labanotation symbol set for professional fidelity may limit descriptive capacity for emerging motion types.
5. **Retrieval dependency**: The effectiveness of the RAG strategy depends on the coverage of the Conceptual Description Database, and performance may be limited for motion patterns unseen during training.

## Related Work & Insights

- **CoMo (ECCV 2024)**: Decomposes motion into body-part-level pose codes using Posescript, but encodes only static poses and lacks temporal expressiveness. LaMoGen's Laban symbols encode both start/end poses and transition processes, providing richer semantics.
- **MotionGPT (NeurIPS 2024)**: Fine-tunes an LLM to process motion tokens. LaMoGen requires no LLM fine-tuning, instead enabling the LLM to operate in the symbolic space via RAG.
- **KP (Kinematic Phrase)**: Abstracts motion signals heuristically but is limited to low-level signals. LabanLite provides professional-grade high-level abstraction.
- **Broader Implications**: The paradigm of symbolic intermediate representation combined with LLM reasoning is transferable to other cross-modal generation tasks (e.g., music-to-dance, text-to-sign language); the key lies in identifying a structured symbolic system for the target modality.

## Rating

⭐⭐⭐⭐ — Introducing Labanotation into LLM-based motion generation is a clever and compelling innovation; the symbolic reasoning approach opens a new direction for interpretable and controllable motion generation, and the evaluation contribution of the Laban Benchmark is substantive. Nevertheless, the elevated FID and LLM dependency represent practical bottlenecks that require resolution in future work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MoLingo: Motion-Language Alignment for Text-to-Human Motion Generation](molingo_motion-language_alignment_for_text-to-motion_generation.md)
- [\[AAAI 2026\] SOSControl: Enhancing Human Motion Generation through Saliency-Aware Symbolic Orientation and Timing Control](../../AAAI2026/human_understanding/soscontrol_enhancing_human_motion_generation_through_saliency-aware_symbolic_ori.md)
- [\[CVPR 2026\] Active Inference for Micro-Gesture Recognition: EFE-Guided Temporal Sampling and Adaptive Learning](active_inference_for_micro-gesture_recognition_efe-guided_temporal_sampling_and_.md)
- [\[CVPR 2026\] HandX: Scaling Bimanual Motion and Interaction Generation](handx_scaling_bimanual_motion_and_interaction_generation.md)
- [\[AAAI 2026\] ReAlign: Text-to-Motion Generation via Step-Aware Reward-Guided Alignment](../../AAAI2026/human_understanding/realign_text-to-motion_generation_via_step-aware_reward-guided_alignment.md)

</div>

<!-- RELATED:END -->
