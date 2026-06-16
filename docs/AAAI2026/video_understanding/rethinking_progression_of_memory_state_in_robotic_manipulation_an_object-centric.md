---
title: >-
  [Paper Note] Rethinking Progression of Memory State in Robotic Manipulation: An Object-Centric Perspective
description: >-
  [AAAI 2026][Video Understanding][Non-Markovian Decision Making] This paper proposes LIBERO-Mem, a benchmark comprising 10 non-Markovian robotic manipulation tasks, and Embodied-SlotSSM…
tags:
  - "AAAI 2026"
  - "Video Understanding"
  - "Non-Markovian Decision Making"
  - "Object-Centric Memory"
  - "State Space Models"
  - "VLA"
  - "Robotics Benchmark"
date: 2026-05-08
content_hash: 657afde530b5a803
---

# Rethinking Progression of Memory State in Robotic Manipulation: An Object-Centric Perspective

**Conference**: AAAI 2026
**arXiv**: [2511.11478](https://arxiv.org/abs/2511.11478)  
**Code**: [libero-mem.github.io](https://libero-mem.github.io)  
**Area**: Video Understanding / Robotic Manipulation
**Keywords**: Non-Markovian Decision Making, Object-Centric Memory, State Space Models, VLA, Robotics Benchmark

## TL;DR

This paper proposes LIBERO-Mem, a benchmark comprising 10 non-Markovian robotic manipulation tasks, and Embodied-SlotSSM, an object-centric memory VLA framework combining Slot Attention with state space models, to address the failure of visuomotor policies in long-horizon tasks that require object-level historical reasoning under partial observability.

## Background & Motivation

**Background**: Humans can effortlessly recall past interactions with specific objects (e.g., where a salt shaker was placed, or whether salt has already been added to a dish), enabling precise execution of multi-step, long-horizon tasks. Current robotic visuomotor policies (e.g., OpenVLA, Octo, RT-1/2), however, typically rely solely on recent sensory inputs for decision-making and lack mechanisms for encoding and recalling **object-level history**.

**Limitations of Prior Work**:

**Markovian Assumption Bottleneck**: Most VLA models assume the current observation is sufficient to predict the optimal action. This assumption breaks down in tasks involving repeated steps, visually similar objects, or long-horizon temporal dependencies — where identical visual inputs may correspond to different semantic states (e.g., a bowl resting on a plate vs. a bowl just placed back onto a plate).

**Benchmark Insufficiency**: Existing benchmarks (e.g., RLBench, LIBERO, RoboCasa) are largely constructed under the Markovian assumption. While MemoryBench and MIKASA-Robo address memory, they lack systematic stress testing for object-level ambiguity and temporal extension.

**Token Scalability Problem**: OpenVLA encodes video sequences with 256 dense tokens, and object-centric VLAs use 16 slot tokens, but token counts grow **linearly** with both slot and sequence dimensions, becoming infeasible for long-horizon tasks spanning hundreds of frames.

**Key Challenge**: When visual observations at two time steps $t_1$ and $t_2$ satisfy $\mathbf{v}_{t_1} \approx \mathbf{v}_{t_2}$ but require different actions (i.e., $P(\mathbf{a}_{t_1}|\mathbf{v}_{1:t_1},l) \neq P(\mathbf{a}_{t_2}|\mathbf{v}_{1:t_2},l)$), purely reactive policies will inevitably fail. This constitutes an **object-level Partially Observable Markov Decision Process (POMDP)**.

**Key Insight**: Drawing from object-centric learning and state space models, this work designs structured, persistent memory representations that support long-horizon non-Markovian reasoning while remaining computationally tractable.

## Method

### Overall Architecture

Embodied-SlotSSM comprises three core components: (1) **Slot Attention**, which decomposes dense visual features into discrete object-centric tokens; (2) **SlotSSM**, which tracks object temporal dynamics via a slot-based state space model; and (3) a **Relation Encoder + LLM Action Decoder**, which aligns object memory with the current scene for action prediction.

### Key Designs

#### 1. **LIBERO-Mem Benchmark**

Ten tasks are designed spanning four object-centric memory dimensions:

- **Object Motion (OM)**: The robot must remember its last action (pick up or place down) to act correctly (T1, T2).
- **Object Sequence (OS)**: Success depends on recalling how many times an object has been manipulated; visual cues alone are insufficient (T3–T6, with 3/5/7 repeated pick-and-place cycles).
- **Object Relation (OR)**: The robot must track the temporal order and relationships of object interactions (T7–T8, swapping bowl positions).
- **Object Occlusion (OO)**: Occluded objects require the robot to rely on memory of past placements to identify the target (T9–T10).

**Distinguishing Features** (compared to MemoryBench/MIKASA-Robo):
- Non-Markovian observations ✓
- Long-horizon trajectories (200–700 frames) ✓
- **Sub-goal-aware evaluation** ✓ (unique; supports fine-grained progress assessment)
- **Object identity ambiguity** ✓ (unique; visually identical bowls/plates distinguished only by asset ID)
- **Temporal extension stress testing** ✓ (unique)

Each task includes 120 collected trajectories (100 training + 20 validation), gathered via keyboard control with multi-key tracking to produce smooth demonstrations.

#### 2. **Slot Attention for Object Localization**

Dense visual embeddings $\mathbf{v}_t \in \mathbb{R}^{K \times D_{\text{enc}}}$ are decomposed into $N$ object-centric tokens $\mathbf{s}_t = \{\mathbf{s}_t^1, ..., \mathbf{s}_t^N\}$, with $N=16$. Spatial features are iteratively bound to a fixed number of learnable object queries via attention and GRU-based recurrent updates.

**Temporally Consistent Initialization**: Slots are randomly initialized at $t=0$; for $t>0$, the final slot outputs from the previous frame initialize the current frame's slots, enabling slot identity propagation and persistent object tracking across time.

$$\mathbf{s}_t^{(0)} = \begin{cases} \text{RandomInit}() & t=0 \\ \mathbf{s}_{t-1}^{(T)} & t>0 \end{cases}$$

**Temporal Contrastive Loss**: Within a fixed temporal window, representations of the same slot in adjacent frames serve as positive pairs, while slots from different videos or positions serve as negatives; contrastive learning reinforces temporal consistency.

#### 3. **SlotSSM Transient Memory**

A Mamba-based state space model with block-diagonal $\overline{A}_t$, $\overline{B}_t$, and $C_t$ matrices, where each block is conditioned solely on its corresponding slot input:

$$\mathbf{h}_t^k = \overline{A}(\mathbf{s}_t^k)\mathbf{h}_{t-1}^k + \overline{B}(\mathbf{s}_t^k)\mathbf{s}_t^k$$

**Window Prediction**: Rather than predicting only the next step, SlotSSM predicts static latent representations over a $P = p+q$ step window centered on the current time step (spanning $p$ past and $q$ future steps), jointly learning forward dynamics and backward temporal consistency.

**Core Proposition**: When $k$ objects are visually indistinguishable at time $t$ (i.e., $z_t^{(i)} \approx z_t^{(j)}$), a policy $\pi(a_t|h_t)$ must condition on object-specific history $\mu_t^{(j)}$ to individuate objects — precisely the capability provided by SlotSSM.

#### 4. **Slot-Conditioned Action Decoding**

**Slot Fusion Module**: Integrates the current slot $\mathbf{s}_t^{(j)}$, the predicted next slot $\hat{\mathbf{s}}_{t+1}^{(j)}$, and an oracle sub-goal embedding $\mathbf{g}_t^{(j)}$ to produce a dynamic latent variable $\mathbf{d}_t^{(j)}$.

**Relation Encoder**: Performs cross-attention between slot latents and raw visual features to produce 16 relational tokens $\{\mathbf{r}_t^{(j)}\}_{j=1}^L$, enabling context-aware reasoning over object states and interactions.

**Action Prediction**:
$$\hat{\mathbf{a}}_t \sim P_\theta(\mathbf{a}_t | \{\mathbf{r}_t^{(j)}\}, \{\mathbf{d}_t^{(j)}\}, l)$$

### Loss & Training

The training objective combines the Slot Attention reconstruction loss, temporal contrastive loss, SlotSSM window prediction loss, and a cross-entropy action prediction loss from the VLA head. The current Naive E-SlotSSM variant uses oracle text sub-goal embeddings (e.g., "bowl 1 on plate 3") as progress supervision.

## Key Experimental Results

### Main Results

**Success Rate on LIBERO-Goal (Standard Markovian Tasks)**:

| Method | # Tokens | bowl in stove | bowl on plate | mid drawer | top drawer→bowl | **Avg.** |
|--------|----------|---------------|---------------|------------|-----------------|---------|
| SlotVLA (h=1) | 16 | 45% | 0% | 5% | 0% | 32% |
| SlotVLA (h=8) | 128 | 95% | 90% | 25% | 65% | 75.5% |
| **Naive E-SlotSSM** | 32 | **100%** | **90%** | **45%** | **70%** | **83.0%** |

**Sub-goal Completion Rate on LIBERO-Mem (Non-Markovian POMDP Tasks)**:

| Task | π₀ (h=1) | SlotVLA (h=1) | SlotVLA (h=8) | **Naive E-SlotSSM** |
|------|----------|---------------|---------------|---------------------|
| T1 (1× pick-place) | 50.0% | 0% | 50.0% | **50.0%** |
| T3 (3× pick-place) | 0% | 0% | 0% | **33.3%** |
| T5 (5× pick-place) | 0% | 0% | 0% | **14.3%** |
| T9 (bowl in basket + move basket) | 0% | 0% | 0% | **30%** |
| T10 (bowl in basket + move empty basket) | 0% | 0% | 0% | **20%** |
| **Avg.** | 5.0% | 0% | 5.0% | **14.8%** |

### Ablation Study

Ablations are presented implicitly through **cross-method comparisons across task dimensions**:

| Comparison Axis | SlotVLA (h=8) | Naive E-SlotSSM | Notes |
|----------------|---------------|-----------------|-------|
| General task avg. | 75.5% | **83.0%** | SSM memory +7.5% |
| POMDP task avg. | 5.0% | **14.8%** | Structured memory critical |
| Token efficiency | 128 tokens | 32 tokens | 4× compression |
| Long-horizon repeated tasks (T3–T6) | All 0% | Partial success | Persistent memory effective |

### Key Findings

1. **Dense tokens and naive context extension fail under POMDP**: π₀ (256 tokens) and SlotVLA (h=8, 128 tokens) both average only 5.0% on LIBERO-Mem, demonstrating that simply increasing frame history does not resolve non-Markovian challenges.
2. **Object-centric memory provides a strong inductive bias**: Embodied-SlotSSM achieves approximately 3× improvement on POMDP tasks (5% → 14.8%) through structured slot-based tracking of object identity and state.
3. **Slot visualizations confirm object permanence**: Visualizations show the model maintains consistent attention to target objects (bowls, gripper) throughout grasping and placement sequences.
4. **Token concatenation causes confusion**: Concatenation-based approaches struggle to disambiguate identical visual states with opposing action directions (lift vs. lower), whereas SlotSSM executes correctly.
5. **Absolute performance remains low** (14.8%), primarily constrained by oracle sub-goal dependency.

## Highlights & Insights

1. **Precise and compelling problem formulation**: The paper formally characterizes non-Markovian robotic manipulation as an object-level POMDP, using $P(\mathbf{a}_{t_1}|\mathbf{v}_{1:t_1},l) \neq P(\mathbf{a}_{t_2}|\mathbf{v}_{1:t_2},l)$ with $\mathbf{v}_{t_1} \approx \mathbf{v}_{t_2}$ to precisely specify when the Markovian assumption fails.
2. **Clever benchmark design**: Object identity ambiguity is induced through visually identical bowls and plates; sequential memory demands arise from repeated pick-and-place operations — simple yet incisive.
3. **Window prediction in SlotSSM**: Predicting slot representations over a past-and-future window, rather than a single next step, simultaneously supports forward dynamics prediction and backward temporal consistency via reconstruction.
4. **Sub-goal-aware evaluation**: Extends beyond binary success/failure metrics to enable fine-grained progress assessment (number of sub-goals completed).
5. **Theoretical analysis of computational efficiency**: SlotSSM requires only 32 tokens versus 256 for OpenVLA and 128 for SlotVLA, achieving 4–8× compression.

## Limitations & Future Work

1. **Oracle sub-goal dependency**: Naive E-SlotSSM relies on oracle text sub-goal embeddings (e.g., "bowl 1 on plate 3") and cannot autonomously discover sub-goals — the most significant limitation.
2. **Simulation-only evaluation**: LIBERO-Mem has not been extended to real physical environments; sim-to-real transfer performance remains unknown.
3. **Low absolute performance**: A POMDP task average of 14.8% positions this as a "weak baseline," still far from practical utility.
4. **Fixed slot count**: The fixed design of $K=16$ may not generalize well to scenarios with highly variable numbers of objects.
5. **Limited task diversity**: Tasks are predominantly pick-and-place and object-swapping; more complex long-horizon manipulation (e.g., cooking, assembly) is not addressed.

## Related Work & Insights

- Integrating the cognitive science concept of "object permanence" into robotic manipulation is a meaningful research direction, and SlotSSM provides a computational realization of this principle.
- Compared to MemoryBench and MIKASA-Robo, LIBERO-Mem's distinctive contributions lie in object identity ambiguity and temporal extension stress testing — critical factors for evaluating "genuine memory" versus "short-term perception."
- The block-diagonal design of SlotSSM (based on Mamba) decomposes a global SSM into object-independent local SSMs, constituting an elegant form of structured inductive bias.

## Rating

- Novelty: ⭐⭐⭐⭐ (Both the LIBERO-Mem benchmark and Embodied-SlotSSM framework are novel; the problem formulation and proposed solution are original.)
- Experimental Thoroughness: ⭐⭐⭐ (Multi-baseline comparisons are adequate, but component-level ablations and real-world experiments are absent.)
- Writing Quality: ⭐⭐⭐⭐ (Problem formalization is clear; framework description is thorough; visualizations are persuasive.)
- Value: ⭐⭐⭐⭐ (Opens a new direction for non-Markovian robotic manipulation; the benchmark has long-term research value.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Out of Sight, Out of Track: Adversarial Attacks on Propagation-based Multi-Object Trackers via Query State Manipulation](../../CVPR2026/video_understanding/out_of_sight_out_of_track_adversarial_attacks_on_propagation-based_multi-object_.md)
- [\[AAAI 2026\] MambaMia: State-Space Hierarchical Compression for Hour-Long Video Understanding in Large Multimodal Models](state-space_hierarchical_compression_with_gated_attention_an.md)
- [\[CVPR 2026\] Reconstruction-Guided Slot Curriculum: Addressing Object Over-Fragmentation in Video Object-Centric Learning](../../CVPR2026/video_understanding/reconstruction-guided_slot_curriculum_addressing_object_over-fragmentation_in_vi.md)
- [\[AAAI 2026\] EmoVid: A Multimodal Emotion Video Dataset for Emotion-Centric Video Understanding and Generation](emovid_a_multimodal_emotion_video_dataset_for_emotion-centric_video_understandin.md)
- [\[ICLR 2026\] From Vicious to Virtuous Cycles: Synergistic Representation Learning for Unsupervised Video Object-Centric Learning](../../ICLR2026/video_understanding/from_vicious_to_virtuous_cycles_synergistic_representation_learning_for_unsuperv.md)

</div>

<!-- RELATED:END -->
