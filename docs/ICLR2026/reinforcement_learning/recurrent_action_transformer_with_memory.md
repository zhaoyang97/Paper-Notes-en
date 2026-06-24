---
title: >-
  [Paper Note] Recurrent Action Transformer with Memory
description: >-
  [ICLR2026][Reinforcement Learning][Offline Reinforcement Learning] RATE (Recurrent Action Transformer with Memory) partitions trajectories into fixed-length segments and uses a set of learnable memory embeddings to pass historical information across segments. It introduces a cross-attention-based "Memory Retention Valve" (MRV) to control whether to retain or overwrite memories. This approach significantly outperforms Decision Transformer on memory-intensive offline RL tasks s…
tags:
  - "ICLR2026"
  - "Reinforcement Learning"
  - "Offline Reinforcement Learning"
  - "Decision Transformer"
  - "Memory Mechanism"
  - "Segment-level Recurrence"
  - "POMDP"
date: 2026-05-08
content_hash: 5d413a8aa7e664fc
---

# Recurrent Action Transformer with Memory

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=kByN4v0M3e](https://openreview.net/forum?id=kByN4v0M3e)  
**Code**: https://sites.google.com/view/rate-model/  
**Area**: Reinforcement Learning / Offline RL  
**Keywords**: Offline Reinforcement Learning, Decision Transformer, Memory Mechanism, Segment-level Recurrence, POMDP

## TL;DR
RATE (Recurrent Action Transformer with Memory) partitions trajectories into fixed-length segments and uses a set of learnable memory embeddings to pass historical information across segments. It introduces a cross-attention-based "Memory Retention Valve" (MRV) to control whether to retain or overwrite memories. This approach significantly outperforms Decision Transformer on memory-intensive offline RL tasks such as ViZDoom, T-Maze, Memory Maze, and POPGym, while remaining competitive on standard Atari/MuJoCo benchmarks.

## Background & Motivation
**Background**: In offline reinforcement learning, methods like Decision Transformer (DT) treat trajectories as sequences of $(Return, Observation, Action)$ and use GPT-style autoregressive modeling to predict actions. This avoids value function estimation and demonstrates strong performance.

**Limitations of Prior Work**: Self-attention has quadratic complexity, and DT uses a fixed and limited context window. Once a critical cue (e.g., a "left or right" hint at the start of a maze) slides out of the context window, DT can no longer access it, leading to failure in long-range credit assignment and sparse-reward POMDP tasks.

**Key Challenge**: The central conflict is how to enable the model to remember ancient, sparse, yet critical information without infinitely extending the context window, which is restricted by quadratic complexity and training instability.

**Goal**: To equip offline RL Transformers with a memory mechanism that extends the effective context $K_{\text{eff}} = N \times K$ far beyond the single-segment attention limit, reliably preserving important information in highly sparse tasks without being overwritten by subsequent noise.

**Key Insight**: The authors draw inspiration from memory-augmented Transformers in NLP (e.g., memory embeddings in RMT, hidden state caching in Transformer-XL). However, they note that RL inputs are multimodal $(o, a, R)$ and often highly sparse. Naively passing memory forward causes error accumulation or the overwriting of crucial information.

**Core Idea**: A triple combination of "Memory Embeddings + Hidden State Caching + a learnable Memory Retention Valve (MRV)" for segment-level recurrence. The MRV allows old memory $M_n$ to "scrutinize" new memory $M_{n+1}$, deciding what to keep and what to overwrite to prevent sparse long-range information from being washed out.

## Method

### Overall Architecture
RATE processes a trajectory $\tau_{0:T-1}$ of length $T$, where each timestep is a triplet $(R_t, o_t, a_t)$. Modality-specific encoders transform these into $\tilde R_t, \tilde o_t, \tilde a_t$, and the sequence is divided into $N = T // K$ non-overlapping segments $S_n$ of length $K$. The model **processes segments sequentially**: each segment is sandwiched between two copies of the same memory embedding $M_n \in \mathbb{R}^{m\times d}$ ($m$ memory tokens of dimension $d$), resulting in $\tilde S_n = \mathrm{concat}(M_n, S_n, M_n)$. This is fed into a Transformer to output action predictions $\hat a_n$ and updated memories $M_{n+1}$. The updated memory $M_{n+1}$ is refined by the **MRV** before being passed to the next segment. Segment-to-segment hidden state caching (following Transformer-XL) is also used, where previous activations serve as extended KV context without gradient backpropagation. Thus, information flows across segments via two channels: trainable memory embeddings and non-trainable cached hidden states.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Trajectory (R,o,a)<br/>Modality Encoding + Split into N segments"] --> B["Double-copy Memory Embeddings<br/>Sandwich each segment with M_n"]
    B --> C["Hidden State Caching<br/>Reuse previous activations as extended KV"]
    C --> D["Transformer Processing Segment<br/>Outputs actions â_n and new memory M_{n+1}"]
    D --> E["Memory Retention Valve (MRV)<br/>Cross-attention to filter new memory"]
    E -->|M_{n+1} passed to next segment| B
    E --> F["Segment-wise action output â_n"]
```

### Key Designs

**1. Double-copy Memory Embeddings: Simultaneous "Read" and "Write" ports**

A limitation of DT is its inability to access cues outside the context window. The authors use learnable tokens $M_n$ as dedicated historical storage. Crucially, each segment is sandwiched: $\tilde S_n = \mathrm{concat}(M_n, S_n, M_n) \in \mathbb{R}^{(3K+2m)\times d}$. Under causal self-attention, the **prefix $M_n$ acts as a "Read" port**, allowing every token in the segment to attend to incoming history. The **suffix $M_n$ acts as a "Write" port**, placed after the segment in causal order, allowing the final layers to attend to $S_n$ and encode new information into $M_{n+1}$. Without the prefix, memory is unreadable; without the suffix, it cannot be updated. This structure is fundamental to RATE’s recurrence.

**2. Hidden State Caching: A continuous information channel**

Discrete memory tokens alone are insufficient. Following Transformer-XL, hidden activations from previous segments are cached as fixed key-value contexts for the current segment. This provides a second channel—while trainable $M_n$ carries "task-level critical cues," cached states carry "continuous, dense near-term context." Ablations reveal they serve different purposes: dense feedback tasks (ViZDoom) rely more on caching, while sparse decision tasks (T-Maze) rely more on memory embeddings.

**3. Memory Retention Valve (MRV): Gating to prevent overwriting**

The core innovation: naively passing $M_{n+1}$ leads to error accumulation or the overwriting of sparse cues by irrelevant updates. MRV is a cross-attention module where the **old memory $M_n$ acts as the Query and the new memory $M_{n+1}$ acts as the Key/Value**:

$$\mathrm{MRV}(M_n, M_{n+1}) = \mathrm{FFN}\big(\mathrm{MultiHead}(Q=M_n,\, K=M_{n+1},\, V=M_{n+1})\big)$$

Intuitively, $M_n$ "scores" each new token based on its existing content to decide what to retain. Unlike the static recurrence in Transformer-XL, MRV is content-aware. The authors provide a **Retention Theorem**: assuming row-wise $\ell_2$ normalization and $\alpha$-alignment, the memory retention ratio after an MRV update is at least $\big(1 - \sqrt{2(1-\frac{\alpha}{m})}\big)$:

$$\|M_{n+1} - M_n\|_F \le \sqrt{2\big(1 - \tfrac{\alpha}{m}\big)}\cdot \|M_n\|_F$$

**Mechanism in Action: T-Maze**
In T-Maze ($T=8$), an agent receives a 1-bit cue $o_0$ at step 0 (deciding the turn at the goal). DT fails once $o_0$ slides out of its window. RATE splits the sequence into segments. $o_0$ is written into $M_1$. The MRV preserves this sparse cue through subsequent updates. Attention maps confirm RATE's tokens "read" $o_0$ even in late segments, whereas DT loses it entirely.

### Loss & Training
The objective is segment-wise action supervision loss $L(a_n, \hat a_n)$ (return-conditioned sequence modeling). Memory $M_0$ is initialized as $\mathcal N(0,1)$ and rolled out recurrently by segment. Cached states do not receive gradients, while MRV weights are learned.

## Key Experimental Results

### Main Results
Evaluated on memory-intensive tasks (ViZDoom-Two-Colors, T-Maze, Minigrid-Memory, Memory Maze, POPGym) and standard RL (Atari, MuJoCo) against baselines like DT, RMT, TrXL, LSDT, DMamba, and CQL.

| Task | Metric | RATE | Representative Baseline | Note |
|--------|------|------|----------|------|
| Memory Maze (9×9, 1000 steps) | Mean Return ± SEM | **7.64±0.41** | DT: 6.83 / RMT: 7.27 | Dataset mean: 4.69 |
| POPGym (All 48) | Normal. Mean Score | **9.5** | DT: 5.8 / BC-LSTM: 9.0 | Highest overall |
| POPGym (Memory subset) | Normal. Mean Score | **0.5** | DT: −3.5 / BC-LSTM: −0.2 | Only positive score |
| POPGym (Reactive subset) | Normal. Mean Score | 9.1 | DT: 9.3 / BC-LSTM: 9.1 | Competitive |
| T-Maze Extrapolation | Success Rate | 100% up to 9600 steps | DT drops to ~50% early | Trained on ≤900 steps |
| MuJoCo D4RL (9 datasets) | Normal. Mean Score | 78.5 | DT: 74.7 | Competitive on standard tasks |

RATE maintains 100% success on T-Maze and extrapolates reliably up to 28,800 tokens. TrXL performance is close to DT, suggesting **hidden state caching alone cannot retrieve long-range sparse information**, highlighting the necessity of MRV.

### Ablation Study (Noising components at inference)

| Configuration | Key Observation | Note |
|------|---------|------|
| RATE Full | 100% T-Maze / Highest ViZDoom | Full model |
| Noised Memory $M$ (T-Maze) | Success drops to ~50% | Agent navigates but loses the initial cue |
| Noised Hidden Cache (ViZDoom) | High sensitivity | Dense feedback tasks rely more on caching |

### Key Findings
- **Dual-channel synergy**: Memory embeddings $M$ are critical for "sparse, discrete decision points" (T-Maze). Hidden state caching is critical for "dense, continuous feedback" (ViZDoom). They are complementary rather than redundant.
- **MRV is the differentiator**: TrXL (cache only, no MRV) fails at T-Maze similarly to DT, proving static recurrence cannot retain sparse cues. MRV's content-aware filtering is key.
- **No degradation on simple tasks**: In reactive benchmarks, RATE matches specialized methods, proving the memory mechanism is "beneficial if needed, harmless otherwise."

## Highlights & Insights
- **Read/Write separation via double-copying**: Exploiting causal attention to turn a single memory bank into read (prefix) and write (suffix) ports is a simple yet elegant structural solution for recurrence.
- **MRV as content-aware gating**: Using old memory as a Query to scrutinize new information provides a principled way to decide on "overwriting" vs. "retention," supported by theoretical bounds.
- **Transferability**: The "segment-level recurrence + gated memory update" framework is not limited to RL and could benefit any sequence task requiring long-term retention of sparse signals.

## Limitations & Future Work
- **$\alpha$-alignment Assumption**: The retention theorem assumes a specific alignment condition observed empirically but not guaranteed for all weights.
- **Hyperparameter Sensitivity**: Memory size $m$ and segment length $K$ require tuning (detailed in Appendix F/G).
- **Offline Constraint**: Currently validated in the offline setting; performance in online interaction/exploration remains to be explored.
- **Scalability**: Future work could explore per-token retention budgets or adaptive memory capacities.

## Related Work & Insights
- **vs. Decision Transformer (DT)**: DT is lost once cues exit the window; RATE extends effective context to $N\times K$.
- **vs. RMT**: RATE adds the MRV gate to RMT's basic memory embedding idea, enhancing stability in sparse tasks.
- **vs. Transformer-XL (TrXL)**: TrXL's static caching is insufficient for sparse long-range cues; RATE combines caching with gated trainable memory.
- **vs. SSMs (e.g., Mamba)**: While SSMs struggle with high-sparsity long sequences, RATE’s attention-based memory handles interpolation and extrapolation more effectively.

## Rating
- Novelty: ⭐⭐⭐⭐ MRV + double-copy embeddings represent solid structural innovations with theoretical grounding.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 5 memory-intensive categories plus standard benchmarks with noise-based ablations and extrapolation analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear mechanics and strong visual evidence (attention maps).
- Value: ⭐⭐⭐⭐ Provides a high-capacity architecture for long-term memory in offline RL with transferable insights.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Composition of Memory Experts for Diffusion World Models](composition_of_memory_experts_for_diffusion_world_models.md)
- [\[ICLR 2026\] Peak-Return Greedy Slicing: Subtrajectory Selection for Transformer-based Offline RL](peak-return_greedy_slicing_subtrajectory_selection_for_transformer-based_offline.md)
- [\[ICLR 2026\] ELMUR: External Layer Memory with Update/Rewrite for Long-Horizon RL Problems](elmur_external_layer_memory_with_updaterewrite_for_long-horizon_rl_problems.md)
- [\[ICLR 2026\] DEAS: DEtached value learning with Action Sequence for Scalable Offline RL](deas_detached_value_learning_with_action_sequence_for_scalable_offline_rl.md)
- [\[ICLR 2026\] The State of Reinforcement Finetuning for Transformer-based Agents](the_state_of_reinforcement_finetuning_for_transformer-based_agents.md)

</div>

<!-- RELATED:END -->
