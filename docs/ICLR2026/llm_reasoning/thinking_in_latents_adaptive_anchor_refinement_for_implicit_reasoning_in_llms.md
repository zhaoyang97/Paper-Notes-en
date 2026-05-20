---
title: >-
  [Paper Note] Thinking in Latents: Adaptive Anchor Refinement for Implicit Reasoning in LLMs
description: >-
  [ICLR 2026][LLM Reasoning][latent-space reasoning] This paper proposes AdaAnchor, a latent-space reasoning framework that appends learnable anchor vectors to input embeddings and refines their states through iterative fo…
tags:
  - "ICLR 2026"
  - "LLM Reasoning"
  - "latent-space reasoning"
  - "adaptive stopping"
  - "anchor vectors"
  - "CoT compression"
  - "implicit computation"
date: 2026-05-08
content_hash: 7953455c92088609
---

# Thinking in Latents: Adaptive Anchor Refinement for Implicit Reasoning in LLMs

**Conference**: ICLR 2026
**arXiv**: [2603.15051](https://arxiv.org/abs/2603.15051)  
**Code**: None  
**Area**: LLM Reasoning / Efficiency
**Keywords**: latent-space reasoning, adaptive stopping, anchor vectors, CoT compression, implicit computation

## TL;DR

This paper proposes AdaAnchor, a latent-space reasoning framework that appends learnable anchor vectors to input embeddings and refines their states through iterative forward passes to achieve "silent thinking." An adaptive stopping mechanism based on anchor stability dynamically allocates computation according to instance difficulty. On mathematical reasoning benchmarks, AdaAnchor achieves up to 5% higher accuracy and 48–60% fewer average steps compared to fixed-step latent reasoning, while reducing output tokens by 92–93% relative to CoT.

## Background & Motivation

**Background**: Chain-of-Thought (CoT) reasoning has become the standard paradigm for eliciting multi-step reasoning in LLMs, yielding significant gains on mathematical reasoning tasks. However, generating long intermediate reasoning chains increases output length and inference cost, which is particularly pronounced in high-concurrency deployment scenarios.

**Limitations of Prior Work**: Latent-space reasoning approaches (e.g., Coconut, iCoT distillation) attempt to shift reasoning into hidden representations and output answers only, but most rely on a **fixed number of latent iteration steps**—introducing a hyperparameter that requires tuning across models and datasets, wasting compute on easy instances and potentially undercomputing on hard ones.

**Key Challenge**: Token-level CoT = high accuracy + high token cost; latent-space reasoning = low token cost + inflexible fixed steps. The core challenge is achieving **instance-level adaptive compute allocation** while keeping output token counts low.

**Key Insight**: The authors propose using a set of learnable "anchor vectors" as explicit latent reasoning states, refined through repeated forward passes, with cosine distance changes between anchor states monitored as a convergence signal—stopping upon convergence to naturally allocate compute on demand.

**Distinction from Soft Prompts / Prefix Tuning**: Conventional soft prompts are fixed at inference time; AdaAnchor's anchor vectors are **iteratively rewritten** during inference, acting as persistent latent memory across iterations.

**Practical Significance**: In deployment scenarios, output token cost is highly correlated with inference latency. Shifting reasoning from token space to latent space, combined with adaptive step control, yields lower serving cost and latency.

## Method

### Overall Architecture

AdaAnchor introduces $m$ learnable anchor vectors $A^{(0)} \in \mathbb{R}^{m \times d}$ on top of a base autoregressive Transformer $f_\theta$. The inference pipeline consists of three stages:

1. **Anchor-augmented input construction**: Anchor vectors are projected via $P(\cdot)$ into the same space as token embeddings and prepended to the input embedding sequence, forming the augmented input $E^{(t)} = [P(A^{(t)}); \text{Emb}(x)]$.
2. **Iterative anchor refinement**: The augmented input is passed through the frozen backbone LM for a forward pass; the representations at anchor positions are extracted from the output hidden states as updated anchors via smooth update $A^{(t+1)} \leftarrow (1-\beta)A^{(t)} + \beta H^{(t)}_{1:m}$, repeated up to $K_{\max}$ times.
3. **Answer-only decoding**: After iteration terminates, the final anchor states combined with the original input are used to decode the answer only, without generating intermediate reasoning tokens.

### Key Design 1: Iterative Anchor Refinement

- **Function**: At each iteration $t$, the current anchor $A^{(t)}$ is projected and prepended to the input embeddings before a full forward pass; the hidden states at anchor positions are extracted as candidate updated anchors $A_{\text{new}}^{(t+1)}$, then combined with the old anchors via exponential moving average.
- **Mechanism**:
  $$H^{(t)} = f_\theta(E^{(t)}) \in \mathbb{R}^{(m+n) \times d}$$
  $$A^{(t+1)} = (1-\beta)\,A^{(t)} + \beta\,H^{(t)}_{1:m}$$
  where $\beta \in (0,1]$ is the smoothing coefficient. Setting $\beta=1$ reduces to direct overwriting; smaller $\beta$ yields smoother anchor evolution and more stable convergence.
- **Design Motivation**: Anchor vectors act as a "compressed reasoning bottleneck"—all multi-step reasoning information must be encoded into a compact $m \times d$ state, analogous to the information bottleneck in autoencoders, compelling the model to extract essential reasoning features rather than redundant narration.

### Key Design 2: Anchor-Stability-Based Adaptive Stopping

- **Function**: After each iteration, the change between consecutive anchor states is computed; iteration terminates early when the change falls below threshold $\tau$ for $s$ consecutive steps.
- **Mechanism**: Change is measured via cosine distance between anchor means:
  $$\bar{a}^{(t)} = \frac{1}{m}\sum_{i=1}^{m} a_i^{(t)}, \quad \Delta^{(t)} = 1 - \cos(\bar{a}^{(t)}, \bar{a}^{(t-1)})$$
  The stopping condition is:
  $$T = \min\{t : \Delta^{(t-j)} < \tau,\;\forall j \in \{0,\ldots,s-1\}\}$$
  requiring $s$ consecutive stable steps to prevent spurious early stopping from a single fluctuation.
- **Design Motivation**: When anchors cease to change significantly, latent reasoning has converged near a fixed point and further iterations yield no useful computation. This heuristic requires no additional stopping controller—zero extra parameters—as the stability check directly reuses existing anchor states.

### Key Design 3: Training Strategy

- **Function**: Backbone LM weights are frozen; only AdaAnchor-specific components (anchor embeddings + projection layer) and LoRA adapters are trained for 20 epochs using AdamW ($\text{lr}=1\times10^{-4}$, weight decay=$1\times10^{-2}$, gradient accumulation over 16 steps).
- **Mechanism**: In addition to the answer-only primary loss, an auxiliary anchor-alignment objective is introduced, using coarse-grained segmentation of rationale text as a weak supervision signal to guide anchor vectors toward meaningful intermediate reasoning representations.
- **Design Motivation**: Training with answer-only supervision alone may cause anchors to degenerate into meaningless noise. Weak alignment signals derived from coarse rationale segmentation guide anchors toward structured intermediate reasoning states without requiring token-level reasoning chain annotations.

## Key Experimental Results

### Main Results (Table 2)

| Model | Method | GSM8K Acc(%) | GSM8K Avg Tok | SVAMP Acc(%) | SVAMP Avg Tok | MultiArith Acc(%) | MultiArith Avg Tok |
|-------|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen2.5-1.5B | No CoT | 13.0 | 2.16 | 42.0 | 2.34 | 22.3 | 2.41 |
| Qwen2.5-1.5B | CoT | 20.0 | 28.27 | 59.3 | 29.09 | 34.3 | 30.2 |
| Qwen2.5-1.5B | iCoT | 12.23 | 2.36 | 48.5 | 2.04 | 28.56 | 1.66 |
| Qwen2.5-1.5B | AdaAnchor (K=8) | 16.0 | 2.73 | 50.5 | 2.12 | 27.6 | 2.34 |
| **Qwen2.5-1.5B** | **AdaAnchor (adaptive)** | **16.0** | **2.17** | **55.2** | **2.23** | **29.4** | **2.16** |
| Llama-3.2-1B | No CoT | 10.5 | 2.98 | 38.2 | 2.10 | 20.56 | 2.08 |
| Llama-3.2-1B | CoT | 23.2 | 25.4 | 57.8 | 28.21 | 43.33 | 28.0 |
| Llama-3.2-1B | iCoT | 11.7 | 2.25 | 54.2 | 2.43 | 30.84 | 2.12 |
| Llama-3.2-1B | AdaAnchor (K=8) | 14.0 | 2.89 | 52.0 | 2.13 | 28.31 | 2.48 |
| **Llama-3.2-1B** | **AdaAnchor (adaptive)** | **17.2** | **2.45** | **53.4** | **2.8** | **32.44** | **2.57** |

### Adaptive Stopping vs. Fixed Steps

| Model | Dataset | Fixed (K=8) Acc | Adaptive Acc | Fixed Avg Steps | Adaptive Avg Steps | Steps Reduced |
|-------|---------|:---:|:---:|:---:|:---:|:---:|
| Qwen2.5-1.5B | GSM8K | 16.0% | 16.0% | 8.0 | 3.23 | 59.6% |
| Qwen2.5-1.5B | SVAMP | 50.5% | 55.2% | 8.0 | 4.12 | 48.5% |
| Qwen2.5-1.5B | MultiArith | 27.6% | 29.4% | 8.0 | 3.82 | 52.3% |
| Llama-3.2-1B | GSM8K | 14.0% | 17.2% | 8.0 | 3.5 | 56.3% |
| Llama-3.2-1B | SVAMP | 52.0% | 53.4% | 8.0 | 3.1 | 61.3% |
| Llama-3.2-1B | MultiArith | 28.31% | 32.44% | 8.0 | 3.5 | 56.3% |

### Key Findings

- **Adaptive stopping consistently outperforms fixed steps**: On Llama-3.2-1B, GSM8K accuracy improves from 14.0% to 17.2% (+3.2%) and MultiArith from 28.31% to 32.44% (+4.13%), while reducing average steps by 48–61%. This indicates that fixed 8 steps causes "over-refinement" on easy instances that is actually detrimental.
- **Substantial token savings but accuracy gap remains**: Output tokens are reduced by ~92–93% compared to CoT (~2 tokens vs. ~28 tokens), yet accuracy remains notably lower (e.g., 17.2% vs. 23.2% on GSM8K), reflecting a different accuracy–efficiency tradeoff.
- **Convergence step distribution is strongly right-skewed**: Most instances converge and stop within 2–4 steps; only a small fraction of hard instances require close to the maximum 8 steps, validating that the adaptive mechanism does differentiate compute allocation by instance difficulty.
- **Diminishing returns for fixed steps**: In ablation experiments, accuracy increases substantially as $K$ grows from 1 to 4, but the gain diminishes significantly from 4 to 8, supporting the premise that running the maximum number of steps is unnecessary in general.

## Highlights & Insights

- **Practicality of "silent reasoning"**: Shifting reasoning entirely to latent space with only ~2 output tokens has direct cost-saving value for high-concurrency LLM serving (e.g., API serving), where output tokens are a primary component of inference cost.
- **Anchor vectors as iteratively writable soft prompts**: The paper creatively extends the "fixed soft prompt" of prefix tuning into an "iteratively writable latent state." The resulting design is conceptually clean and elegantly implemented—after each forward pass, output hidden states are written back into the anchor vectors, yielding recurrent-like information propagation without modifying the Transformer architecture.
- **Simplicity of convergence detection**: No additional stopping policy network is needed; effective adaptive stopping is achieved solely via cosine distance monitoring with a patience mechanism. This design incurs zero extra parameters, zero additional training overhead, and is transferable to any iterative latent reasoning method.
- **Information bottleneck perspective**: The $m$ anchor vectors form an information bottleneck, compelling the model to compress all reasoning-relevant information into a fixed-dimensional representation—echoing research directions in Markovian and compressed reasoning.

## Limitations & Future Work

1. **Persistent accuracy gap relative to CoT**: Even in the best setting (Llama-3.2-1B, SVAMP 53.4% vs. CoT 57.8%), latent reasoning accuracy remains below explicit CoT, indicating that current anchor refinement cannot fully replicate the expressive capacity of token-level reasoning.
2. **Stopping strategy relies on hand-crafted thresholds**: Both $\tau$ and patience $s$ are manually selected hyperparameters that may cause premature or delayed stopping on out-of-distribution data or atypical inputs. The authors acknowledge that replacing this with a learned stopping strategy is an important direction for future work.
3. **Validation limited to small models and math tasks**: Experiments use only 1B–1.5B scale models and three mathematical datasets; generalization to larger models (7B+) and broader reasoning types (logical reasoning, code generation) remains unvalidated.
4. **Anchor semantics are uninterpretable**: The semantic content of anchor vectors cannot be directly read out—it is unclear what the model "considers" at each iteration, and no interpretability analysis is provided.
5. **Inference latency not thoroughly analyzed**: Although output tokens are reduced, each iteration requires a full forward pass (averaging 3–4 passes), so actual wall-clock latency may not be lower than CoT. The paper does not report wall-clock time comparisons.

## Related Work & Insights

### vs. Coconut (Hao et al., 2024)
Coconut also performs reasoning in continuous latent space by feeding hidden states directly back into model inputs. AdaAnchor differs in three respects: (1) it uses explicit anchor vectors as latent states rather than reusing the last token's hidden state, yielding a more structured representation; (2) it introduces adaptive stopping, eliminating the need for a fixed step count; (3) the smooth EMA update of anchors enhances convergence stability.

### vs. iCoT / Implicit CoT (Deng et al., 2023/2024)
iCoT uses knowledge distillation to internalize the reasoning process—progressively removing token-level reasoning steps so the model learns to skip intermediate steps and directly output answers. AdaAnchor is more explicit: anchor vectors are externally maintained observable states (albeit semantically opaque), providing a clear iterative refinement framework. AdaAnchor (adaptive) outperforms iCoT in most settings (e.g., SVAMP: 55.2% vs. 48.5%).

### vs. Pause Tokens (Goyal et al., 2024)
Pause Tokens insert `<pause>` tokens into inputs to encourage internal computation but do not iteratively update any state. AdaAnchor's key advancement is **iterativity**—anchors are not processed in a single forward pass but refined repeatedly. This introduces temporal depth into latent reasoning, analogous to the Universal Transformer concept.

## Rating

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Novelty | ★★★★☆ | The combination of iterative anchor refinement and convergence-based adaptive stopping is novel, though latent-space reasoning itself has substantial prior work |
| Technical Quality | ★★★☆☆ | The method is clearly designed and well-motivated, but experimental scale is limited (only 1B-scale models, three math datasets) with no wall-clock time comparisons or validation on larger models |
| Practical Value | ★★★★☆ | 92–93% token reduction has direct value for LLM serving; the adaptive stopping mechanism is simple and generalizable, though the accuracy gap constrains scenarios where it could replace CoT |
| Writing Quality | ★★★★☆ | The paper is well-structured with clear Algorithm 1 pseudocode and complete notation, though some experimental analyses could be deeper |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Is It Thinking or Cheating? Detecting Implicit Reward Hacking by Measuring Reasoning Effort](is_it_thinking_or_cheating_detecting_implicit_reward_hacking_by_measuring_reason.md)
- [\[ICLR 2026\] Why is Your Language Model a Poor Implicit Reward Model?](why_is_your_language_model_a_poor_implicit_reward_model.md)
- [\[AAAI 2026\] LLMs for Game Theory: Entropy-Guided In-Context Learning and Adaptive CoT Reasoning](../../AAAI2026/llm_reasoning/llms_for_game_theory_entropy-guided_in-context_learning_and_adaptive_cot_reasoni.md)
- [\[ICLR 2026\] GeoGramBench: Benchmarking the Geometric Program Reasoning in Modern LLMs](geogrambench_benchmarking_the_geometric_program_reasoning_in_modern_llms.md)
- [\[ICLR 2026\] Are Reasoning LLMs Robust to Interventions on Their Chain-of-Thought?](are_reasoning_llms_robust_to_interventions_on_their_chain-of-thought.md)

</div>

<!-- RELATED:END -->
