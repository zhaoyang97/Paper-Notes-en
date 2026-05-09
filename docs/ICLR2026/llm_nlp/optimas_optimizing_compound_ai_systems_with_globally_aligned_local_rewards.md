---
title: >-
  [Paper Note] Optimas: Optimizing Compound AI Systems with Globally Aligned Local Rewards
description: >-
  [ICLR 2026][LLM/NLP][compound AI systems] This paper proposes Optimas, a framework that maintains a locally aligned reward function (LRF) per component in compound AI systems, enabling independent optimization of heterogeneous components (prompts, model parameters, hyperparameters, model selection), achieving an average improvement of 11.92% across five real-world systems.
tags:
  - ICLR 2026
  - LLM/NLP
  - compound AI systems
  - local reward functions
  - global alignment
  - heterogeneous parameter optimization
  - convergence guarantees
date: 2026-05-08
content_hash: 3575a1d96cd66831
---

# Optimas: Optimizing Compound AI Systems with Globally Aligned Local Rewards

**Conference**: ICLR 2026
**arXiv**: [2507.03041](https://arxiv.org/abs/2507.03041)
**Code**: [https://optimas.stanford.edu/](https://optimas.stanford.edu/)
**Area**: LLM NLP / System Optimization
**Keywords**: compound AI systems, local reward functions, global alignment, heterogeneous parameter optimization, convergence guarantees

## TL;DR
This paper proposes Optimas, a framework that maintains a locally aligned reward function (LRF) per component in compound AI systems, enabling independent optimization of heterogeneous components (prompts, model parameters, hyperparameters, model selection), achieving an average improvement of 11.92% across five real-world systems.

## Background & Motivation
**Background**: Modern AI systems increasingly integrate multiple components—LLMs, retrievers, tool calls, and classical ML models—forming compound AI systems for complex tasks. These systems are highly sensitive to component failures, as errors in one component cascade and amplify downstream.

**Limitations of Prior Work**: (a) Components are non-differentiable across boundaries, precluding end-to-end gradient optimization; (b) configuration spaces are highly heterogeneous—textual prompts, continuous hyperparameters, model weights, and discrete model selections each require fundamentally different optimization strategies; (c) evaluating global performance requires running the full system, making each query costly and data-inefficient.

**Key Challenge**: Existing methods (DSPy for prompt optimization, TextGrad for text-feedback optimization, OPRO for single-step optimization) handle only **a single type** of parameter. Even when each component is independently optimized to its local best, upstream components remain unaware of downstream preferences, leading to potentially suboptimal inter-component coordination. No unified framework exists to simultaneously optimize heterogeneous configurations.

**Core Idea**: Learn a local reward function (LRF) for each component. As long as each LRF remains aligned with the global reward—i.e., locally optimal directions are consistent with global improvement—each component can be independently optimized using its most suitable method, without repeatedly running the full system. This effectively decomposes joint optimization into multiple independent coordinate optimization subproblems.

## Method

### Overall Architecture
The compound system is modeled as a DAG $\mathcal{G}=(\mathcal{C},\mathcal{E})$ containing $K$ components $\{C_k\}_{k=1}^K$. Each component $C_k$ has a configuration policy $\mathbf{v}_k$ (which may be a prompt, hyperparameter, or model weights). The system supports dynamic routing—the inter-component connections $\mathcal{E}(x)$ can adapt per input $x$. Inputs pass through components in topological order to produce output $y=f(x;\mathbf{v})$, with the objective of maximizing $\mathbf{v}^{\star}=\arg\max_{\mathbf{v}} \mathbb{E}_{x\sim\mathcal{D}}[R(x,f(x;\mathbf{v}))]$.

### Key Designs
1. **Local Reward Function (LRF)**:

    - **Function**: For each component $C_k$, learn a scoring function $r_k(x_k,y_k)$ that evaluates the contribution of its output to global performance.
    - **Mechanism**: All LRFs share an LLM backbone $\phi$ with component-specific linear projection heads $h_k$: $r_k(x_k,y_k) = h_k \circ \phi([x_k, y_k])$. The shared backbone ensures scalability; independent heads capture component-specific characteristics.
    - **Alignment Property (Key)**: If $r_k(x_k,y_k^+) \geq r_k(x_k,y_k^-)$, then replacing the component output with $y_k^+$ should yield a higher global reward downstream. Training uses a pairwise log-sigmoid ranking loss: $\mathcal{L}_k = -\mathbb{E}[\log\sigma(r_k(x_k,y_k^+)-r_k(x_k,y_k^-))]$, with preference data constructed via Monte Carlo sampling of downstream outputs.
    - **Design Motivation**: This serves as the theoretical foundation for decomposing global optimization into independent local optimizations—Theorem 4.1 proves that LRFs minimizing this loss necessarily satisfy the alignment property.

2. **Adaptive LRF Update**:

    - **Function**: Lightweight LRF updates after configuration changes to maintain alignment.
    - **Mechanism**: Stage 1 performs offline training of LRFs to convergence; Stage 2 performs online adaptation by sampling small batches of preference data after each configuration update, maintaining a historical buffer for stability.
    - **Design Motivation**: After configuration updates, LRFs become stale—upstream changes alter the global value of the same output, while downstream changes expose the LRF to out-of-distribution inputs. This avoids the costly overhead of retraining LRFs from scratch.

3. **Heterogeneous Component Optimization**:

    - Textual prompts: OPRO is used to select the best candidate prompt ranked by average LRF score.
    - Trainable models (e.g., LLMs): RL algorithms such as PPO are used with the LRF as the critic.
    - Discrete/low-dimensional continuous configurations (model selection, hyperparameters): A probability distribution over LRF scores is constructed and used for sampling-based updates.
    - Validation gating: New configurations are only accepted when global reward improves on a small validation set, preventing cascading errors.

### Theoretical Guarantees
- **Theorem 4.1**: The minimizer of the LRF ranking loss satisfies the local-global alignment property, and maximizing the LRF is equivalent to maximizing the conditional global reward.
- **Theorem 4.2**: Under compactness and unique component optimum conditions, Optimas converges to a component-wise maximum (a direct corollary of classical coordinate maximization results).

## Key Experimental Results

### Main Results (Five Real-World Compound Systems)

| System | Task | Unoptimized | DSPy | TextGrad | **Optimas** | Gain |
|--------|------|-------------|------|----------|-------------|------|
| Amazon Product Recommendation | Acc | 21.21 | 18.18 | 20.88 | **24.24** | +14.3% |
| PubMedQA Medical QA | Acc | 57.46 | 60.26 | 56.96 | **69.13** | +1.8% |
| STaRK-Prime Retrieval | MRR | 40.73 | 41.40 | 41.31 | **50.54** | +22.1% |
| HotpotQA RAG | F1 | 33.80 | 44.90 | 24.86 | **50.48** | +12.4% |
| BigCodeBench Code | Pass | 36.67 | 33.81 | 35.71 | **38.92** | +9.0% |

### Ablation Study

| Configuration | Description |
|---------------|-------------|
| Optimas (full) | All components independently optimized with aligned LRFs; improvements on all 5 systems |
| w/o LRF adaptation | 2–5% drop; alignment degrades as LRF is not updated |
| Global reward only | 3–8% drop; low data efficiency without local signals |
| DSPy (prompt only) | 14.3% degradation on Amazon Recommendation; optimizing a single configuration type is unreliable |

### Key Findings
- **Optimas is the only method that improves performance on all 5 tasks**; DSPy and TextGrad degrade performance on some systems.
- LRF ranking accuracy averages 77.96%, far exceeding LLM Judge (49.52%), indicating that learned LRFs are more reliable than direct LLM scoring.
- Average system evaluation count is 0.71k vs. DSPy's 0.79k, demonstrating higher data efficiency.
- Adaptive LRF updates are critical for long-term performance—without updates, performance degrades noticeably in later optimization stages.
- Joint optimization of heterogeneous configurations is the decisive factor: optimizing only prompts fails on behavior-driven recommendation tasks that require hyperparameter tuning.
- The alignment property of LRFs holds empirically—local improvements consistently yield global gains.
- Bottleneck components differ across systems: the bottleneck in Amazon Recommendation lies in hyperparameters, while in HotpotQA it lies in prompts.

## Highlights & Insights
- A unified framework handles heterogeneous configuration optimization, whereas DSPy/TextGrad are restricted to a single parameter type.
- LRF alignment is backed by rigorous theoretical guarantees (convergence to component-wise optimum).
- The shared backbone + independent head LRF architecture is scalable and memory-efficient.
- Consistent improvements across 5 real-world systems; DSPy degrades by 14.3% on Amazon Recommendation.

## Limitations & Future Work
- Coordinate maximization guarantees only component-wise optimality in non-convex settings, not global optimality.
- Online LRF adaptation still requires a small number of system evaluations and Monte Carlo samples; cost is not zero.
- Experiments involve a limited number of components (2–5); scalability to larger systems remains unvalidated.
- The shared LRF backbone may learn conflicting representations when component input distributions diverge significantly.

## Related Work & Insights
- **DSPy/TextGrad**: Optimize prompts only; do not support heterogeneous configurations; DSPy exhibits unstable performance on some tasks.
- **OPRO**: Single-step generation-based optimization; cannot handle multi-component, multi-step pipelines.
- **LLMSelector**: Performs model routing only; system evaluation cost is 3× that of Optimas.
- **Process Reward Models**: Rely on human annotation or MCTS; Optimas automatically constructs alignment data via preferences.

## Rating
- Novelty: ⭐⭐⭐⭐ (LRF alignment is a novel idea; unifies heterogeneous optimization)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (5 real-world systems + extensive ablations + theoretical analysis)
- Writing Quality: ⭐⭐⭐⭐ (clear structure, rich figures and tables)
- Value: ⭐⭐⭐⭐ (compound AI system optimization is an important research direction)

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] ELLMob: Event-Driven Human Mobility Generation with Self-Aligned LLM Framework](ellmob_event-driven_human_mobility_generation_with_self-aligned_language_models.md)
- [\[AAAI 2026\] Collaborative LLM Numerical Reasoning with Local Data Protection](../../AAAI2026/llm_nlp/collaborative_llm_numerical_reasoning_with_local_data_protection.md)
- [\[CVPR 2026\] PhysVid: Physics Aware Local Conditioning for Generative Video Models](../../CVPR2026/llm_nlp/physvid_physics_aware_local_conditioning_for_generative_video_models.md)
- [\[ACL 2026\] XtraGPT: Context-Aware and Controllable Academic Paper Revision via Human-AI Collaboration](../../ACL2026/llm_nlp/xtragpt_context-aware_and_controllable_academic_paper_revision_via_human-ai_coll.md)
- [\[NeurIPS 2025\] In-Context Learning of Linear Dynamical Systems with Transformers: Approximation Bounds and Depth-Separation](../../NeurIPS2025/llm_nlp/in-context_learning_of_linear_dynamical_systems_with_transformers_approximation_.md)

<!-- RELATED:END -->
