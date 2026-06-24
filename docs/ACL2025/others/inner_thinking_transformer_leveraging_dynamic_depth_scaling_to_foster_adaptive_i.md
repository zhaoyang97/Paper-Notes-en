---
title: >-
  [Paper Note] Inner Thinking Transformer: Leveraging Dynamic Depth Scaling to Foster Adaptive Internal Thinking
description: >-
  [ACL 2025][Dynamic depth] Proposes Inner Thinking Transformer (ITT), which dynamically allocates more computational steps to key tokens without increasing parameters through adaptive token routing and residual thinking connections, achieving implicit deep reasoning. With only 162M parameters, it achieves 96.5% of the performance of a 466M Transformer.
tags:
  - "ACL 2025"
  - "Dynamic depth"
  - "Adaptive computation"
  - "Transformer architecture"
  - "Implicit reasoning"
  - "Token-level routing"
date: 2026-05-08
content_hash: 53e9eda5439b8b4e
---

# Inner Thinking Transformer: Leveraging Dynamic Depth Scaling to Foster Adaptive Internal Thinking

**Conference**: ACL 2025  
**arXiv**: [2502.13842](https://arxiv.org/abs/2502.13842)  
**Code**: None  
**Area**: Others  
**Keywords**: Dynamic depth, Adaptive computation, Transformer architecture, Implicit reasoning, Token-level routing

## TL;DR

Proposes Inner Thinking Transformer (ITT), which dynamically allocates more computational steps to key tokens without increasing parameters through adaptive token routing and residual thinking connections, achieving implicit deep reasoning. With only 162M parameters, it achieves 96.5% of the performance of a 466M Transformer.

## Background & Motivation

Large language models face performance bottlenecks under parameter-constrained conditions, particularly when processing critical tokens that require complex reasoning. Existing methods such as Test-Time Scaling (slow thinking) allocate more computation through inference search but are limited by the accurate generation of key tokens, making small models particularly prone to catastrophic reasoning failures. Layer sharing, recursion, and implicit reasoning methods also fail to flexibly enhance the model's reasoning capability for critical tokens.

By analyzing the Gradient Nuclear Norm (GNN) of GPT-2 on the AQuA dataset, the authors discovered two key phenomena:
- **Simple samples**: GNN decays exponentially in early layers (L0-L2) and remains stably below 3 in intermediate layers (L3-L10).
- **Hard samples**: GNN continuously oscillates across all 12 layers, with sudden spikes at L3, L5, L7, and L9.

This indicates that hard tokens face optimization difficulties in each model layer due to architectural or parameter limitations, inspiring the concept of "Inner Thinking"—interpreting the transformation of each layer as an implicit reasoning step.

## Method

### Overall Architecture

ITT redefines layer computation as implicit thinking steps, which centrally includes three components:
1. **Adaptive Token Routing (ATR)**: Dynamically selects critical tokens requiring deep thinking.
2. **Residual Thinking Connection (RTC)**: Iteratively accumulates results of each step to refine representations.
3. **Thinking Step Encoding (TSE)**: Distinguishes different reasoning stages.

ITT layers are inserted between original model layers at fixed intervals, and all parameters are optimized under a unified language modeling cross-entropy loss.

### Key Designs

1. **Inner Thinking Step**: Decomposes the generation of a single token into a series of internal thinking steps $X^{(t)} = f^{(t)}(x^{(t-1)})$, supporting two scenarios—early exit (intermediate steps are already sufficient) and performance deficit (still insufficient after all steps).

2. **Residual Thinking Connection (RTC)**: The core innovation, which iteratively refines representations through cumulative residual connections. The final output is a weighted accumulation of outputs from all steps: $x^{(t)} = \sum_{i=1}^{t}(f(x^{(i-1)}) \odot \phi^{(i)})$, where $\phi^{(i)}$ is a learnable thinking position encoding. Compared to direct looping, RTC not only achieves deeper thinking but also effectively measures and combines the results of individual steps.

3. **Adaptive Token Routing (ATR)**: Generates an importance score for each token via a linear weight predictor and uses a percentile threshold $P_\rho$ to select the most critical tokens for deep processing. Selected tokens undergo weighted transformation, while unselected ones retain their original representations. Routing weights participate in gradient propagation.

4. **Thinking Step Encoding (TSE)**: A learnable positional encoding $\phi^{(t)}$ used to distinguish different thinking steps and measure the importance of each step.

### Loss & Training

- Uses the standard language modeling cross-entropy loss $\mathbb{L} = \mathbb{L}_{\text{CE}}$
- ITT layers replace every other layer of the original model at fixed intervals
- Trained on 50B tokens (50,000 steps) with a learning rate of 3e-4
- Uses a fixed routing ratio during training (e.g., 70% of tokens participating), which can be elastically adjusted during inference
- Theoretical proof: RTC extends single-step optimization to multi-step optimization, where the error per step decreases by a factor $c$, ensuring stable and efficient convergence while avoiding vanishing or exploding gradients

## Key Experimental Results

### Main Results

| Model Config | Params | FLOPs | Average Accuracy | Comparison |
|---------|--------|-------|-----------|------|
| LLaMA2-162M | 162M | 1.88 | 40.4 | Baseline |
| ITT ×4-162M | 162M | 3.29 | 42.1 | +1.7% |
| LLaMA2-230M | 230M | 2.87 | 41.8 | - |
| ITT ×4-230M | 230M | 3.41 | 43.9 | +2.1% |
| LLaMA2-466M | 466M | 4.92 | 43.6 | - |
| ITT ×4-466M | 466M | 5.84 | 45.3 | +1.7% |

ITT ×4-162M surpasses the 230M Transformer across 11 benchmarks, achieving 96.5% of the performance of the 466M Transformer.

### Ablation Study

| Configuration | Eval PPL | Description |
|------|---------|------|
| ITT ×4 Full | 10.25 | Baseline |
| W/o RTC | 11.02 (+0.77) | Most critical component |
| W/o ATR | 10.44 (+0.19) | Affects efficiency |
| W/o TSE | 10.56 (+0.22) | Loses step information |
| LLaMA2-162M | 11.13 (+1.36) | Original baseline |

### Elastic Inference Experiments

| Selection Ratio | FLOPs | PPL |
|---------|-------|-----|
| 90%, 90%, 90% | 4.42 | 10.27 |
| 70%, 70%, 90% | 4.04 | 10.21 (Optimal) |
| 70%, 70%, 70% (Training) | 3.85 | 10.52 |
| 50%, 50%, 50% | 3.29 | 10.47 |

### Key Findings

- **Data Efficiency**: ITT requires only 56.8% of the training data to match the performance of LLaMA2-162M, saving 43.2% of the training budget.
- **Computational Efficiency**: 3-step thinking requires only 84% of the computation of Loop, which drops to 70% for 4-step thinking.
- **Elastic Thinking**: Token selection ratios can be flexibly adjusted during inference to balance performance and efficiency.
- **Routing Visualization**: Roughly 30%-50% of tokens undergo iterative thinking, with task-critical tokens (verbs, semantic key points) being more likely to receive multi-step thinking; complementary thinking patterns are observed across consecutive steps.

## Highlights & Insights

1. **Conceptual Innovation**: Reinterprets Transformer layer computation as "internal thinking steps," elegantly linking implicit reasoning with dynamic computational allocation.
2. **Extremely High Parameter Efficiency**: Significantly improves performance without introducing any extra parameters, enabling a 162M model to attain the capability level of a 466M model.
3. **Elastic Inference**: Computational allocation can be flexibly adjusted after training to suit various deployment scenarios.
4. **Complementarity of Routing**: The model spontaneously learns an alternating strategy of "deep thinking" and "breadth compensation."
5. **Theoretical Support**: Demonstrates that multi-step optimization converges more easily compared to single-step mapping.

## Limitations & Future Work

1. Utilizing a fixed routing pattern during training may limit dynamic adaptation to diverse token complexities.
2. Experiments were only verified at the scale of 162M-466M parameters; larger, scaled-up models might introduce new architectural interactions.
3. RTC incurs additional memory overhead during backpropagation, requiring optimization for industrial deployment.
4. The thinking step encoding is relatively simple; more sophisticated temporal modeling might further enhance reasoning depth.
5. Integration with explicit reasoning methods (such as CoT) is yet to be fully explored.

## Related Work & Insights

- **Recurrent Computation**: Includes deep recurrent schemes like LSTM, Universal Transformer, and Loop Transformer.
- **Dynamic Computation Allocation**: MoE, Early Exit, and Parameter Sharing to reduce redundant calculations.
- The token-level dynamic depth allocation scheme in this paper may inspire finer-grained computational resource management, particularly regarding potential integration with MoE.
- The RTC mechanism resembles residual learning in iterative optimization and can be generalized to other tasks requiring step-by-step refinement.

## Rating

- Novelty: ⭐⭐⭐⭐ The concept of viewing layer computation as thinking steps is novel, though the dynamic routing idea draws inspiration from prior work.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across three scales, detailed ablation, and elastic inference analysis, though lacking large-scale validation.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, smooth narrative, and intuitive, abundant figures and tables.
- Value: ⭐⭐⭐⭐ Highly practical in parameter-constrained scenarios; the elastic inference feature is valuable for deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Tree-of-Debate: Multi-Persona Debate Trees Elicit Critical Thinking for Scientific Comparative Analysis](tree-of-debate_multi-persona_debate_trees_elicit_critical_thinking_for_scientifi.md)
- [\[ACL 2025\] Dolphin: Moving Towards Closed-loop Auto-research through Thinking, Practice, and Feedback](dolphin_moving_towards_closed-loop_auto-research_through_thinking_practice_and_f.md)
- [\[ACL 2025\] Explaining Matters: Leveraging Definitions and Semantic Expansion for Sexism Detection](explaining_matters_leveraging_definitions_and_semantic_expansion_for_sexism_dete.md)
- [\[ACL 2025\] Tuna: Comprehensive Fine-grained Temporal Understanding Evaluation on Dense Dynamic Videos](tuna_temporal_understanding.md)
- [\[ICCV 2025\] AdaptiveAE: An Adaptive Exposure Strategy for HDR Capturing in Dynamic Scenes](../../ICCV2025/others/adaptiveae_an_adaptive_exposure_strategy_for_hdr_capturing_i.md)

</div>

<!-- RELATED:END -->
