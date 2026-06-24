---
title: >-
  [Paper Note] Recurrent Knowledge Identification and Fusion for Language Model Continual Learning
description: >-
  [ACL 2025][LLM (Other)][Continual Learning] Proposes the Recurrent-KIF continual learning framework, which dynamically estimates parameter importance distribution via an inner-outer loop iterative mechanism and utilizes importance-based binary masks for knowledge fusion, effectively mitigating catastrophic forgetting and promoting knowledge transfer.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Continual Learning"
  - "Knowledge Fusion"
  - "Parameter Importance"
  - "Catastrophic Forgetting"
  - "Model Merging"
date: 2026-05-08
content_hash: 2da2840fbc1b70a5
---

# Recurrent Knowledge Identification and Fusion for Language Model Continual Learning

**Conference**: ACL 2025  
**arXiv**: [2502.17510](https://arxiv.org/abs/2502.17510)  
**Code**: [https://github.com/WoodScene/Recurrent_KIF](https://github.com/WoodScene/Recurrent_KIF)  
**Area**: Continual Learning / LLM Efficiency  
**Keywords**: Continual Learning, Knowledge Fusion, Parameter Importance, Catastrophic Forgetting, Model Merging

## TL;DR

Proposes the Recurrent-KIF continual learning framework, which dynamically estimates parameter importance distribution via an inner-outer loop iterative mechanism and utilizes importance-based binary masks for knowledge fusion, effectively mitigating catastrophic forgetting and promoting knowledge transfer.

## Background & Motivation
**Background**: Continual learning (CL) is a crucial capability for deploying LLMs in dynamic environments. PEFT-based model mixing methods (model ensemble and model merging) have become the mainstream.

**Limitations of Prior Work**: Existing methods rely on static parameter importance estimation—the importance scores of historical tasks are no longer updated after training. As model parameters evolve, the truncation error of Taylor expansion increases, leading to inaccurate importance estimation.

**Key Challenge**: It is necessary to balance knowledge transfer (KT) and catastrophic forgetting (CF), yet static importance analysis fails to accurately reflect the true importance of parameters under the new model state.

**Goal**: To achieve dynamic updates of parameter importance and multi-round knowledge fusion, rather than a one-time merging after training.

**Key Insight**: Inspired by the Complementary Learning Systems (CLS) theory, an iterative framework with an inner loop (fast learning) and an outer loop (global consolidation) is designed.

**Core Idea**: To iteratively update the parameter importance distribution using inner and outer loops, achieving smoother continual learning optimization through multi-round knowledge fusion.

## Method

### Overall Architecture
Recurrent-KIF restructures the training process into multiple iterative learning cycles, each containing: (1) Inner learner + Knowledge identification: rapidly adapting to new tasks and estimating parameter importance; (2) Outer learner + Knowledge fusion: utilizing a memory buffer to retrieve historical task information and executing knowledge fusion.

### Key Designs
1. **Inner Learner**: Performs $Q$-step gradient updates on the new task data to generate the inner task vector $\tau_b^{\text{in}} = \theta_{b(Q)} - \theta_{b(0)}$. Meanwhile, the absolute value of the gradient-weight product $I(w_{ij}) = |w_{ij} \cdot \nabla L|$ is used as the importance metric, which is smoothed via Exponential Moving Average (EMA).

2. **Outer Learner**: Samples data from the memory buffer to dynamically update the importance distribution of historical tasks $I_b^{\text{out}}$ based on the latest model state. It also uses EMA smoothing (with coefficient $\alpha_2$) to alleviate the variance issue of small samples. The key innovation is updating importance conditioned on the current model state.

3. **Importance-Based Binary Mask Knowledge Fusion**: Obtains the top-20% quantile threshold on both inner and outer importance distributions to generate binary masks $m^{\text{in}}$ and $m^{\text{out}}$, and then performs: $\theta_{b+1} = \theta_b + (m^{\text{in}} \odot \tau^{\text{in}} + m^{\text{out}} \odot \tau^{\text{out}}).$ This filters redundant information, retains task-specific knowledge to prevent forgetting, and merges task-shared knowledge to facilitate transfer.

### Loss & Training
The base optimization objective is the standard cross-entropy loss: $L = \mathbb{E}[-\log p_{\Theta}(y|x)]$. LoRA is used for PEFT, and a memory buffer stores 2% of the training samples for each historical task for replay. The inner loop has $Q=8$ steps, and the outer loop has 4 steps. The smoothing coefficients are $\alpha_1 = \alpha_2 = 0.55$. The binary mask threshold $\delta$ is selected as the top-20% quantile. The total number of training iterations is fixed to $N'$, and the number of fusion steps is $N'/Q$.

## Key Experimental Results

### Main Results

Results of the T5-large model on two CL Benchmarks (average of 3 task sequences):

| Method | Standard OP↑ | Standard BWT↑ | Long Seq OP↑ | Long Seq BWT↑ |
|------|-------------|---------------|-------------|---------------|
| SeqLoRA | 43.7 | -50.4 | 11.6 | -73.4 |
| O-LoRA | 75.8 | -3.8 | 69.6 | -4.1 |
| TaSL | 76.3 | -4.0 | 74.4 | -5.3 |
| VR-MCL | 76.0 | -3.7 | 74.8 | -4.9 |
| MIGU | 76.6 | - | 76.5 | - |
| **Recurrent-KIF** | **78.4** | **-2.8** | **77.8** | **-3.6** |
| MTL (Upper Bound) | 80.3 | - | 81.8 | - |

### Ablation Study

Ablation study on the Long Sequence Benchmark (Task Sequence 1):

| Variant | OP | BWT |
|------|-----|------|
| Recurrent-KIF (Full) | 77.9 | -3.4 |
| - DIE (Static Importance) | 74.8 | -4.8 |
| - KI (w/o Knowledge Identification) | 52.3 | -21.5 |
| + GM (Global Merging) | 72.1 | -11.2 |
| + Adaptive (Adaptive Fusion) | 76.1 | -4.1 |
| - Share (No Shared Area Update) | 75.8 | -4.3 |

### Key Findings
- Dynamic Importance Estimation (DIE) improves OP by 3.1% and BWT by 1.4% compared to the static method, validating the necessity of dynamic updates.
- Removing Knowledge Identification (KI) causes a 25.6% drop in OP, indicating that direct merging of task vectors severely damages historical knowledge.
- Multi-round fusion outperforms single-step fusion (analogous to the effect of SGD over GD), but excessive fusion steps introduce noise leading to overfitting.
- Consistent advantages are observed across different model scales from 770M to 13B, with OP improving from 75.6% to 78.2% on LLaMA2-7B.
- Visualizations show that most parameters in the task vector are redundant, and only a small subset of parameters in the encoder is truly important.
- Performance close to the MTL (multi-task learning) upper bound is achieved on the IMDB and AG News tasks, indicating significant knowledge transfer effects.

## Highlights & Insights
- The engineering implementation of CLS theory (fast learning in hippocampus + slow consolidation in neocortex) in CL; the biologically inspired inner-outer loop design is compelling.
- The finding that multi-round fusion outperforms one-time merging, analogized to SGD vs. GD, is highly ingenious.
- Smoothing importance scores with Exponential Moving Average (EMA) is a practical and elegant technique.
- Updating task-shared regions is crucial for knowledge transfer (substantiated by the "- Share" ablation).
- Visualization in Figure 5(a) reveals that parameters with large task vector magnitudes are not necessarily important, validating the necessity of knowledge identification.
- Visualization in Figure 5(b) shows that the importance distribution indeed changes after training on new tasks, confirming the necessity of dynamic updates.
- Performance on IMDB and AG News is close to the MTL upper bound, demonstrating that the method approaches theoretical optimality on certain tasks.
- Consistent advantages from 770M to 13B showcase the excellent scalability of the proposed method.

## Limitations & Future Work
- Relies on a memory buffer for replay, which is restricted in privacy-sensitive scenarios; generative replay could be considered as an alternative.
- Element-wise operations and multi-round fusion increase time complexity; layer-wise or module-wise fusion might be required for larger models.
- Only evaluated on text classification tasks, lacking validation on generative tasks (e.g., summarization, dialogue, code generation).
- The quantization threshold (top-20%) is fixed, which may benefit from adaptive adjustment strategies.
- The choice of inner and outer loop steps ($Q=8, M=4$) lacks theoretical guidance and relies on empirical tuning.
- Cross-modal (vision-language, audio, etc.) continual learning scenarios have not been explored.

## Related Work & Insights
- Forms a spectrum of parameter importance estimation methods along with TaSL and VR-MCL: static $\to$ partially dynamic $\to$ fully dynamic (ours).
- The application of model merging techniques in CL is an active research direction, related to methods like Task Arithmetic and TIES-Merging.
- The inner-outer loop paradigm can be transferred to other scenarios requiring a balance between exploration and consolidation, similar to structures found in meta-learning.
- The design philosophy of the binary mask shares commonalities with model merging methods such as DARE (Drop and Rescale).
- Compared to orthogonalization methods (O-LoRA), importance-guided fusion is more flexible and achieves better performance.

## Rating
- Novelty: ⭐⭐⭐⭐ The CLS-inspired inner-outer loop design is novel, and dynamic importance estimation is a key contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple backbones (770M to 13B), two benchmarks, and thorough ablation.
- Writing Quality: ⭐⭐⭐⭐ Clear framework diagrams and visualizations, with detailed description of the methodology.
- Value: ⭐⭐⭐⭐ Possesses practical value in the field of CL for LLMs, with open-sourced code and strong generalizability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] GORP: Continual Gradient Low-Rank Projection Fine-Tuning for LLMs](gorp_continual_gradient_projection.md)
- [\[ACL 2025\] On Entity Identification in Language Models](on_entity_identification_in_language_models.md)
- [\[NeurIPS 2025\] C²Prompt: Class-aware Client Knowledge Interaction for Federated Continual Learning](../../NeurIPS2025/llm_nlp/c2prompt_class-aware_client_knowledge_interaction_for_federated_continual_learni.md)
- [\[ICCV 2025\] Any-SSR: How Recursive Least Squares Works in Continual Learning of Large Language Models](../../ICCV2025/llm_nlp/any_ssr_how_recursive_least_squares_works_in_continual_learning_of_large_language_models.md)
- [\[ACL 2025\] SudoLM: Learning Access Control of Parametric Knowledge with Authorization Alignment](sudolm_authorization_alignment.md)

</div>

<!-- RELATED:END -->
