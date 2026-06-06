---
title: >-
  [Paper Note] Unveiling Chain of Step Reasoning for Vision-Language Models with Fine-grained Rewards
description: >-
  [NeurIPS 2025][Multimodal VLM][chain-of-step] This paper proposes the Chain-of-Step (CoS) reasoning framework, which decomposes VLM reasoning chains into structured steps consisting of Name, Thought…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "chain-of-step"
  - "process reward model"
  - "step-level reasoning"
  - "iterative DPO"
  - "inference-time scaling"
date: 2026-05-08
content_hash: c3de1035cd23e821
---

# Unveiling Chain of Step Reasoning for Vision-Language Models with Fine-grained Rewards

**Conference**: NeurIPS 2025
**arXiv**: [2509.19003](https://arxiv.org/abs/2509.19003)  
**Code**: [https://github.com/baaivision/CoS](https://github.com/baaivision/CoS)  
**Area**: Multimodal VLM / Visual Reasoning / Process Reward Model
**Keywords**: chain-of-step, process reward model, step-level reasoning, iterative DPO, inference-time scaling

## TL;DR

This paper proposes the Chain-of-Step (CoS) reasoning framework, which decomposes VLM reasoning chains into structured steps consisting of Name, Thought, and Reflection components. A step-level Process Reward Model (PRM) is trained to provide fine-grained reward signals. Combined with iterative DPO and step-level beam search, the framework systematically improves VLM reasoning—achieving an average of 73.4% (+4.0%) across 6 benchmarks on InternVL-2.5-MPO-8B and 64.2% (+12.1%) on LLaVA-NeXT-8B—while revealing the counterintuitive finding that quality matters far more than length in VLM reasoning, contrary to trends observed in LLM research.

## Background & Motivation

**Background**: Chain-of-Thought reasoning has achieved remarkable success in the LLM domain, with OpenAI-o1 and DeepSeek-R1 demonstrating substantial reasoning gains through large-scale RL combined with CoT. The VLM community is actively exploring CoT reasoning (e.g., LLaVA-CoT, Insight-V, URSA), but the field remains largely at a coarse-grained stage.

**Limitations of Prior Work**: Current VLM CoT reasoning outputs consist of unstructured monolithic "thought" blocks—lacking uniform format and clear step boundaries—resulting in two core problems: (1) reasoning processes tend to become verbose and disorganized, hampering systematic structured reasoning; and (2) the quality of intermediate reasoning steps cannot be evaluated, leaving both RL training and inference-time scaling without effective reward signals.

**Key Challenge**: PRMs in the LLM domain (e.g., Math-Shepherd, Let's Verify Step by Step) have demonstrated the value of step-level rewards, but applying them to VLMs presents two non-trivial challenges: how to define a "step" (decomposing the reasoning chain into logically coherent, progressive units) and how to evaluate a "step" (providing fine-grained step-level reward signals).

**Goal**: To establish a complete step-level reasoning framework for VLMs, encompassing the definition of structured reasoning formats, SFT data construction, process reward model training, and RL training and inference-time scaling based on fine-grained rewards.

**Key Insight**: The framework begins with structured design of the reasoning chain, using special tokens to delineate step boundaries. A Reflection component is introduced at each step to anchor reasoning to visual content and mitigate hallucinations, ensuring stable and parseable step segmentation that provides a solid foundation for PRM training and RL.

**Core Idea**: By structuring VLM reasoning chains into evaluable discrete steps and training a PRM to provide step-level fine-grained rewards, both RL training and inference-time scaling can benefit from intermediate step quality.

## Method

### Overall Architecture

A three-stage pipeline:
1. **SFT on ShareGPT-Step-300K**: SFT on 300K structured step-reasoning data to teach the model to produce step-wise reasoning chains.
2. **PRM Training**: Step-level data is annotated via Monte Carlo estimation and GPT-4o-as-Judge (100K each), and InternVL-2.5-MPO-38B is trained as the process reward model.
3. **Iterative DPO with PRM**: The PRM scores sampled reasoning paths; positive and negative sample pairs are selected for 3 rounds of iterative DPO to progressively enhance reasoning ability.

### Key Designs

1. **Structured Reasoning Template**

    - **Function**: Decomposes the VLM's free-form reasoning chain into discrete steps that are format-stable, parseable, and evaluable.
    - **Mechanism**: Each reasoning step contains three components—**Name** (a summary of the step, e.g., "identify geometric shapes"), **Thought** (detailed reasoning content), and **Reflection** (establishing connections to visual content and prior steps to mitigate hallucinations). Special tokens (`<|reasoning_start|>`, `<|reasoning_proceed|>`, `<|reasoning_end|>`, etc.) mark step boundaries; the number and length of steps are determined autonomously during autoregressive generation.
    - **Design Motivation**: Prompt-based format control is unstable and requires additional data cleaning; embedding format via special tokens ensures output stability. The Reflection component is specifically designed for VLMs—while LLMs do not need to revisit visual content, VLMs are prone to generating content inconsistent with the image, and an explicit reflection step mitigates this issue.

2. **ShareGPT-Step-300K Dataset Construction**

    - **Function**: Provides high-quality structured step-reasoning training data for the SFT stage.
    - **Mechanism**: A "reasoning from outcome" strategy is employed—both the question and the ground-truth answer are provided to GPT-4o, which is prompted to reverse-engineer a step-wise reasoning process. The dataset covers 17 datasets across 4 task categories (mathematical reasoning, scientific reasoning, chart/document analysis, and world knowledge), yielding 300K high-quality samples after rigorous format filtering.
    - **Design Motivation**: Prompting LLMs to generate reasoning from scratch is error-prone; providing reference answers substantially reduces generation difficulty and improves quality. Diversity across 17 datasets ensures that the learned reasoning capabilities generalize broadly.

3. **Process Reward Model (PRM)**

    - **Function**: Assigns a quality score to each step of the reasoning chain, providing fine-grained reward signals for RL training and inference-time scaling.
    - **Mechanism**: Two complementary annotation strategies are employed—the Math-Shepherd method (MC estimation, sampling 16 continuation paths per step to estimate correctness probability) and GPT-4o-as-Judge (three-level scoring: Good/Neutral/Bad), each annotating 100K step-level samples. InternVL-2.5-MPO-38B serves as the backbone, trained with BCE loss for 2 epochs. At evaluation time, the final score is computed as a weighted sum: step score × 20% + answer score × 80%.
    - **Design Motivation**: MC estimation objectively assesses the probability of a step leading to a correct answer from a statistical perspective, while LLM-as-Judge evaluates the logical correctness of a step from a semantic perspective—the two methods are complementary, yielding a more robust PRM. The 38B model is chosen as the PRM backbone (over 8B) because larger models produce more accurate step evaluations (step accuracy on unseen data: 87.3% vs. 83.7%).

### Loss & Training

- **SFT Stage**: Standard next-token prediction loss; trained for 1 epoch on ShareGPT-Step-300K.
- **PRM Training**: Binary Cross Entropy loss; predicts correctness probability for each step.
- **Iterative DPO**: Standard DPO loss, $\mathcal{L}_{\text{DPO}}(\pi_\theta;\pi_{\text{ref}}) = -\mathbb{E}[\log\sigma(\beta\log\frac{\pi_\theta(y_+|x)}{\pi_{\text{ref}}(y_+|x)} - \beta\log\frac{\pi_\theta(y_-|x)}{\pi_{\text{ref}}(y_-|x)})]$, with $\beta=0.1$. Each round initializes policy and reference from the SFT model; 16 reasoning paths are sampled per instance and scored by the PRM to select positive/negative pairs (score gap must exceed threshold $t$); approximately 20K pairs per round over 3 rounds.
- **Training Cost**: SFT requires approximately 9 hours on a single A800 node; 3 rounds of DPO require approximately 6 hours total on a single A800 node.
- **Step-level Beam Search (Inference-time)**: At each step, $N$ candidate continuations are sampled → scored by the PRM → the best step is selected → the next step is sampled from this basis → repeated until the answer is produced. Total compute is identical to Best-of-N.

## Key Experimental Results

### Main Results

| Method | MathVista | MMStar | MMMU | M3CoT | AI2D | ChartQA | Avg |
|--------|-----------|--------|------|-------|------|---------|-----|
| InternVL2.5-MPO-8B (baseline) | 65.0 | 60.7 | 53.8 | 67.5 | 84.2 | 85.0 | 69.4 |
| + CoS SFT | 65.9 | 61.0 | 53.7 | 75.7 | 81.6 | 88.3 | 71.0 |
| + CoS Iterative DPO | **67.8** | **63.5** | **55.5** | **81.0** | **84.9** | **87.4** | **73.4** |
| LLaVA-NeXT-8B (baseline) | 45.9 | 43.1 | 36.9 | 45.6 | 71.5 | 69.4 | 52.1 |
| + CoS SFT | 51.4 | 54.7 | 39.6 | 67.4 | 76.1 | 75.7 | 60.8 |
| + CoS Iterative DPO | **54.7** | **58.9** | **41.8** | **71.7** | **79.2** | **79.1** | **64.2** |

### Ablation Study

**RL Reward Strategy Ablation** (on LLaVA-NeXT-SFT):

| Method | MathVista | MMStar | M3CoT |
|--------|-----------|--------|-------|
| LLaVA-NeXT-SFT | 51.4 | 54.7 | 67.4 |
| Answer Only (PRM) | 53.1 | 57.3 | 69.7 |
| Outcome (GT labels) | 53.5 | 58.1 | 70.0 |
| Step&Answer (PRM) | **54.7** | **58.9** | **71.7** |

**Reasoning Mode Ablation** (on LLaVA-NeXT):

| Method | Reward | MathVista | MMStar | M3CoT | Avg |
|--------|--------|-----------|--------|-------|-----|
| No Reason SFT→RL | outcome | 51.5 | 56.4 | 63.4 | 57.1 (+2.1) |
| Direct Prompt SFT→RL | outcome | 53.1 | 58.2 | 69.3 | 60.2 (+2.7) |
| CoS SFT→RL | PRM | **54.7** | **58.9** | **71.7** | **61.8 (+4.0)** |

**GRPO Validation**:

| Method | MathVista | MMStar | M3CoT | Avg |
|--------|-----------|--------|-------|-----|
| Outcome GRPO | 54.3 | 57.9 | 71.4 | 61.2 |
| CoS GRPO (PRM) | **56.3** | **59.1** | **73.7** | **63.0** |

### Key Findings

- **Optimal Step Weight of 20%**: Neither pure step scores nor pure answer scores are optimal; Best-of-16 accuracy peaks at a step weight of 20%, indicating that jointly considering step and answer quality is most effective.
- **Step-level Beam Search > Best-of-N**: At $N=64$, PRM-BS outperforms Self-Consistency by more than 5%, while maintaining identical compute cost to Best-of-N PRM yet achieving superior performance.
- **Counterintuitive Reasoning Length Dynamics**: In the early stages of PRM DPO training, the model actively shortens reasoning chains to improve quality, with length only gradually increasing after stabilization. In contrast, Outcome DPO causes length to increase monotonically. This indicates that quality matters far more than length in VLM reasoning—contrary to the "longer = stronger" pattern observed in LLM research.
- **Strong vs. Weak Model Differences**: The weaker model (LLaVA-NeXT) benefits substantially from both SFT and DPO (+12.1%), while the stronger model (InternVL2.5-MPO) shows limited gains from SFT but still improves significantly with DPO, demonstrating that RL is more critical for stronger models.
- **Step-wise DPO Failure**: Attempts to construct preference pairs at each step for step-level DPO failed because chosen and rejected responses were too similar, causing the model to collapse—sufficient divergence between positive and negative samples is necessary to form an effective learning signal.

## Highlights & Insights

- **Practicality of Structured Design**: The three-component design of Name, Thought, and Reflection has a clear division of labor—Name provides navigation, Thought carries the reasoning, and Reflection is dedicated to grounding visual information and prior reasoning. The decision to use special tokens rather than prompt-based format control is pragmatically sound.
- **"Quality > Length" Insight**: This finding carries important implications—visual reasoning relies more on effectively leveraging visual information and triggering knowledge connections than on lengthy intermediate derivations, as in pure mathematical reasoning.
- **Scale Efficiency of the PRM**: The 38B PRM needs to be trained only once and can serve the RL training and inference scaling of multiple 8B models, representing an efficient allocation of resources.
- **Comprehensive Failure Analysis**: The paper honestly reports the failure of step-wise DPO, illuminating the importance of sufficient divergence between positive and negative samples in preference learning.
- **Transparency and Reproducibility**: The dataset, PRM, and code are all open-sourced, establishing a solid baseline for fine-grained VLM reasoning research.

## Limitations & Future Work

- **Dependence on Closed-Source Models for Data Construction**: Both ShareGPT-Step-300K and GPT-4o-as-Judge annotation rely on GPT-4o, introducing cost and uncertainty.
- **Limited Validation at Scale**: Comprehensive validation is performed only on 8B models; performance and reasoning length dynamics on larger models (e.g., 72B) remain unexplored.
- **Uncertain Contribution of the Reflection Component**: No ablation removing the Reflection component is provided, making it impossible to quantitatively verify its contribution to hallucination mitigation.
- **PRM Inference Cost**: The 38B PRM incurs high inference costs in production deployment; knowledge distillation into a smaller PRM is a worthwhile direction.
- **Generality of Step Definition**: The current step segmentation approach is primarily designed for QA and mathematical reasoning; its applicability to more open-ended visual tasks (e.g., creative generation, long-document understanding) has not been validated.

## Related Work & Insights

- **vs. LLaVA-CoT**: LLaVA-CoT uses coarse-grained four-segment reasoning (SUMMARY/CAPTION/REASONING/CONCLUSION), whereas CoS employs fine-grained Name/Thought/Reflection steps with PRM. This granularity difference directly impacts the precision of reward signals and the effectiveness of inference-time scaling.
- **vs. URSA**: URSA also employs a PRM but operates on coarse-grained reasoning chains; CoS's structured steps enable more accurate per-step PRM evaluation, making step-level beam search feasible.
- **vs. Insight-V**: Insight-V uses a multi-agent system (reasoning agent + summary agent), while CoS adopts a simpler single-model + PRM architecture—suggesting that complex problems do not necessarily require complex systems.
- **vs. NoisyRollout/Sherlock**: NoisyRollout enhances exploration diversity and Sherlock performs response-level self-correction; both are complementary to CoS's step-level fine-grained rewards and could be combined.
- **Relationship to Inference-Time Scaling**: PRM + step-level beam search provides a new paradigm for utilizing inference-time compute in VLMs, offering greater efficiency compared to Self-Consistency and Best-of-N.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of structured steps and PRM is relatively novel for VLMs; the three-component Name+Thought+Reflection design and step-level beam search are innovative contributions, though each individual component (PRM, iterative DPO, structured reasoning) is not entirely new.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Ablations are exceptionally comprehensive, covering step weight, PRM backbone selection, reasoning length dynamics, reasoning pattern comparisons, GRPO validation, and step-wise DPO failure analysis; every conclusion is empirically supported.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The narrative progression is clear (define steps → evaluate steps → leverage steps); Figure 1 provides an accessible overview of the complex design; failed experiments are honestly reported; the paper is compact with high information density.
- **Value**: ⭐⭐⭐⭐⭐ — The paper provides a complete, fully open-sourced framework for VLM reasoning post-training (SFT + PRM + DPO + beam search); the "quality > length" insight directly informs adaptive inference research; PRM + step-level beam search can serve as a standard approach for VLM inference-time scaling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] AffordBot: 3D Fine-grained Embodied Reasoning via Multimodal Large Language Models](affordbot_3d_fine-grained_embodied_reasoning_via_multimodal_large_language_model.md)
- [\[NeurIPS 2025\] SpatialThinker: Reinforcing 3D Reasoning in Multimodal LLMs via Spatial Rewards](spatialthinker_reinforcing_3d_reasoning_in_multimodal_llms_via_spatial_rewards.md)
- [\[ICCV 2025\] Fine-Grained Evaluation of Large Vision-Language Models in Autonomous Driving](../../ICCV2025/multimodal_vlm/fine-grained_evaluation_of_large_vision-language_models_in_autonomous_driving.md)
- [\[ICCV 2025\] LLaVA-CoT: Let Vision Language Models Reason Step-by-Step](../../ICCV2025/multimodal_vlm/llava-cot_let_vision_language_models_reason_step-by-step.md)
- [\[NeurIPS 2025\] FOCUS: Internal MLLM Representations for Efficient Fine-Grained Visual Question Answering](focus_internal_mllm_representations_for_efficient_fine-grained_visual_question_a.md)

</div>

<!-- RELATED:END -->
