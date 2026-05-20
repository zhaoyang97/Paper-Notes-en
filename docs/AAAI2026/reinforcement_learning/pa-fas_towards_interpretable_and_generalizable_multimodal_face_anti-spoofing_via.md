---
title: >-
  [Paper Note] PA-FAS: Towards Interpretable and Generalizable Multimodal Face Anti-Spoofing via Path-Augmented Reinforcement Learning
description: >-
  [AAAI 2026][Reinforcement Learning][Face Anti-Spoofing] This paper proposes PA-FAS, a framework that addresses two critical bottlenecks of the SFT+RL paradigm in multimodal FAS — insufficient reasoning path diversity and…
tags:
  - "AAAI 2026"
  - "Reinforcement Learning"
  - "Face Anti-Spoofing"
  - "Multimodal Fusion"
  - "Domain Generalization"
  - "Interpretability"
date: 2026-05-08
content_hash: 102c04fedf2ab1c3
---

# PA-FAS: Towards Interpretable and Generalizable Multimodal Face Anti-Spoofing via Path-Augmented Reinforcement Learning

**Conference**: AAAI 2026
**arXiv**: [2511.17927](https://arxiv.org/abs/2511.17927)  
**Code**: None  
**Area**: Reinforcement Learning
**Keywords**: Face Anti-Spoofing, Multimodal Fusion, Domain Generalization, Reinforcement Learning, Interpretability

## TL;DR

This paper proposes PA-FAS, a framework that addresses two critical bottlenecks of the SFT+RL paradigm in multimodal FAS — insufficient reasoning path diversity and reasoning shortcut — via a Reasoning Path Augmentation strategy and an answer shuffling mechanism, achieving the first unified solution for multimodal fusion, domain generalization, and interpretability simultaneously.

## Background & Motivation

### Problem Definition

Face Anti-Spoofing (FAS) aims to distinguish genuine faces from spoofing presentations (printed photos, replay videos, 3D masks, etc.), serving as a critical safeguard for the security of face recognition systems. Multimodal FAS leverages RGB, depth, and infrared modalities to improve detection accuracy and robustness.

### Three Research Gaps in Existing Methods

**Insufficient domain generalization**: Most existing methods are designed for single-modality settings with limited cross-domain generalization capability.

**Lack of interpretability in multimodal methods**: Despite superior performance, multimodal methods lack explicit interpretability mechanisms to identify spoofing cues in depth and infrared modalities.

**Limitations of MLLMs in domain generalization scenarios**: Existing MLLM-based FAS methods (e.g., FaceCoT, SHIELD) demonstrate strong reasoning ability but overlook generalization and cannot handle cross-modal cue integration.

### Failure Analysis of the SFT+RL Paradigm (Core Motivation)

The paper conducts an in-depth analysis of the failure mechanism when applying the SFT+RL paradigm to multimodal FAS:

**Problem 1: Insufficient reasoning path diversity**
- Multimodal FAS datasets typically provide only simple binary labels, lacking language-level annotations of key visual cues.
- The SFT stage suffers from task homogeneity and limited data, causing the model to overfit rigid patterns.
- The RL stage lacks effective feedback and exploration space — most samples receive extreme rewards (all 1s or all 0s), providing no informative intermediate signals.
- Empirical evidence: Models fine-tuned on the original data exhibit a nearly linear growth in accumulated invalid samples.

**Problem 2: Reasoning shortcut**
- Even when reasoning chains are augmented via conventional data augmentation, the model bypasses the chain-of-thought (CoT) by directly predicting answers from images.
- Evidence: Randomly replacing the reasoning text in the SFT stage with CoT from other samples yields nearly identical performance, demonstrating that the model relies on visual prediction while ignoring the CoT.
- This leads to overconfidence and significantly narrows the exploration space in the RL stage.

## Method

### Overall Architecture

PA-FAS follows a four-step pipeline: (a) low-level data annotation → (b) high-level data annotation to generate CoT → (c) positive-negative random path sampling to expand reasoning paths → (d) two-stage training with SFT (with answer shuffling) + RL (GRPO).

### Key Designs

#### 1. **Reasoning Path Augmentation**

The core innovation, which systematically expands reasoning paths by constructing a structured reasoning tree:

- **Reasoning tree construction**: Based on a fine-grained hierarchical taxonomy of FAS tasks (sunburst diagram in Figure 4), a formal reasoning tree $\mathcal{T} = (\mathcal{V}, \mathcal{E})$ is established:
    - $\mathcal{V}$: set of reasoning nodes (each node represents a semantic category or logical decision unit)
    - $\mathcal{E}$: set of directed edges
    - Path $\mathcal{P} = (v_1, v_2, \ldots, v_n)$: a complete reasoning chain from the root to the target leaf node

- **Positive-Negative Random Path Sampling (PNRPS)**: The core mechanism of Algorithm 1:
    - **Single-node operation constraint**: Each node permits at most one forward exploration step $(+, v)$ and one backward reflection step $(-, v)$ to avoid redundant traversal.
    - **Path length constraint**: $L_{\max} = \alpha(D-1)$, where $D$ is the maximum classification depth and $\alpha > 1$ is a tunable scaling factor.
    - **Semantic consistency**: Each node is associated with predefined CoT clause templates, which are concatenated along the path to generate the final reasoning text.
    - **Structured sampling**: Rule-based depth-first traversal randomly samples $N$ valid paths utilizing arbitrary combinations of RGB, IR, and DEPTH modalities.

- **Data augmentation effect**:
  $$\{(x_i, \ell_i)\}_{i=1}^{M} \rightarrow \bigcup_{i=1}^{M}\{(x_i, \text{CoT}(\mathcal{P}_i^{(j)})) \mid j=1,\ldots,N\}$$
  Starting from 800 annotated samples, $N=50$ reasoning paths are generated per sample, yielding approximately 40,000 structurally diverse and semantically consistent augmented samples.

#### 2. **Answer Shuffling**

A key design to address the reasoning shortcut problem:

- **Motivation**: The coupling of a single-task objective with rich reasoning paths causes the model to directly predict answers from visual input, bypassing the reasoning chain.
- **Method**: During the SFT stage, the final answer in each CoT is randomly replaced with the answer from another sample.
- **Effect**: This forces the model to focus on learning diverse reasoning paths rather than memorizing answers, thereby preserving sufficient exploration space for the RL stage.

#### 3. **RL Stage (GRPO)**

Group Relative Policy Optimization is adopted for policy optimization:

- For each question-answer pair $(q, a)$, $G$ responses are sampled from the old policy.
- Reward definition: $\mathcal{R} = \mathcal{R}_{\text{format}} + \mathcal{R}_{\text{classification}}$
- Relative advantage estimation: $\hat{A}_{i,t} = \frac{\mathcal{R}_i - \text{mean}(\{\mathcal{R}_i\})}{\text{std}(\{\mathcal{R}_i\})}$
- The KL divergence term $D_{KL}(\pi_\theta \| \pi_{\text{ref}})$ is removed under data-scarce conditions to avoid suppressing exploration.

### Loss & Training

- **Backbone model**: Qwen2.5VL-3B
- **SFT and RL stages**: Both trained for 500 steps with a constant learning rate of 1e-6.
- **Data**: Only approximately 800 high-quality structured reasoning path samples are required.
- **Evaluation datasets**: WMCA, CASIA-SURF, CASIA-CeFA, PADISI.

## Key Experimental Results

### Main Results (Protocol 1: Full-modality Cross-dataset Testing)

| Method | Category | Avg. HTER (%)↓ | Avg. AUC (%)↑ |
|--------|----------|----------------|---------------|
| FLIP | Unimodal DG | 16.11 | 90.83 |
| MMDG | Multimodal DG | 22.93 | 84.19 |
| DADM | Multimodal DG | 13.63 | 92.96 |
| Qwen2.5-VL-3B (zero-shot) | Interpretable Multimodal DG | 33.46 | 69.36 |
| Qwen2.5-VL-3B-SFT | Interpretable Multimodal DG | 23.25 | 79.32 |
| Qwen2.5-VL-3B-SFT+GRPO | Interpretable Multimodal DG | 34.37 | 68.68 |
| **PA-FAS (Ours)** | **Interpretable Multimodal DG** | **15.21** | **89.13** |

Key observation: Naïve SFT+GRPO performs even worse than SFT alone (HTER increases from 23.25% to 34.37%), validating the authors' failure analysis. PA-FAS reduces HTER to 15.21%, significantly outperforming all interpretable models.

### Ablation Study (Missing Modality Scenario, Protocol 2)

| Method | Missing D HTER↓ | Missing I HTER↓ | Missing D&I HTER↓ | Avg. HTER↓ |
|--------|-----------------|-----------------|-------------------|------------|
| DADM | 21.56 | 20.82 | 22.61 | 21.66 |
| Qwen2.5-VL-3B-SFT | 23.25 | 23.25 | 23.25 | 23.25 |
| **PA-FAS** | **15.68** | **17.32** | **14.67** | **15.85** |

PA-FAS maintains superior performance under missing modality conditions, demonstrating that the reasoning path augmentation strategy effectively exploits complementary multimodal information.

### Key Findings

1. **SFT+RL collapse validated**: On datasets with only binary labels, SFT+GRPO degrades performance (HTER: 23.25% → 34.37%).
2. **Reasoning shortcut confirmed**: Replacing reasoning text leaves model performance unchanged, proving the model completely ignores the CoT reasoning chain.
3. **Importance of path diversity**: Datasets with diverse reasoning paths significantly outperform those with single-path data under both SFT and SFT+RL settings.
4. **Data efficiency**: Only 800 samples with path augmentation suffice to surpass methods trained on 35,000 raw data samples.
5. **Limited source domain scenario (Protocol 3)**: PA-FAS achieves an average HTER of 9.22% (as low as 0.15% on CW→PS), demonstrating strong domain generalization capability.

## Highlights & Insights

1. **In-depth diagnosis of SFT+RL failure**: Rather than simply proposing a method, the paper first rigorously diagnoses the failure causes through experiments (cumulative invalid sample analysis and reasoning text replacement experiments).
2. **Elegant design of reasoning path augmentation**: A formal reasoning tree is constructed based on a hierarchical taxonomy, and positive-negative random path sampling expands the reasoning space at minimal cost.
3. **Counter-intuitive answer shuffling design**: Deliberately introducing incorrect answers during the SFT stage forces the model to learn the reasoning process — a counter-intuitive yet highly effective design.
4. **First unification of three objectives**: An integrated solution for multimodal fusion, domain generalization, and interpretability.
5. **Exceptional data efficiency**: SOTA-level performance is achieved with only 800 samples.

## Limitations & Future Work

1. **Backbone model constraint**: Only Qwen2.5VL-3B (3B parameters) is used; larger models may yield further improvements.
2. **Manual reasoning tree construction**: The hierarchical taxonomy requires domain expert knowledge; automated construction remains a future direction.
3. **Annotation cost**: Although only 800 samples are needed, their CoT annotation still requires non-trivial human effort.
4. **Evaluation scope**: Validation is conducted primarily on four datasets; more diverse attack types and scenarios remain to be explored.
5. **Inference latency**: MLLM-based methods may not meet the real-time deployment requirements.

## Related Work & Insights

- **"SFT Memorizes, RL Generalizes" theory**: The paper builds upon Chu et al.'s SFT-memorization + RL-generalization theory, while noting that this theory requires new adaptations for data-scarce multimodal scenarios.
- **GRPO applied to FAS**: This work represents the first application of GRPO to the FAS task, demonstrating the feasibility of rule-based group advantage strategies under low training cost conditions.
- **Deeper understanding of reasoning shortcuts**: The answer shuffling mechanism provides an important reference for other visual understanding tasks combining SFT and RL.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The combination of reasoning path augmentation and answer shuffling is highly original, with rigorous analysis of the failure mechanism.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three protocols, ablation studies, and diverse baselines constitute a rigorous experimental design.
- Writing Quality: ⭐⭐⭐⭐ — Logic is clear and motivation is well-justified, though some notation is slightly complex.
- Value: ⭐⭐⭐⭐⭐ — The failure analysis of the SFT+RL paradigm and the proposed solutions have broad reference value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Distilling Deep Reinforcement Learning into Interpretable Fuzzy Rules: An Explainable AI Framework](distilling_deep_reinforcement_learning_into_interpretable_fuzzy_rules_an_explain.md)
- [\[AAAI 2026\] Think, Speak, Decide: Language-Augmented Multi-Agent Reinforcement Learning for Economic Decision-Making](think_speak_decide_language-augmented_multi-agent_reinforcement_learning_for_eco.md)
- [\[ACL 2026\] Language-Coupled Reinforcement Learning for Multilingual Retrieval-Augmented Generation](../../ACL2026/reinforcement_learning/language-coupled_reinforcement_learning_for_multilingual_retrieval-augmented_gen.md)
- [\[ICLR 2026\] Spotlight on Token Perception for Multimodal Reinforcement Learning](../../ICLR2026/reinforcement_learning/spotlight_on_token_perception_for_multimodal_reinforcement_learning.md)
- [\[AAAI 2026\] MMhops-R1: Multimodal Multi-hop Reasoning](mmhops-r1_multimodal_multi-hop_reasoning.md)

</div>

<!-- RELATED:END -->
