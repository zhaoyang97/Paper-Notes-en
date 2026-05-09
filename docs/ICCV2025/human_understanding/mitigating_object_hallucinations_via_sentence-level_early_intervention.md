---
title: >-
  [Paper Note] Mitigating Object Hallucinations via Sentence-Level Early Intervention
description: >-
  [ICCV 2025][Human Understanding][Object Hallucination] This paper proposes SENTINEL, a framework that mitigates object hallucinations in MLLMs via sentence-level early intervention and in-domain preference learning. It reduces hallucination rates by over 90% on Object HalBench while maintaining or even improving general-purpose capabilities.
tags:
  - ICCV 2025
  - Human Understanding
  - Object Hallucination
  - Preference Learning
  - Early Intervention
  - In-domain Data
  - DPO
date: 2026-05-08
content_hash: 2c152b65d97e6989
---

# Mitigating Object Hallucinations via Sentence-Level Early Intervention

**Conference**: ICCV 2025  
**arXiv**: [2507.12455](https://arxiv.org/abs/2507.12455)  
**Code**: [https://github.com/pspdada/SENTINEL](https://github.com/pspdada/SENTINEL)  
**Area**: Human Understanding  
**Keywords**: Object Hallucination, Preference Learning, Early Intervention, In-domain Data, DPO

## TL;DR

This paper proposes SENTINEL, a framework that mitigates object hallucinations in MLLMs via sentence-level early intervention and in-domain preference learning. It reduces hallucination rates by over 90% on Object HalBench while maintaining or even improving general-purpose capabilities.

## Background & Motivation

Multimodal large language models (MLLMs) have achieved significant progress in cross-modal understanding, yet suffer from severe hallucination problems—generating fabricated content that contradicts visual inputs. Existing mitigation approaches face multiple challenges:

**Decoding-strategy methods (VCD, OPERA, DoLa)**: Introduce additional inference overhead, increase latency, and may rely on specific modules, limiting scalability.

**Preference alignment methods**:
- Rely on large closed-source models (GPT-4) or manual annotation, incurring high costs
- Rewriting outputs with external models causes distributional mismatch between training data and the model's original outputs
- Out-of-domain training data damages generalization ability

**Key observations in this paper**:
1. Hallucinations intensify as generated text grows longer—hallucinated objects become more frequent in later tokens
2. Intervening at the sentence where hallucinations first appear can significantly reduce hallucination propagation in subsequent outputs

These two observations motivate the core idea: **early intervention**—correcting hallucinations at the moment of first occurrence to prevent them from cascading forward.

## Method

### Overall Architecture

SENTINEL (Sentence-level Early iNtervention Through IN-domain prEference Learning) consists of three stages:
1. **In-domain Candidate Bootstrapping**: Sampling multiple candidates from the current model, extracting objects, and cross-validating them
2. **Context-aware Preference Data Generation**: Constructing positive–negative sample pairs with contextual information
3. **Context-aware Preference Learning**: Training with the C-DPO loss

### Key Designs

**1. In-domain Candidate Bootstrapping**

- Generates $n$ candidate sentences via sampling decoding from the current model (preserving in-domain distribution consistency)
- Extracts object entities from text using SceneGraphParser
- Performs cross-validation with two open-vocabulary detectors (GroundingDINO + YOLO World):
    - Both models confirm absence → labeled as "hallucination"
    - Both models confirm presence → labeled as "factual"
    - Conflicting results → labeled as "uncertain" (discarded)

**2. Context-aware Preference Data Generation**

Positive samples are further divided into:
- **Context-consistent positives ($y_{w+}$)**: Described objects are contextually associated with the preceding context
- **Context-irrelevant positives ($y_{w-}$)**: Described objects have no contextual association

Experiments confirm that $y_{w+}$ performs better, as richer contextual information enhances the model's ability to maintain contextual coherence.

**3. Iterative Context Bootstrapping (ICB)**

Preference data is generated iteratively sentence by sentence: in each round, hallucination-free sentences that pass verification are appended to the context before the next round of sampling and annotation. This ensures that preference data covers diverse contextual scenarios, improving generalization.

### Loss & Training

This paper proposes Context-aware DPO (C-DPO), which extends standard DPO by incorporating context $c$ as part of the input:

The input is $x' = [v, q, c]$, where context $c$ is excluded from gradient computation; gradients arise solely from the discrimination between $y_{w+}$ and $y_l$.

Training details:
- LLaVA-v1.5 serves as the reference model
- Optimized with LoRA + AdamW
- 7B model: 8.6K samples, learning rate 2e-7, trained for 1 epoch
- 13B model: 7.0K samples, learning rate 3e-7, trained for 1 epoch

## Key Experimental Results

### Main Results

Comparison with SOTA methods on LLaVA-v1.5-7B:

| Method | Resp. (↓) | Ment. (↓) | CHAIR (↓) | Hal. (↓) | VQAv2 (↑) | MM-Vet (↑) |
|--------|-----------|-----------|-----------|----------|-----------|------------|
| Baseline | 52.7 | 28.0 | 8.4 | 35.5 | 78.5 | 31.0 |
| RLAIF-V | 7.8 | 4.2 | 2.8 | 15.7 | 75.2 | 29.9 |
| TPO | 5.6 | 3.2 | 3.6 | 20.5 | 75.9 | 25.7 |
| **SENTINEL** | **4.3** | **2.6** | **2.9** | **14.6** | **78.4** | **32.6** |

Effectiveness is also demonstrated on the 13B model: CHAIR decreases from 6.9 to 2.7, and Hal. from 31.9 to 11.7.

### Ablation Study

In-domain data vs. rewritten data:

| Method | Resp. (↓) | Ment. (↓) | AMBER Acc (↑) | MM-Vet (↑) |
|--------|-----------|-----------|---------------|------------|
| In-domain data (8.6K) | 4.3 | 2.6 | 76.1 | 32.6 |
| Rewritten data (8.6K) | 4.8 | 2.9 | 75.0 | 31.3 |

Comparison of positive sample types:

| Positive Sample | Data Size | Resp. (↓) | TextVQA | MM-Vet |
|----------------|-----------|-----------|---------|--------|
| $y_{w+}$ 100% | 8.6K | 4.3 | 58.2 | 32.6 |
| $y_{w+}$ 50% + $y_{w-}$ 50% | 10.0K | 4.8 | 58.1 | — |

### Key Findings

- Compared to the previous SOTA (TPO), SENTINEL further reduces hallucinations by 24% on Object HalBench
- Outperforms the baseline across all 6 hallucination types; existence hallucination shows the most significant improvement (7B: +6.3)
- Unlike most hallucination mitigation methods that degrade general capabilities, SENTINEL maintains or improves performance on VQAv2, ScienceQA, and MM-Vet
- In-domain data outperforms GPT-4-rewritten data, validating the importance of distributional consistency
- Dual-detector cross-validation outperforms single-detector approaches

## Highlights & Insights

1. **Concise and compelling motivation for early intervention**: Hallucinations snowball during generation; cutting them off at the source is most effective
2. **No external large models required**: Preference data is constructed entirely from the model's own samples, preserving distributional consistency
3. **Context-aware preference learning**: Iterative bootstrapping generates preference pairs across diverse contexts, enhancing robustness
4. **General capabilities improve rather than degrade**: This is a notably rare property among hallucination mitigation methods

## Limitations & Future Work

- Relies on two object detectors for validation; errors from the detectors themselves may propagate
- The bootstrapping process requires multiple rounds of sampling, making data construction non-trivial in cost
- Validated only on LLaVA-v1.5; generalization to other architectures (e.g., InternVL, Qwen-VL) remains untested
- Primarily targets object hallucinations; effectiveness on other hallucination types such as attributes and relations remains to be verified

## Related Work & Insights

- Compared to preference learning methods such as RLAIF-V and TPO, SENTINEL's in-domain data strategy avoids distributional drift
- The early intervention paradigm can be transferred to other sequential generation tasks (e.g., error propagation in code generation)
- The context-aware design of C-DPO is generalizable to broader scenarios that require maintaining contextual coherence

## Rating

- Novelty: ⭐⭐⭐⭐ (The motivation analysis for early intervention is insightful; in-domain preference construction is creative)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Multiple benchmarks, multiple model scales, comprehensive ablations)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, well-articulated motivation)
- Value: ⭐⭐⭐⭐⭐ (Significant effectiveness without sacrificing general capabilities; highly practical)

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] CleanPose: Category-Level Object Pose Estimation via Causal Learning and Knowledge Distillation](cleanpose_category-level_object_pose_estimation_via_causal_learning_and_knowledg.md)
- [\[ICCV 2025\] Bi-Level Optimization for Self-Supervised AI-Generated Face Detection](bi-level_optimization_for_self-supervised_ai-generated_face_detection.md)
- [\[ICCV 2025\] HUMOTO: A 4D Dataset of Mocap Human Object Interactions](humoto_a_4d_dataset_of_mocap_human_object_interactions.md)
- [\[ICCV 2025\] MixRI: Mixing Features of Reference Images for Novel Object Pose Estimation](mixri_mixing_features_of_reference_images_for_novel_object_pose_estimation.md)
- [\[ICCV 2025\] Dynamic Reconstruction of Hand-Object Interaction with Distributed Force-aware Contact Representation](dynamic_reconstruction_of_hand-object_interaction_with_distributed_force-aware_c.md)

<!-- RELATED:END -->
