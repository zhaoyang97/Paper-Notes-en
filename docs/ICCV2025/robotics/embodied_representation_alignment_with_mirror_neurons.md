---
title: >-
  [Paper Note] Embodied Representation Alignment with Mirror Neurons
description: >-
  [ICCV 2025][Robotics][mirror neurons] Inspired by mirror neurons, this paper aligns the intermediate representations of action understanding (observing others' behavior) and embodied execution (autonomously performing actions) into a shared latent space via contrastive learning. The work reveals a spontaneous alignment phenomenon between the two model families that correlates with task success rate, and demonstrates that explicit alignment yields improvements on action recognition (+3.3%) and robot manipulation (+3.5%).
tags:
  - ICCV 2025
  - Robotics
  - mirror neurons
  - representation alignment
  - embodied execution
  - action understanding
  - contrastive learning
date: 2026-05-08
content_hash: 96f49414d86cb0f5
---

# Embodied Representation Alignment with Mirror Neurons

**Conference**: ICCV 2025
**arXiv**: [2509.21136](https://arxiv.org/abs/2509.21136)
**Code**: None
**Area**: Robotics / Embodied Intelligence
**Keywords**: mirror neurons, representation alignment, embodied execution, action understanding, contrastive learning

## TL;DR
Inspired by mirror neurons, this paper aligns the intermediate representations of action understanding (observing others' behavior) and embodied execution (autonomously performing actions) into a shared latent space via contrastive learning. The work reveals a spontaneous alignment phenomenon between the two model families that correlates with task success rate, and demonstrates that explicit alignment yields improvements on action recognition (+3.3%) and robot manipulation (+3.5%).

## Background & Motivation
- **Background**: Neuroscience has identified mirror neurons that activate both during observation and execution of the same action, revealing an intrinsic connection between action understanding and action execution.
- **Limitations of Prior Work**: Current machine learning approaches treat action understanding (e.g., video action recognition) and embodied execution (e.g., robot manipulation) as independent tasks trained in isolation, ignoring their complementary nature.
- **Key Challenge**: Biological systems mutually reinforce both capabilities through shared representations (embodied cognition theory), whereas independently trained ML models lack representational generalizability and completeness.
- **Core Problem**: Whether observation and execution neural representations can be explicitly aligned—analogous to biological mirror neurons—to achieve mutual benefit.
- **Key Insight**: Modeling both capabilities from a unified representation learning perspective by first probing spontaneous alignment and then explicitly promoting it.
- **Core Idea**: Two linear layers map representations from both model families into a shared space, with an InfoNCE contrastive loss enforcing alignment between representations of corresponding actions.

## Method

### Overall Architecture
The framework jointly trains an action understanding model $\mathcal{U}$ (ViCLIP video encoder) and an embodied execution model $\mathcal{E}$ (ARP robot policy network). In addition to their respective original task losses, an alignment loss is introduced. Two linear layers project intermediate representations into a shared latent space $\mathbb{Z} \subset \mathbb{R}^{1024}$, and bidirectional InfoNCE contrastive learning is applied for alignment.

### Key Designs
1. **Alignment Probing**:

    - **Function**: Probes the degree of alignment in existing model representations by training two linear transformations without modifying the original models.
    - **Mechanism**: The pretrained $\mathcal{U}$ and $\mathcal{E}$ are frozen; only $\mathcal{T}_u$ and $\mathcal{T}_e$ are trained to minimize the bidirectional InfoNCE loss. Recall@1 is used to measure alignment.
    - **Design Motivation**: To validate two core hypotheses—(1) whether independently trained models spontaneously produce representational alignment; and (2) whether alignment degree correlates with task success rate.

2. **Mirror Neuron Alignment Module**:

    - **Function**: Explicitly aligns intermediate representations of both models during joint training.
    - **Mechanism**: The total loss is $\mathcal{L}_{\text{final}} = \mathcal{L}_{\text{AU}} + \lambda_{\text{EE}} \mathcal{L}_{\text{EE}} + \lambda_{\text{align}} \mathcal{L}_{\text{align}}$, where the alignment loss is the bidirectional InfoNCE: $\mathcal{L}_{\text{align}} = -\frac{1}{2B}\sum_{i=1}^{B}[\log\frac{\exp(\text{sim}(\mathbf{z}_u^{(i)}, \mathbf{z}_e^{(i)})/\tau)}{\sum_j \exp(\text{sim}(\mathbf{z}_u^{(i)}, \mathbf{z}_e^{(j)})/\tau)} + \text{symmetric term}]$
    - **Design Motivation**: From an information-theoretic perspective, this is equivalent to maximizing a lower bound on the mutual information between the action understanding representation $\mathbf{u}$ and the embodied execution representation $\mathbf{e}$.

3. **Positive Pair Construction Strategy**:

    - **Function**: Defines which observation–execution pairs should serve as positive samples in contrastive learning.
    - **Mechanism**: Three granularity levels are explored—by Episode (same trajectory), by Instruction (same instruction but different scenes), and by Task (same task category).
    - **Design Motivation**: Instruction-level pairing is the optimal trade-off, maintaining semantic consistency while introducing variation, thereby avoiding overly strict or overly loose alignment.

### Loss & Training
- Action understanding: video–text contrastive learning (ViCLIP's original objective)
- Embodied execution: next-action prediction (ARP's original objective)
- Alignment loss weight $\lambda_{\text{align}} = 0.5$, $\lambda_{\text{EE}} = 1$
- Temperature parameter $\tau = 0.1$
- Alignment layer learning rate $1 \times 10^{-4}$

## Key Experimental Results

### Main Results

| Task | Metric | Ours (MN) | Baseline | Gain |
|------|--------|-----------|----------|------|
| Action Recognition (avg. 18 tasks) | Accuracy | 74.9% | 71.6% (ViCLIP finetune) | +3.3% |
| Robot Manipulation (avg. 18 tasks) | Success Rate | 88.8% | 85.3% (ARP) | +3.5% |
| Sort Shape | Success Rate | 72.0% | 56.0% | +16.0% |
| Stack Cup | Success Rate | 93.3% | 82.7% | +10.6% |
| Sweep Dust | Success Rate | 80.0% | 69.3% | +10.7% |

### Ablation Study

| Configuration | AU Acc | EE SR | Note |
|---------------|--------|-------|------|
| By Episode, τ=0.1 | 72.9 | 88.1 | Same-trajectory pairing |
| By Instruction, τ=0.1 (default) | 74.9 | 88.8 | Same-instruction pairing, best |
| By Class, τ=0.1 | 71.6 | 85.7 | Same-category pairing, too loose |
| By Instruction, τ=0.02 | 74.9 | 86.7 | Low temperature, overly strict alignment |
| By Instruction, τ=0.2 | 77.1 | 87.0 | High temperature, better AU but lower EE |

### Key Findings
- Independently trained models exhibit spontaneous representational alignment, with retrieval accuracy exceeding 60% using only linear transformations and contrastive learning.
- The alignment degree of successful task subsets is significantly higher than that of failure subsets, suggesting a positive correlation between alignment and quality.
- Explicit alignment yields the greatest gains on tasks requiring fine-grained manipulation reasoning (Sort Shape, Stack Cup).
- t-SNE visualizations show that the MN method not only promotes cross-model alignment but also enhances the discriminability of fine-grained instructions.

## Highlights & Insights
- Biologically inspired yet practically simple: the mirror neuron mechanism is elegantly reduced to "two linear layers + contrastive loss."
- The probe-then-apply research paradigm is instructive: hypotheses are first validated via probing, then methods are designed, forming a complete causal chain.
- A positive correlation between representational alignment and task success rate is identified, providing empirical evidence for *why* alignment is beneficial.
- The findings connect to the Platonic Representation Hypothesis—models trained with different objectives tend to converge toward a shared statistical model of reality.

## Limitations & Future Work
- Both action understanding and embodied execution data originate from the same simulated environment (RLBench); real-world modality gaps would be considerably larger.
- Only linear transformations are used for alignment; nonlinear mappings may capture more complex cross-modal relationships.
- Constructing positive pairs requires shared semantic labels (language instructions), leaving alignment strategies for unlabeled settings unexplored.
- The exploration of alignment granularity is limited to three strategies; hierarchical alignment (coarse-to-fine) warrants further investigation.
- The impact of multi-sensory inputs (tactile, auditory) on representational alignment is not explored, despite biological mirror neuron systems being inherently multimodal.
- Joint training requires paired data from both models simultaneously, which may be difficult to obtain in practical deployment scenarios.

## Related Work & Insights
- **Mirror Neurons**: This biological mechanism is systematically introduced into embodied AI representation learning for the first time.
- **ViCLIP**: A video–text foundation model serving as the backbone for action understanding after fine-tuning.
- **ARP**: An autoregressive policy network incorporating MVT for multi-view input processing.
- **Platonic Representation Hypothesis**: Provides theoretical support for the tendency of models trained on different modalities/tasks to converge toward shared representations.
- **Insight**: Any two models processing the same underlying reality may mutually benefit through representational alignment—a paradigm generalizable to a broader range of tasks.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Mirror neuron perspective applied to embodied intelligence; the probe-then-apply research paradigm is distinctive.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 18 manipulation tasks + action recognition evaluation + ablation + representation visualization, though validated only in simulation.
- Writing Quality: ⭐⭐⭐⭐⭐ — Narrative flows smoothly from neuroscience to method design; figures and tables are well crafted.
- Value: ⭐⭐⭐⭐ — Proposes a unified representation learning paradigm connecting perception and action, with meaningful implications for embodied AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] To Align or Not to Align: Strategic Multimodal Representation Alignment for Optimal Performance](../../AAAI2026/robotics/to_align_or_not_to_align_strategic_multimodal_representation_alignment_for_optim.md)
- [\[ICCV 2025\] Rep-MTL: Unleashing the Power of Representation-Level Task Saliency for Multi-Task Learning](rep-mtl_unleashing_the_power_of_representation-level_task_saliency_for_multi-tas.md)
- [\[ICCV 2025\] EvolvingGrasp: Evolutionary Grasp Generation via Efficient Preference Alignment](evolvinggrasp_evolutionary_grasp_generation_via_efficient_preference_alignment.md)
- [\[ICCV 2025\] TesserAct: Learning 4D Embodied World Models](learning_4d_embodied_world_models.md)
- [\[NeurIPS 2025\] NeSyPr: Neurosymbolic Proceduralization For Efficient Embodied Reasoning](../../NeurIPS2025/robotics/nesypr_neurosymbolic_proceduralization_for_efficient_embodied_reasoning.md)

</div>

<!-- RELATED:END -->
