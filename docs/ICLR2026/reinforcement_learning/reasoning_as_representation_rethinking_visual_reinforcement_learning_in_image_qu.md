---
title: >-
  [Paper Note] Reasoning as Representation: Rethinking Visual Reinforcement Learning in Image Quality Assessment
description: >-
  [ICLR 2026 (Oral)][Reinforcement Learning][Image Quality Assessment] Through systematic experimentation, this paper reveals the fundamental mechanism underlying the generalization capability of RL-trained reasoning-based…
tags:
  - "ICLR 2026 (Oral)"
  - "Reinforcement Learning"
  - "Image Quality Assessment"
  - "Reasoning as Representation"
  - "Contrastive Learning"
  - "Cross-Domain Generalization"
date: 2026-05-08
content_hash: ff92e0f5b5012a91
---

# Reasoning as Representation: Rethinking Visual Reinforcement Learning in Image Quality Assessment

**Conference**: ICLR 2026 (Oral)
**arXiv**: [2510.11369](https://arxiv.org/abs/2510.11369)  
**Code**: None  
**Area**: Reinforcement Learning / Image Quality Assessment
**Keywords**: Image Quality Assessment, Reinforcement Learning, Reasoning as Representation, Contrastive Learning, Cross-Domain Generalization

## TL;DR

Through systematic experimentation, this paper reveals the fundamental mechanism underlying the generalization capability of RL-trained reasoning-based IQA models — the reasoning process essentially transforms redundant visual representations into compact, cross-domain aligned textual representations. Building on this insight, the paper proposes the RALI algorithm, which directly aligns image representations to these textual representations via contrastive learning, achieving comparable generalization performance with less than 5% of the parameters and inference time.

## Background & Motivation

Image Quality Assessment (IQA) is a fundamental task in computer vision, aiming to automatically evaluate the perceptual quality of images. Recent reasoning-based IQA models, trained via reinforcement learning (RL) on multimodal large language models (MLLMs), have demonstrated **remarkable generalization capability** — maintaining high performance on unseen distortion types and datasets.

However, two critical open questions remain:

**Unclear Mechanism**: *Why* do these reasoning-based IQA models generalize? What is the specific connection between the reasoning capability endowed by RL training and generalization? Existing research remains at the level of empirical observation — knowing that it "works" but not "why it works."

**Efficiency Bottleneck**: Despite their strong performance, these models incur extremely high inference costs — requiring a full MLLM and autoregressive text generation, with energy consumption and latency **orders of magnitude higher** than traditional IQA methods, severely limiting practical deployment.

The core motivation of this paper is: **if the fundamental reason for the generalization of reasoning-based IQA models can be understood, it may be possible to retain generalization capability while substantially reducing computational overhead.**

## Method

### Overall Architecture

The work proceeds in two phases:
- **Analysis Phase**: Systematic experiments to reveal the generalization mechanism of RL-trained reasoning-based IQA models.
- **Application Phase**: Based on analytical insights, a new algorithm — RALI (Reasoning-Aligned Lightweight IQA) — is proposed to achieve efficient cross-domain generalization.

### Key Designs

1. **Core Finding: "Reasoning as Representation Transformation"**

    - **Function**: Validates, through systematic experiments, the true role of the reasoning process in IQA model generalization.
    - **Core Finding**: Through RL training, MLLMs leverage their reasoning capability to transform **redundant visual representations** into **compact, cross-domain aligned textual representations**. This transformation is precisely the source of the generalization capability exhibited by reasoning-based IQA models.
    - **Specifically**:
        - Raw visual representations (e.g., ViT features) are high-dimensional and redundant, with representations from different distortion types and domains being mutually separated.
        - The reasoning process "compresses" these visual representations into the textual representation space, and RL training causes **image quality concepts from different domains to become aligned in the textual space**.
        - This alignment is the fundamental source of generalization — similar quality levels across different domains are mapped to nearby positions in the textual representation space.
    - **Design Motivation**: If reasoning is merely one form of representation transformation rather than the reasoning content itself being important, a more efficient alternative may be found.

2. **RALI Algorithm (Reasoning-Aligned Lightweight IQA)**

    - **Function**: Proposes a lightweight algorithm to replace the full MLLM reasoning process.
    - **Mechanism**: Employs **contrastive learning** to directly align image representations to the generalizable textual representations learned by the RL-trained MLLM.
    - **Key Steps**:
        - *Step 1*: Use the RL-trained MLLM to generate reasoning text for training images and extract textual representations (as alignment targets).
        - *Step 2*: Train a lightweight image encoder via contrastive learning to directly align image representations to these textual representations.
        - *Step 3*: At inference time, only the lightweight image encoder is required — **no LLM loading, no text generation**.
    - **Design Motivation**: Since generalization stems from the cross-domain alignment of textual representations, directly endowing image representations with this alignment suffices; the reasoning process is merely an "intermediate means."

3. **Knowledge Distillation Perspective**

    - RALI can essentially be viewed as a specialized form of knowledge distillation — distilling the generalizable representation alignment capability from the large reasoning model into a lightweight model.
    - Unlike conventional distillation, however, RALI distills not the prediction outputs but the **structure of the representation space** (i.e., cross-domain alignment properties).
    - The contrastive learning objective ensures that the student model learns not only correct quality scores but, more importantly, the correspondences between quality concepts across different domains.

### Loss & Training

- **Teacher Model**: An RL-trained reasoning MLLM (e.g., Q-Instruct), used to generate cross-domain aligned textual representations.
- **Student Model**: A lightweight visual encoder (less than 5% of the teacher's parameter count), aligned to the teacher's textual representations via contrastive learning.
- **Contrastive Learning Objective**: Pulls together representations of images with the same quality level (across domains) and pushes apart representations of images with different quality levels.
- **No LLM Required at Inference**: Only the student visual encoder is needed at inference time, completely eliminating the computational overhead of the LLM.

## Key Experimental Results

### Main Results

Cross-domain generalization performance comparison on the image quality scoring task:

| Method | Generalization | Parameters | Inference Time | Notes |
|---|---|---|---|---|
| RL-based Reasoning MLLM | SOTA | 100% | 100% | Full model + reasoning |
| RALI | **Comparable** | **<5%** | **<5%** | Visual encoder only |
| Traditional IQA | Inferior | Small | Fast | Insufficient generalization |
| Non-reasoning MLLM | Moderate | Large | Medium | Lacks cross-domain alignment |

### Ablation Study

| Configuration | Key Metric | Notes |
|---|---|---|
| Visual representation space analysis | Inter-domain separation | Raw visual representations lack cross-domain alignment |
| Textual representations before RL training | Inter-domain separation | SFT-stage textual representations are not aligned |
| Textual representations after RL training | Inter-domain alignment | RL training achieves cross-domain alignment |
| Alignment to SFT representations only | Poor generalization | Confirms RL-trained alignment is the key |
| Alignment to RL representations (RALI) | Good generalization | Successfully inherits cross-domain generalization |
| Contrastive learning removed | Generalization degraded | Contrastive objective is critical for alignment |

### Key Findings

1. **The True Role of Reasoning is Representation Transformation**: The textual content within reasoning chains is not the key to generalization; rather, the key lies in the alignment across different domains that is naturally produced in the textual representation space during reasoning. This challenges the intuitive assumption that "reasoning content = source of performance."

2. **The Unique Contribution of RL Training**: Comparing textual representations before and after RL training reveals that RL training significantly improves the alignment of textual representations across different domains. Representations at the SFT stage remain inter-domain separated; genuine cross-domain alignment is only achieved through the RL stage.

3. **Substantial Efficiency Gains**: RALI reduces both model parameters and inference time to below 5% of the original while maintaining comparable generalization performance, enabling deployment in resource-constrained settings.

4. **No LLM Inference Required**: At inference time, there is no need to load an LLM or perform text generation whatsoever; only a forward pass through a lightweight visual encoder is required, greatly simplifying the deployment pipeline.

## Highlights & Insights

1. **Profound Mechanistic Insight**: The most significant contribution of this paper is not the RALI algorithm itself, but the insight that "reasoning is representation transformation." This finding changes our understanding of the generalization capability of reasoning-based models — the reasoning process is a **means** rather than an **end**, and what truly matters is the implicit restructuring of the representation space that occurs during reasoning.

2. **A Closed Loop from Theoretical Insight to Practical Algorithm**: The paper first establishes theoretical understanding through experiments (why do reasoning-based models generalize?), then designs an efficient alternative based on theory (RALI), forming a complete research loop. This paradigm has reference value for other areas as well.

3. **ICLR 2026 Oral Quality**: As an Oral paper, its contribution lies in providing an insight that may redirect research in the field — if reasoning is merely representation transformation, can the generalization of all reasoning-based models be similarly "distilled"?

4. **A New Understanding of RL Training**: The role of RL training in reasoning-based models is not merely to "teach the model to reason," but more fundamentally to "restructure the representation space" — aligning concepts across different domains. This provides a new perspective for understanding the nature of RL training.

5. **Practical Deployment Value**: An efficiency improvement of more than 20× enables IQA models to be deployed on edge devices, directly advancing real-world applications.

## Limitations & Future Work

1. **Validated Only on IQA**: Whether the core insight (reasoning = representation transformation) generalizes to other reasoning tasks (e.g., mathematical reasoning, code generation) remains unclear. IQA may be a relatively simple task, and the role of reasoning may differ in more complex settings.

2. **Quality Scoring vs. Quality Description**: RALI primarily targets quality scoring (regression) tasks. For application scenarios requiring detailed quality descriptions, a full MLLM remains necessary.

3. **Teacher Model Dependency**: RALI training depends on an RL-trained teacher MLLM to generate textual representations as alignment targets; if the teacher model is insufficiently capable, the student model's generalization will also be limited.

4. **Contrastive Learning Bottleneck**: The effectiveness of contrastive learning depends on the quality and diversity of negative samples, and the quality of cross-domain alignment may be influenced by the training data distribution.

5. **Loss of Reasoning Interpretability**: RALI eliminates the reasoning process, simultaneously forfeiting the interpretability advantage of reasoning-based models — it can no longer provide reasons or evidence for quality assessments.

## Related Work & Insights

- **Reasoning-based IQA Models**: Methods such as Q-Instruct and Q-Boost that perform quality assessment via MLLM reasoning.
- **Visual Reinforcement Learning**: Application of R1-style RL training to visual tasks.
- **Knowledge Distillation**: Knowledge transfer from large models to small models; RALI can be viewed as distillation of representation space structure.
- **Contrastive Learning**: Vision-language alignment methods such as CLIP; RALI specializes this approach for the IQA scenario.
- **Traditional IQA**: From handcrafted feature methods such as NIQE and BRISQUE to deep learning methods such as MUSIQ and DBCNN.
- **Inspiration**: The insight that **"reasoning is representation transformation"** may apply to understanding the success mechanisms of other reasoning-based models, and warrants validation across more domains.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ ("Reasoning as representation transformation" is a highly insightful finding that shifts the understanding paradigm of reasoning-based models.)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (From mechanistic analysis to algorithm design, the experimental chain is complete and rigorous, meriting Oral.)
- Writing Quality: ⭐⭐⭐⭐⭐ (The argumentation logic is clear, and the transition from insight to algorithm is natural.)
- Value: ⭐⭐⭐⭐⭐ (Offers both profound theoretical contributions and substantial practical value; a >20× efficiency gain is highly significant.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] PreferThinker: Reasoning-based Personalized Image Preference Assessment](preferthinker_reasoning-based_personalized_image_preference_assessment.md)
- [\[ICLR 2026\] DiVE-k: Differential Visual Reasoning for Fine-grained Image Recognition](dive-k_differential_visual_reasoning_for_fine-grained_image_recognition.md)
- [\[ICLR 2026\] Stackelberg Coupling of Online Representation Learning and Reinforcement Learning](stackelberg_coupling_of_online_representation_learning_and_reinforcement_learnin.md)
- [\[ICLR 2026\] RewardMap: Tackling Sparse Rewards in Fine-grained Visual Reasoning via Multi-Stage Reinforcement Learning](rewardmap_tackling_sparse_rewards_in_fine-grained_visual_reasoning_via_multi-sta.md)
- [\[ICLR 2026\] Rethinking Policy Diversity in Ensemble Policy Gradient in Large-Scale Reinforcement Learning](rethinking_policy_diversity_in_ensemble_policy_gradient_in_large-scale_reinforce.md)

</div>

<!-- RELATED:END -->
