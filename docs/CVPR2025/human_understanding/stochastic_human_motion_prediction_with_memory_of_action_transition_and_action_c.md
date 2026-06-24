---
title: >-
  [Paper Note] Stochastic Human Motion Prediction with Memory of Action Transition and Action Characteristic
description: >-
  [CVPR 2025][Human Understanding][Stochastic Human Motion Prediction] To address the challenges of unsmooth action transitions and the difficulty in learning action characteristics in action-driven stochastic human motion prediction, this paper proposes two memory modules: Soft-Transition Action Bank (STAB) and Action Characteristic Bank (ACB), along with an Adaptive Attention Adjustment (AAA) strategy for feature fusion. The proposed method achieves SOTA performance on four d…
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "Stochastic Human Motion Prediction"
  - "Action-driven"
  - "Memory Bank"
  - "Action Transition"
  - "CVAE"
date: 2026-05-08
content_hash: 6c567ea10d1d4db3
---

# Stochastic Human Motion Prediction with Memory of Action Transition and Action Characteristic

**Conference**: CVPR 2025  
**arXiv**: [2507.04062](https://arxiv.org/abs/2507.04062)  
**Code**: [https://hyqlat.github.io/STABACB.github.io/](https://hyqlat.github.io/STABACB.github.io/)  
**Area**: Human Understanding / Motion Prediction  
**Keywords**: Stochastic Human Motion Prediction, Action-driven, Memory Bank, Action Transition, CVAE

## TL;DR

To address the challenges of unsmooth action transitions and the difficulty in learning action characteristics in action-driven stochastic human motion prediction, this paper proposes two memory modules: Soft-Transition Action Bank (STAB) and Action Characteristic Bank (ACB), along with an Adaptive Attention Adjustment (AAA) strategy for feature fusion. The proposed method achieves SOTA performance on four datasets: GRAB, NTU, BABEL, and HumanAct12.

## Background & Motivation

**Background**: Action-driven stochastic human motion prediction aims to generate diverse future motion sequences based on observed historical motion sequences and specified future action labels. This task is widely used in fields such as virtual reality and human-computer interaction. Existing methods (such as WAT) have proposed frameworks to control future motion based on action labels, but still suffer from notable limitations.

**Limitations of Prior Work**: (1) Generating smooth transition motions is highly difficult because transition speeds vary significantly between different actions. For instance, the transition from "warming up" to "drinking water" is completely different from the transition from "warming up" to "throwing". (2) Action characteristics are hard to learn due to standard high similarity between certain actions; for example, parts of "drinking water" and "raising hands" are extremely close, making it difficult for the model to distinguish them finely.

**Key Challenge**: Existing methods do not fully utilize action transition information and action characteristic information. Transition information is critical for generating smooth action connections, while action characteristic information is the key basis for distinguishing similar actions. These two types of information play different roles in different prediction phases: more transition information is required in the beginning, and more action characteristic information is needed in the later stage.

**Goal**: Design specialized memory mechanisms to store and retrieve action transition information and action characteristics, and adaptively utilize these two types of information during the prediction process.

**Key Insight**: Taking inspiration from the concept of memory banks, action transition characteristics and action characteristics are stored in two structured memory banks separately and retrieved using a key-value mechanism. At the same time, a soft search method is designed to handle the classification uncertainty of observed actions.

**Core Idea**: Utilize two memory banks to separately store action transition patterns and action inherent characteristics, cooperating with adaptive weight adjustment to dynamically switch the focus at different stages of prediction, thereby achieving smoother transitions and more accurate action generation.

## Method

### Overall Architecture

The model consists of two main components: the Action Recognition Module (ARM) and the Motion Prediction Module (MPM). ARM is based on GRU to classify the actions of the input observed motion sequences. MPM is based on a conditional VAE (CVAE) architecture for motion prediction, which integrates two memory banks, STAB and ACB, and the AAA fusion strategy. The training pipeline consists of two steps: first training the ARM until convergence, and then freezing the ARM parameters to train the MPM. The inputs are the historical pose sequence $\mathbf{X} \in \mathbb{R}^{K \times N}$ and the target action label $\mathbf{a}$, and the output is the predicted future motion $\hat{\mathbf{Y}} \in \mathbb{R}^{K \times T}$.

### Key Designs

1. **Soft-Transition Action Bank (STAB)**:

    - **Function**: Programmed to store transition information between different actions, assisting in the generation of smooth and natural action connections.
    - **Mechanism**: STAB is a key-value memory structure indexed by (past action, future action). Each element $\mathbf{S}^{\hat{a}_p, a_f}$ contains M (key, value) tuples. Retrieval consists of two steps: first, using action labels as indices to locate the corresponding element; second, computing the similarity between the query (encoder output) and each key, selecting the most similar value, and weighting retrieve outputs by similarity. The innovation lies in the "soft search": instead of using the top-1 classification result of ARM, it takes the top-k classification results and uses their respective softmax probabilities to weight the retrieval results of multiple branches for fusion, i.e., $\mathbf{F}_{st} = \sum_{j=1}^{k} w_p^{(j)} \cdot \mathbf{F}_{st}^{(j)}$.
    - **Design Motivation**: Since different actions share similar parts, the observed sequence might be classified into multiple candidate categories. Soft search allows the model to attend to multiple possibilities of the observed action, preventing the retrieval of incorrect transition patterns due to a single classification error.

2. **Action Characteristic Bank (ACB)**:

    - **Function**: Stores the inherent motion characteristics of each action, providing finer action semantic information for long-sequence prediction.
    - **Mechanism**: ACB is indexed by the future action label $a_f$. Its structure is similar to STAB but uses only a single label index. Each action corresponds to N (key, value) tuples, and the action characteristics $\mathbf{F}_{ac}$ are also obtained via similarity search and weighted retrieval. Compared to STAB, ACB does not focus on transition information, but instead centers on the motion patterns of the target action itself.
    - **Design Motivation**: Having transition information alone is insufficient to maintain action accuracy over longer prediction periods. ACB provides the "identity card" of the action, helping the model generate motion that conforms to target action characteristics in the later stage of prediction.

3. **Adaptive Attention Adjustment (AAA)**:

    - **Function**: Dynamically adjusts the fusion ratio of the output features of STAB and ACB at different prediction time steps.
    - **Mechanism**: Uses a parameter $\alpha$ to weighted-fuse the outputs of the two banks: $\mathbf{F} = \frac{\alpha}{1+\alpha} \mathbf{F}_{st} + \frac{1}{1+\alpha} \mathbf{F}_{ac}$. The value of $\alpha$ is dynamically calculated based on the ARM's classification cross-entropy loss on the predicted frames—when the classification is accurate (meaning transition is completed), $\alpha$ decreases (focusing more on action characteristics); otherwise, $\alpha$ is higher (focusing more on transition information). To avoid severe fluctuations, a running-mean method is used to smooth the change of $\alpha$, and a time threshold $\tau$ is set such that adjustments only start after predicting more than $\tau$ steps.
    - **Design Motivation**: The early stage of prediction needs to focus on transition information to ensure smooth action connection, while the later stage of prediction needs to focus on action characteristics to ensure the generation of accurate target actions. This time-varying weight allocation aligns with the physical laws of human movement.

### Loss & Training

ARM is trained for 500 epochs using cross-entropy loss, where the CE loss is computed for each frame greater than the threshold $\tau$. MPM is trained for 500 epochs with a loss function including the reconstruction loss $\mathcal{L}_{rec}$ of CVAE, the KL divergence loss $\mathcal{L}_{KL}$, and the ARM classification cross-entropy loss on the predicted motion sequences. Both phases use the ADAM optimizer with an initial learning rate of 0.002.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | Prev. SOTA | Gain |
|--------|------|------|------------|------|
| GRAB | Acc↑ | **95.23** | 92.6 | +2.63 |
| GRAB | FID_tr↓ | **43.39** | 44.59 | -1.20 |
| GRAB | Div_w↑ | **1.14** | 1.10 | +0.04 |
| NTU | Acc↑ | **80.50** | 76.0 | +4.50 |
| NTU | FID_tr↓ | **65.11** | 72.18 | -7.07 |
| BABEL | Acc↑ | **55.37** | 49.6 | +5.77 |
| BABEL | FID_tr↓ | **20.35** | 22.54 | -2.19 |
| HumanAct12 | Acc↑ | **61.57** | 59.0 | +2.57 |
| HumanAct12 | FID_tr↓ | **112.85** | 129.95 | -17.10 |

### Ablation Study

| Configuration | Acc↑ | FID_tr↓ | Div_w↑ | Note |
|------|------|---------|--------|------|
| Full model | **95.23** | **43.39** | **1.14** | Full model |
| w/o AAA | 91.93 | 44.00 | 1.10 | Remove adaptive adjustment, Acc drops by 3.3% |
| w/o STAB | 92.18 | 43.97 | 1.11 | Remove transition bank, Acc drops by 3.05% |
| w/o ACB | 93.45 | 43.61 | 1.13 | Remove action characteristic bank, smallest impact |
| w/o RM (running-mean) | 90.84 | 48.99 | 1.11 | Remove smoothing, FID rises significantly |

### Key Findings

- **Running-mean is crucial for stable training**: Removing running-mean degrades the FID from 43.39 to 48.99, demonstrating that non-smooth changes of $\alpha$ severely disrupt the training process.
- **top-k=2 is the optimal setting for soft search**: When k=1, performance is poor without soft search. While k=3/4 offers higher diversity, the accuracy drops instead, indicating that too many candidates introduce noise.
- **STAB and ACB are complementary**: STAB mainly impacts transition quality and accuracy, while ACB primarily influences generation precision but has little effect on diversity.
- **Greatest improvements on NTU and BABEL datasets**: These two datasets contain more combinations of different actions, showing that this method has a clearer advantage in complex action transition scenarios.

## Highlights & Insights

- **Clever soft search mechanism**: Weighting multiple retrieval paths using ARM's top-k classification probabilities elegantly handles action classification uncertainty. This concept can be migrated to any scenario requiring memory retrieval based on classification results.
- **Adaptive time-varying fusion**: The AAA strategy dynamically adjusts weights using classification loss as an indicator of "transition completeness," converting a time-dependent relationship that is hard to model directly into an observable signal.
- **Structured design of memory banks**: The idea of decoupling action transitions and action characteristics prevents information conflation in a single memory bank, which is transferable to other generation tasks requiring various prior knowledges.

## Limitations & Future Work

- **Increased computational overhead**: Introducing memory banks drops the FPS from 2.76 to 1.98 and increases the GPU memory footprint from 2262MB to 2837MB, which could be optimized by parallel retrieval.
- **Based only on SMPL pose parameters**: Excluding fine-grained movements like hands and face limits applicability in refined interaction scenarios.
- **Manual tuning required for the k value in soft search**: Different datasets may require different k values, and there is a lack of an adaptive method to determine k.
- Future work can explore combining memory banks with diffusion models to further improve generation quality using the denoising capabilities of diffusion processes.

## Related Work & Insights

- **vs WAT**: WAT is the direct baseline of this paper, which proposed the action-driven stochastic prediction paradigm but lacked transition modeling. By adding a memory bank mechanism to the WAT architecture, this work comprehensively outperforms its performance.
- **vs DLow**: DLow focuses on generation diversity but ignores action semantics. This work significantly improves semantic accuracy while maintaining diversity through STAB and ACB.
- **vs ACTOR**: ACTOR uses Transformers to generate specific actions but does not handle transitions. This paper focuses on natural transitions from one action to another, filling this gap.

## Rating

- Novelty: ⭐⭐⭐ The design of memory banks is relatively intuitive; the core contribution lies in problem decomposition and engineering integration.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensively verified on four datasets, meticulous ablation study, including sensitivity analysis of k value.
- Writing Quality: ⭐⭐⭐⭐ Clearly defined problem, intuitive diagrams, and structured description of the method.
- Value: ⭐⭐⭐ Valuable in the niche area (action-driven prediction), though the application scope is relatively narrow.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] SimMotionEdit: Text-Based Human Motion Editing with Motion Similarity Prediction](simmotionedit_text-based_human_motion_editing_with_motion_similarity_prediction.md)
- [\[CVPR 2026\] Gaussian-Mixture Latent Flow for Stochastic 3D Human Motion Prediction](../../CVPR2026/human_understanding/gaussian-mixture_latent_flow_for_stochastic_3d_human_motion_prediction.md)
- [\[ECCV 2024\] Bridging the Gap Between Human Motion and Action Semantics via Kinematic Phrases](../../ECCV2024/human_understanding/bridging_the_gap_between_human_motion_and_action_semantics_via_kinematic_phrases.md)
- [\[CVPR 2025\] Few-Shot Personalized Scanpath Prediction](few-shot_personalized_scanpath_prediction.md)
- [\[CVPR 2025\] Human Motion Instruction Tuning](human_motion_instruction_tuning.md)

</div>

<!-- RELATED:END -->
