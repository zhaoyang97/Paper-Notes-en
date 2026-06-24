---
title: >-
  [Paper Note] Do Language Models Use Their Depth Efficiently?
description: >-
  [NeurIPS 2025][LLM (Other)][Depth Use Efficiency] Through causal interventions, residual stream analysis, and cross-model linear mapping, it is demonstrated that the layers in the latter half of current LLMs do not participate in compositional computation but merely iteratively refine the output probability distribution. Deeper models simply stretch the computation of shallower models across more layers.
tags:
  - "NeurIPS 2025"
  - "LLM (Other)"
  - "Depth Use Efficiency"
  - "Residual Stream"
  - "Layer Contribution"
  - "Causal Intervention"
  - "Phase Transition"
date: 2026-05-08
content_hash: 4b70e75c4f19f4a7
---

# Do Language Models Use Their Depth Efficiently?

**Conference**: NeurIPS 2025  
**arXiv**: [2505.13898](https://arxiv.org/abs/2505.13898)  
**Code**: [https://github.com/robertcsordas/llm_effective_depth](https://github.com/robertcsordas/llm_effective_depth)  
**Area**: LLM/NLP  
**Keywords**: Depth Use Efficiency, Residual Stream, Layer Contribution, Causal Intervention, Phase Transition

## TL;DR
Through causal interventions, residual stream analysis, and cross-model linear mapping, it is demonstrated that the layers in the latter half of current LLMs do not participate in compositional computation but merely iteratively refine the output probability distribution. Deeper models simply stretch the computation of shallower models across more layers.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: **Background**: Modern LLMs are becoming increasingly deep, and statistics of 132 models from the Open LLM Leaderboard show that depth is positively correlated with performance (even when controlling for parameters and other factors). Theoretically, deeper models should be capable of performing more complex compositional computations—building higher-order representations on top of prior layer outputs. In practice, however, Gromov et al. found that half of the layers could be removed without significantly affecting MMLU performance, and Lad et al. found that models exhibit surprising robustness to layer-dropping and neighbor-swapping.

The Core Problem of this paper: **Do deeper models utilize extra depth to compose more features and perform higher-order computations that are impossible for shallower models, or do they merely distribute the same type of computation across more layers?**

## Technical Background
All analyzed subjects are pre-layernorm Transformers. In this architecture, each layer composed of an attention sub-layer and an MLP sub-layer interacts with the residual stream through pure addition: $h_{l+1} = h_l + a_l + m_l$. This means $a_l + m_l = h_{l+1} - h_l$ can be directly quantified as the layer's contribution to the residual stream. The residual stream is initialized as $h_0 = \text{Embedding}(x)$, and the final output is $y = \text{softmax}(\text{Norm}(h_L) W^{out})$.

Since the output norms of the sub-layers are identical at initialization (guaranteed by normalization layers) while the residual norm continuously increases and weight decay inhibits subsequent layer compensation, later layers are inherently harder to change the residual direction. This constitutes an important structural premise for understanding the phenomenon of "underutilized depth".

## Method

### Overall Architecture
Llama 3.1 (8B/70B) is used as the primary analysis subject, with Qwen 3 and OLMo 2 as auxiliary verification. Experiments are executed on GSM8K using the NDIF/NNsight platform, systematically analyzing depth utilization efficiency from five perspectives.

### Analysis 1: Residual Stream and Sub-layer Contributions
The relative contribution of each layer to the residual stream $\|a_l + m_l\|_2 / \|h_l\|_2$ and the cosine similarity between sub-layer outputs and the residual are measured. Since sub-layer interaction with the residual in pre-layernorm Transformers is additive ($h_{l+1} = h_l + a_l + m_l$), the residual norm grows with the number of layers while the sub-layer output norm grows more slowly.

**Key Findings**: The contributions of layers in the first half of the model are stable, while a significant drop occurs in the middle (especially for attention layers). Cosine similarity displays a clear **phase transition**—the first half primarily erases/writes new features (negative/zero similarity), while the second half transitions to reinforcing existing features (positive similarity).

### Analysis 2: Causal Intervention via Layer Skipping
After skipping layer $s$, the relative change in computation for all subsequent layers is measured: $\|(h_{l+1}-h_l) - (\bar{h}_{l+1}-\bar{h}_l)\|_2 / \|h_{l+1}-h_l\|_2$. The maximum value across all sequence positions and multiple prompts is taken.

**Impact on the current token**: The mutual influence between layers in the second half is significantly lower than in the first half, though each layer remains important for the final output prediction—indicating that the second half executes independent, non-compositional distribution-refinement operations.

**Impact on future tokens**: Only layers at the current and preceding positions are skipped, and the impact on future positions is measured. The results are even more dramatic—layers in the second half have almost no impact on future computations, indicating that these layers do not yield reusable intermediate results.

### Analysis 3: Logitlens Verification
Logitlens is applied to the residual of each layer (using the output classifier directly to read intermediate representations) to measure the KL divergence from the final prediction and the overlap of top-5 predictions.

KL divergence drops sharply starting in the middle of the network, and top-5 overlap rises drastically, aligned with the position of the phase transition—confirming that the second half indeed only iteratively fine-tunes the probability distribution. This finding is independent of causal intervention results, providing complementary validation for "distribution-refinement in the second half".

### Analysis 4: Relationship between Depth and Problem Complexity
Define the **depth score** $d = \sum_{l=1}^{L} l \cdot e_l / \sum_m e_m$, where $e_l$ is the importance of the $l$-th layer. Analysis is performed on two datasets:
- **MQuAKE** (Multi-hop reasoning): 2-hop to multi-hop, with no change in depth score
- **MATH** (Different difficulty levels): Increased difficulty does not lead to greater utilization of deeper layers

Individual case analysis (integrated gradients + residual erasing intervention) also shows that all input tokens are equally important before the middle of the network, and subsequent steps of multi-step computations are not delayed to deeper layers, providing **no evidence of compositional computation**.

Specific implementation of residual erasing: Replace the residual at position $t$ of layer $l$ with the multi-sample mean $\tilde{h}_l$, and measure the maximum change in the predicted answer token $\|y - \bar{y}\|_2$.

### Analysis 5: Cross-Model Linear Mapping
Using independently trained Qwen 2.5 1.5B (28 layers) and 14B (48 layers), a linear mapping $f_{lm}$ is trained for each pair of layers to measure the relative prediction error $\|h_l^{14B} - f_{lm}(h_m^{1.5B})\|_2 / \|h_l^{14B}\|_2$.

The results exhibit a clear **diagonal pattern**: layers with the same relative depth map best to each other (e.g., layer 14 of 1.5B maps best to layer 24 of 14B). This directly proves that deep models are merely "stretched" versions of shallow models, rather than executing brand-new types of computation. If deep models performed novel computations in their second half that shallow models do not possess, those layers should be unpredictable from the shallow model—which is not the case.

## Key Experimental Results

### Main Results

| Analysis Method | Model | Dataset | Core Finding |
|---------|------|--------|---------|
| Relative Contribution | Llama 3.1 70B | GSM8K | First half >0.15, second half <0.05, dropping sharply in the middle |
| Layer Skipping (current token) | Llama 3.1 70B | GSM8K | Low mutual influence in the second half, but each layer remains important for output |
| Layer Skipping (future token) | Llama 3.1 70B | GSM8K | Second half layers have almost no impact on the future |
| Depth Score | Llama 3.1 70B | MQuAKE/MATH | Increased complexity does not alter computation depth |
| Linear Mapping | Qwen 2.5 1.5B→14B | General | Diagonal pattern, deep model = stretched computation |

## Exploratory Experiment: MoEUT
Performance of standard Transformers is compared with MoEUT (a parameter-shared Universal Transformer, 244M parameters) on the DeepMind Math dataset. An additional "non-modeled question" variant is introduced—where loss is not calculated for tokens in the question part during training.

**Results**:
- When not modeling questions, MoEUT significantly utilizes deeper layers, calculation depth increases with steps, and extrapolation accuracy improves from 36% to 63%
- Standard Transformers show smaller gains (41% → 48%)
- Residual erasing visualization shows that MoEUT's depth usage is more input-dependent
- Models trained from scratch all exhibit a trend of increasing depth with computational steps, but fine-tuning pre-trained models fails to alter this behavior

This suggests that: (1) parameter-shared architectures might utilize depth more effectively; (2) the pre-training objective (modeling unpredictable questions) might be one of the factors leading to depth underutilization.

## Highlights & Insights
1. **Unified Phase Transition**: The drop in residual contribution, sign change of cosine similarity, convergence of Logitlens predictions, and disappearance of future token impact all occur at the same position in the middle of the network, forming a consistent picture of evidence.
2. **The Essence of the Second Half is Distribution Refinement**: They perform important but independent operations, without participating in compositional computation or generating reusable intermediate results.
3. **Deep Models ≠ More Complex Computation**: The diagonal pattern of cross-model linear mapping directly proves that deep models are merely "stretched".
4. **Consequences of Fixed-depth Computation**: The model does not adjust computation depth based on problem complexity; all problems are processed using a fixed circuit. This might explain the diminishing returns of scaling.
5. **Implications for CoT and Latent Thinking**: CoT bypasses depth limitations by outsourcing compositional computation to the input/output space; if depth insensitivity in latent thinking methods is caused by the training objectives, such approaches may not work at all.
6. **Residual Width May Be the Bottleneck**: Independent operations in the second half imply that all information must reside in the residual simultaneously; thus, $d_{model}$ might be the true capacity bottleneck.

## Limitations & Future Work
- The analysis is mainly conducted on 10 random samples of GSM8K, which is a small sample size.
- The exploratory experiment of MoEUT only uses 4 exemplars and a 244M-parameter model, which possesses limited scale.
- Linear mapping assumes layer-wise correspondences can be captured by linear transformations, potentially missing non-linear correspondences.
- Interpretation of cosine similarity is limited under feature superposition.
- The study primarily focuses on the mathematical domain (due to its high sensitivity to layer interventions); conclusions in other domains might differ.

## Related Work & Insights
- **Lad et al.**: proposed four stages of inference (detokenization -> feature engineering -> ensemble prediction -> residual sharpening) and found that intermediate layers are robust to layer skipping but did not study the effect of input complexity. This work complements this with causal intervention analysis and demonstrates that layers in the second half have almost no impact on future tokens.
- **Gromov et al.**: demonstrated that half of the layers could be removed without significantly affecting MMLU, with mathematics being an exception. This work further reveals that the specific function of these "removable" layers is distribution refinement.
- **Skean et al.**: identified an information bottleneck in the middle of autoregressive Transformers, where intermediate layer representations often outperform final layers on downstream tasks, which is consistent with the phase transition observed in this work.
- **Sun et al.**: compared Transformer layers to painters, finding that layers are exchangeable, which supports the notion of compute independence between layers.
- **Petty et al.**: found that increasing depth does not aid compositional generalization, corroborating this study's conclusion that depth does not yield brand-new computations.

## Rating
⭐⭐⭐⭐ (4/5)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] From Data to Knowledge: Evaluating How Efficiently Language Models Learn Facts](../../ACL2025/llm_nlp/from_data_to_knowledge_evaluating_how_efficiently_language_models_learn_facts.md)
- [\[ICLR 2026\] When Language Models Lose Their Mind: The Consequences of Brain Misalignment](../../ICLR2026/llm_nlp/when_language_models_lose_their_mind_the_consequences_of_brain_misalignment.md)
- [\[NeurIPS 2025\] In-Context Learning of Linear Dynamical Systems with Transformers: Approximation Bounds and Depth-Separation](in-context_learning_of_linear_dynamical_systems_with_transformers_approximation_.md)
- [\[ACL 2025\] Do Language Models Understand Honorific Systems in Javanese?](../../ACL2025/llm_nlp/do_language_models_understand_honorific_systems_in_javanese.md)
- [\[ACL 2025\] CogSteer: Cognition-Inspired Selective Layer Intervention for Efficiently Steering Large Language Models](../../ACL2025/llm_nlp/cogsteer_cognition-inspired_selective_layer_intervention_for_efficiently_steerin.md)

</div>

<!-- RELATED:END -->
