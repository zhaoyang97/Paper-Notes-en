---
title: >-
  [Paper Note] Learning What Matters: Dynamic Dimension Selection and Aggregation for Interpretable Vision-Language Reward Modeling
description: >-
  [ACL 2026][Interpretability][Visual-Language Reward Models] VL-MDR upgrades "single-scalar black-box" discriminative vision-language reward models into a triple-head architecture featuring "dynamic dimension selection +…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "Visual-Language Reward Models"
  - "Multi-dimensional Evaluation"
  - "Dynamic Gating"
  - "DPO Alignment"
date: 2026-05-08
content_hash: 4d540ccd44b7bd87
---

# Learning What Matters: Dynamic Dimension Selection and Aggregation for Interpretable Vision-Language Reward Modeling

**Conference**: ACL 2026  
**arXiv**: [2604.05445](https://arxiv.org/abs/2604.05445)  
**Code**: TBD  
**Area**: Interpretability / Multimodal Reward Modeling / RLHF  
**Keywords**: Visual-Language Reward Models, Multi-dimensional Evaluation, Dynamic Gating, DPO Alignment, Interpretability

## TL;DR
VL-MDR upgrades "single-scalar black-box" discriminative vision-language reward models into a triple-head architecture featuring "dynamic dimension selection + per-dimension scoring + adaptive weighting." Coupled with a 321k-sample dataset featuring 21-dimensional fine-grained preference annotations, it outperforms existing open-source RMs on VL-RewardBench and produces higher-quality DPO preference pairs to mitigate VLM hallucinations.

## Background & Motivation
**Background**: Multimodal reward models (RM) are critical infrastructure for LVLM alignment. Existing approaches primarily fall into two categories: generative RMs (e.g., LLaVA-Critic), which use natural language for critiques and scoring (interpretable but slow and prone to positional bias), and discriminative RMs (e.g., Skywork-VL), which directly regress a scalar score (high throughput but entirely black-box).

**Limitations of Prior Work**: Discriminative RMs compress orthogonal dimensions such as "image fidelity, spatial reasoning, style, and safety" into a single scalar. This prevents distinguishing whether a response failed due to "misinterpretation of the image (perceptual failure)" or "correct perception but incorrect reasoning (reasoning failure)." Such coarse-grained feedback leaves downstream RLHF/DPO without clear signals for optimization.

**Key Challenge**: Interpretability requires multi-dimensional output signals, while efficiency demands single-forward passes without lengthy text generation—two requirements that remain irreconcilable on the "scalar vs. text critique" axis. Furthermore, multimodal tasks are query-dependent; a geometry problem does not require "stylistic quality," and an artistic image does not require "code reasoning." Fixed weighing fails to adapt to these needs.

**Goal**: (1) Design a reward model that mimics human reviewers by "first identifying necessary ability dimensions for a task, scoring each dimension individually, and then aggregating them with weights"; (2) Ensure the entire process completes in a **single forward pass** to retain the efficiency of discriminative RMs; (3) Provide a large-scale preference dataset supporting such fine-grained supervision.

**Key Insight**: The authors observe that multimodal evaluation is naturally "hierarchical and conditionally dependent"—evaluation criteria should be determined solely by the instruction (image + text), while scoring should be determined by the response. This Query-Response Decoupling serves as the theoretical foundation for the dynamic gating architecture.

**Core Idea**: Use the instruction side to predict "which dimensions are relevant and their respective weights," and use the response side to score each dimension independently. Applying masked weighted summation yields an interpretable scalar reward within a single forward pass.

## Method

### Overall Architecture
VL-MDR implements three lightweight heads on a shared pretrained VLM backbone, following the **Query-Response Decoupling** principle:

- **Input**: A multimodal instruction $x$ (image + text) and a pair of candidate responses $(y_A, y_B)$.
- **Three Signal Paths**:
    - Instruction Path → Dimension Prediction Head: Outputs correlation probabilities for $K=21$ dimensions; Top-$k$ selections determine active dimensions.
    - Instruction Path → Dimension Weighting Head: Outputs normalized weights for the selected dimensions.
    - Response Path → Scoring Head: Outputs scores $s_k(y) \in \mathbb{R}$ for each dimension for every candidate response.
- **Aggregation**: A weighted sum is calculated over the masked active dimensions $\mathcal{S}$ as $R(y) = \sum_{k \in \mathcal{S}} w_k \cdot s_k(y)$, resulting in the final scalar reward.
- **Output**: A scalar reward per response alongside a 21-dimensional fine-grained scoring vector (naturally interpretable).

### Key Designs

1. **Visual-Aware Dynamic Dimension Selection (Dimension Prediction Head)**:
    - **Function**: Predicts the relevance probability of each dimension based on instruction $x$ as $\hat{z}_k = \sigma(f_{\text{dim}}(h_x))_k$, and selects the active dimension set $\mathcal{S}$ via Top-$k$.
    - **Mechanism**: Modeling "which dimensions should be evaluated" as a multi-label classification problem, supervised by gold labels $z_k$ from a 21-dimensional taxonomy. Unlike MoE routing which selects experts, this selects "scoring dimension subsets," internalizing interpretability into the structure.
    - **Design Motivation**: Fixing all 21 dimensions would introduce noise (e.g., calculating a "geometry" score for an art piece). Dynamic Top-$k$ selection reduces computational redundancy and ensures reward factorization aligns with human intuition.

2. **Fine-grained Multi-dimensional Scoring (Scoring Head)**:
    - **Function**: Outputs independent scores $s_k(y)$ for all 21 dimensions for a candidate response $y$, where only scores within $\mathcal{S}$ participate in aggregation.
    - **Mechanism**: A 21-way parallel lightweight MLP head reads response-side hidden states. During training, sparse labels $\mathbf{p} \in \{1,0,-1\}^K$ apply Bradley-Terry preference loss only on dimensions where $z_k=1$: $\mathcal{L}_{\text{pref}} = -\log \sigma(s_k(y_A) - s_k(y_B)) \cdot \mathbb{1}[p_k = 1]$.
    - **Design Motivation**: Sparse supervision avoids meaningless signals (like scoring geometry for art), allowing each dimension head to learn only from truly relevant samples.

3. **Adaptive Masked Aggregation (Dimension Weighting Head)**:
    - **Function**: Outputs softmax weights $w_k = \mathrm{softmax}_{\mathcal{S}}(f_w(h_x))_k$ for the selected dimensions to fuse multi-dimensional scores into a final scalar.
    - **Mechanism**: Weight calculation depends solely on the instruction (decoupled from the response), ensuring $y_A$ and $y_B$ are compared using identical weights for the same query. The final reward is $R(y) = \sum_{k \in \mathcal{S}} w_k s_k(y)$.
    - **Design Motivation**: Dimension importance varies significantly across tasks—"numerical calculation" should dominate in math, while "harm detection" should act as a veto in safety scenarios. Fixed or global weights cannot capture this conditional dependence.

### Loss & Training
The total loss is optimized via three terms:
- **Dimension Correlation Loss**: 21-dimensional BCE, $\mathcal{L}_{\text{dim}} = \mathrm{BCE}(\hat{\mathbf{z}}, \mathbf{z})$.
- **Fine-grained Preference Loss**: Masked Bradley-Terry, $\mathcal{L}_{\text{fine}} = \sum_k \mathbb{1}[z_k=1] \cdot \mathrm{BT}(s_k(y_A), s_k(y_B), p_k)$.
- **Overall Preference Loss**: Applied to the final aggregated scalar, $\mathcal{L}_{\text{overall}} = \mathrm{BT}(R(y_A), R(y_B), o)$.

**Data**: 321k preference pairs were constructed starting from 7 public VLM preference datasets (VLFeedback, RLAIF-V, SPA-VL, etc., totaling 414.2k). Three strong VLM judges (Qwen3-VL-235B, GLM-4.5V, InternVL3-78B) were used for multi-model fine-grained overall-consistency filtering (77.6% retention). Each sample targets the top-3 relevant dimensions across a taxonomy of 7 core capabilities $\times$ 3 sub-dimensions = 21 dimensions.

## Key Experimental Results

### Main Results
VL-MDR was compared against open-source RMs on VL-RewardBench and two other multimodal RM benchmarks. Downstream LVLMs were trained via DPO using VL-MDR generated preference pairs to evaluate hallucination mitigation.

| Setting | Benchmark | Key Metric | VL-MDR | Prev. SOTA | Trend |
|---------|-----------|------------|--------|------------|-------|
| Direct RM Eval | VL-RewardBench | Overall Acc | Significant Lead | Skywork-VL / LLaVA-Critic | Beats both discr. & gener. |
| Direct RM Eval | Comprehensive Bench | Category Avg | Stable Lead | Balanced across categories | No performance drop |
| DPO Alignment | Hallucination Suite | Hallucination Rate↓ | Significantly Superior | Using original pairs | Validates fine-grained value |
| Efficiency | Inference Latency | Single Forward | ≈ Discr. RM | Faster than Gen. RM | Maintains discr. throughput |

### Ablation Study

| Configuration | Key Metric Trend | Description |
|---------------|------------------|-------------|
| Full VL-MDR | Optimal | 3 heads + Top-$k$ selection + adaptive weights |
| w/o Dynamic Selection (Full 21 dims) | Significant Drop | Irrelevant dimensions introduce noise; proves gating necessity |
| w/o Adaptive Weighting (Uniform) | Significant Drop | Proves weights must change dynamically with instruction |
| w/o Fine-grained Loss (Overall only) | Drop | Degenerates to traditional discr. RM; loss of granularity |
| w/o Multi-model Filter | Drop | Proves data noise affects fine-grained supervision |

### Key Findings
- **Fine-grained preference loss** contributes most to performance: Removing it reverts the model to traditional discriminative RM levels, proving "per-dimension supervision" is the root cause of interpretability gains.
- **Dynamic Top-$k$ selection** outperforms "weighted full 21 dimensions": Forcing irrelevant weights to zero is more reliable than letting the model learn to suppress them.
- **DPO** with VL-MDR preference pairs significantly reduces hallucination rates: Fine-grained scores identify pairs that are "specifically worse in the hallucination dimension," which is far less noisy than "overall preference."
- The distribution of Top-3 dimension labels is highly consistent with human annotation, proving the dimension head learns meaningful "task type recognition."

## Highlights & Insights
- **Query-Response Decoupling is a structural innovation**: Explicitly encoding "criteria determined by instruction / results determined by response" prevents competition between weights and scores, making it more robust than simple multi-dimension scoring models.
- **Sparse Masked Preference Loss $\mathbb{1}[z_k=1] \cdot \mathrm{BT}$ is a critical detail**: Avoiding supervision on irrelevant dimensions prevents the "task conflict" common in multi-task learning.
- **21-dimensional Hierarchical Taxonomy**: Offers 7x the granularity of generic "quality, fluency, relevance" classifications while remaining more controllable than open-ended labels.
- **Multi-model Consistency Filtering**: Essential for fine-grained data quality. Requiring top-3 dimension consistency + overall preference consistency across three 70B+ models ensured a high-quality (though smaller) training set.

## Limitations & Future Work
- **Handcrafted Taxonomy**: The 21 dimensions are vision-centric; extending to audio or code requires redesigning the taxonomy.
- **Fixed Hyperparameter $k$**: In reality, the "relevant dimension count" should be adaptive (e.g., $k=1$ for math vs. $k=5$ for complex reasoning).
- **High Data Filtering Cost**: Reliance on three 70B+ judges is expensive and may propagate specific model biases (e.g., GPT-style sensitivity to "politeness").
- **Closed-source Comparison**: Not compared against the strongest closed-source judges (e.g., GPT-4V-as-Judge).
- **Future Work**: Transforming the dimension head into learnable prototypes (similar to MoE routing) to support open dimension sets; exploration of process reward modeling (PRM) using VL-MDR for multimodal GRPO.

## Related Work & Insights
- **vs. LLaVA-Critic (Generative RM)**: While LLaVA-Critic provides interpretability via text, it suffers from latency and positional bias; VL-MDR achieves "equivalent interpretability" via structured heads while maintaining discriminative efficiency.
- **vs. Skywork-VL (Discriminative RM)**: Skywork-VL is a black box; VL-MDR provides a "dimension decomposition" layer that identifies *why* a chosen response is superior (e.g., 30% fewer hallucinations rather than better reasoning).
- **vs. MoE Router**: Similarity exists in input-selection logic, but while MoE selects experts for different paths, VL-MDR selects "scoring subsets" for aggregation—serving as a lightweight "Structured Interpretable MoE" alternative.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Query-Response Decoupling + Dynamic Selection is a clean structural innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Includes 3 RM benchmarks, downstream DPO, and full ablation, though lacks closed-source judge comparison.
- **Writing Quality**: ⭐⭐⭐⭐ Clear problem setting and intuitive methodology diagrams.
- **Value**: ⭐⭐⭐⭐ Interpretable RMs are a high-demand necessity for VLM alignment; the 321k dataset and taxonomy offer long-term utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Mechanisms of Prompt-Induced Hallucination in Vision–Language Models](mechanisms_of_prompt-induced_hallucination_in_vision-language_models.md)
- [\[ICML 2026\] IdEst: Assessing Self-Supervised Learning Representations via Intrinsic Dimension](../../ICML2026/interpretability/idest_assessing_self-supervised_learning_representations_via_intrinsic_dimension.md)
- [\[ACL 2026\] Retrieval Heads are Dynamic](retrieval_heads_are_dynamic.md)
- [\[ACL 2026\] Aligning What LLMs Do and Say: Towards Self-Consistent Explanations](aligning_what_llms_do_and_say_towards_self-consistent_explanations.md)
- [\[NeurIPS 2025\] Rectifying Shortcut Behaviors in Preference-based Reward Learning](../../NeurIPS2025/interpretability/rectifying_shortcut_behaviors_in_preference-based_reward_learning.md)

</div>

<!-- RELATED:END -->
