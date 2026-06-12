---
title: >-
  [Paper Note] MultiHaluDet: Multilingual Hallucination Detection via LLM Hidden State Probing
description: >-
  [ACL2026][Hallucination Detection][Multilingual Hallucination Detection] MultiHaluDet utilizes full-layer hidden state trajectories of frozen LLMs for multi-scale sequence modeling…
tags:
  - "ACL2026"
  - "Hallucination Detection"
  - "Multilingual Hallucination Detection"
  - "Hidden State Probing"
  - "Multi-scale Attention"
  - "OOF Stacking"
  - "Cross-lingual Robustness"
date: 2026-05-08
content_hash: f0f26e63e6b464c0
---

# MultiHaluDet: Multilingual Hallucination Detection via LLM Hidden State Probing

**Conference**: ACL2026  
**arXiv**: [2605.24919](https://arxiv.org/abs/2605.24919)  
**Code**: https://github.com/alvi-uiu/MultiHaluDet  
**Area**: Hallucination Detection  
**Keywords**: Multilingual Hallucination Detection, Hidden State Probing, Multi-scale Attention, OOF Stacking, Cross-lingual Robustness

## TL;DR
MultiHaluDet utilizes full-layer hidden state trajectories of frozen LLMs for multi-scale sequence modeling, further identifying hallucinations through out-of-fold representations and ensemble meta-learners. It achieves approximately 98% AUROC on HaluEval / TriviaQA and generalizes to French, Bengali, and Amharic.

## Background & Motivation

**Background**: LLM hallucination detection is generally categorized into three types: evidence-based methods that retrieve and verify evidence, evidence-free methods based on output probabilities or consistency, and hidden-state probing methods that directly probe internal model states. The first two are limited by retrieval latency, external evidence quality, multiple sampling costs, or unreliable probability calibration. The third is more lightweight but often only examines the last layer, the last token, or a few fixed layers.

**Limitations of Prior Work**: The paper notes that hallucinations are often semantic confabulations rather than single-token low-confidence issues. Therefore, simple $P(\text{True})$, average probability, entropy, single-layer probes, or fixed token positions easily miss factual inconsistencies distributed across the entire response. This problem is more acute in non-English and low-resource languages where internal representation quality and corpus coverage are inconsistent.

**Key Challenge**: If hallucination signals gradually form along the Transformer depth, capturing only the final output or a static representation of a single layer loses dynamic information on "how the model arrived at this answer." However, reading all layers entirely introduces issues with dimensionality, model depth inconsistency, and overfitting.

**Goal**: The authors aim to build a hallucination detector that does not require fine-tuning for target languages, does not rely on external retrieval, and works across models and languages. It needs to address three sub-problems: mapping hidden states of LLMs with different depths to a unified sequence, capturing local and global depth patterns, and avoiding data leakage and overfitting during depth feature training.

**Key Insight**: Starting from "hidden state trajectories," the paper treats each hidden state layer as a sequence evolving with depth, rather than a one-time feature. The hypothesis is that the difference between factual consistency and hallucination manifests in the coupling between inter-layer norms, distributional statistics, logit confidence, and depth dynamics.

**Core Idea**: Use dynamic layer sampling + multi-scale attention + OOF stacking to transform full-depth internal trajectories of frozen LLMs into robust features for hallucination detection.

## Method

MultiHaluDet is a four-stage framework: it first extracts per-layer statistical features and global logit features from a frozen LLM, then models the depth sequence using multi-scale attention + Transformer encoder, generates leakage-free depth representations via an out-of-fold approach, and finally outputs hallucination probabilities using a stacking ensemble of multiple traditional/neural classifiers.

### Overall Architecture

The input consists of question-answer pairs $(q_i, a_i)$, with labels $y_i \in \{0,1\}$ indicating whether the answer is a hallucination. The system concatenates the QA into a structured prompt, feeds it into a frozen and quantized LLM, and obtains hidden states $\{H^{(l)}\}_{l=0}^{L}$ for all layers and the logit vector at the final position after a single forward pass. LLM parameters remain fixed.

To adapt to models of different depths, the method maps any $L$ layers to a fixed set of $K=32$ layer indices. For each sampled layer, statistics such as the last token representation, sequence mean, norm, mean, standard deviation, extreme values, sparsity, near-zero ratio, kurtosis, and MAD are extracted to form a depth sequence $S \in \mathbb{R}^{K \times d_s}$. Simultaneously, global features $g$ are constructed, including top-$k$ token probabilities, logit entropy, logit standard deviation, inter-layer norm trajectory statistics, and anchor layer features.

Subsequently, $S$ enters the sequence branch of MultiHaluDet, and $g$ enters the global MLP branch. The two-way representations are fused via gated fusion to obtain sample-level embeddings. During training, embeddings are not fed directly to the final classifier; instead, 5-fold out-of-fold training is used: the deep features for each sample come from a fold model that has not seen that sample. Finally, probabilities from multiple base classifiers are fused via a logistic meta-regressor, with the threshold selected by Youden's J statistic.

### Key Designs

1. **Dynamic Layer Sampling and Trajectory Features**:
    - **Function**: Standardizes layer counts from different LLM architectures into a fixed-length depth sequence, allowing Mistral-7B and LLaMA2-7B to share the same detector design.
    - **Mechanism**: If the model layer count equals the target, it is taken directly; if shallower, the deepest layer is repeated; if deeper, layers are sampled via uniform interpolation. Each sampled layer preserves not just the last token but also distributional statistics like sequence mean, norm, sparsity, kurtosis, and MAD.
    - **Design Motivation**: Hallucinations are not necessarily concentrated in the last token or layer. Dynamic sampling preserves the "shallow-to-deep" evolution while avoiding manual layer indexing for each model.

2. **Multi-scale Attention + Layer-weighted Transformer**:
    - **Function**: Simultaneously captures short-range local depth mutations and long-range inter-layer dependencies.
    - **Mechanism**: The sequence is projected into a unified hidden space, then local average pooling, linear projection, and upsampling are applied using multiple scale factors. Different scales are fused via position-dependent gating. The model then modulates each depth position with a learnable layer importance vector $\lambda$ before feeding it into a Pre-LN Transformer encoder.
    - **Design Motivation**: Hallucination signals may manifest as sudden semantic shifts in intermediate layers or as overall norm trajectory changes. Single mean pooling is too coarse; the multi-scale module observes both fine-grained and coarse-grained patterns.

3. **OOF Stacking and Ensemble Meta-learner**:
    - **Function**: Reduces overfitting of deep features to local training noise and combines inductive biases of different classifiers.
    - **Mechanism**: Fused embeddings for each training sample are generated by fold models not trained on it; test samples average embeddings from all fold models. Subsequently, classifiers like RandomForest, XGBoost, GradientBoosting, LightGBM, LogisticRegression, and SVM output probabilities, and a logistic meta-regressor learns final fusion weights.
    - **Design Motivation**: Hidden state statistics have high dimensionality and limited samples, and distributions vary across languages. Directly training a single classifier is prone to overfitting. The OOF mechanism mitigates leakage risk, while the ensemble meta-learner enhances cross-architecture robustness.

### Loss & Training

The deep model is trained using AdamW with a learning rate of $2 \times 10^{-4}$, weight decay of $6 \times 10^{-5}$, and a ReduceLROnPlateau scheduler for 45 epochs (early stopping patience 15). The framework combines BCE, focal, asymmetric, and contrastive objectives, incorporating label smoothing, Mixup, and CutMix. The hidden layers are fixed to $K=32$ samples, the sequence model hidden dimension is 384, with 8 attention heads and 6 Transformer encoder layers. Experiments employ 5-fold stratified cross-validation.

Multilingual evaluations do not use language-specific fine-tuning. The authors extended the English HaluEval / TriviaQA to French, Bengali, and Amharic using Gemini 1.5 Flash, with manual inspection of 100 samples per language (600 total). Initial translation accuracy was 96%, with the remaining 4% refined manually.

## Key Experimental Results

### Main Results

| Dataset | Base LLM | Best Baseline AUROC | MultiHaluDet AUROC | Key Conclusion |
|---------|----------|---------------------|--------------------|----------------|
| HaluEval | Mistral-7B | Neural CDEs 95.4 | 98.43 | Outperforms strongest continuous dynamics baseline by ~3.03 pts |
| HaluEval | LLaMA2-7B | Neural SDEs 92.8 | 98.55 | Maintains ~98.5 AUROC across architectures |
| TriviaQA | Mistral-7B | Neural SDEs 85.1 | 98.30 | Significant improvement on plausible hard negatives |
| TriviaQA | LLaMA2-7B | Neural CDEs 83.7 | 98.26 | More stable than hidden-state/probabilistic baselines |

### Cross-lingual Results

| Language Resource Level | Dataset | Mistral-7B AUROC | LLaMA2-7B AUROC | Observation |
|-------------------------|---------|------------------|-----------------|-------------|
| English | HaluEval | 98.4 | 98.5 | English benchmark is near saturation |
| French high-resource | HaluEval | 96.2 | 95.8 | Only slight decline compared to English |
| Bangla medium-resource | HaluEval | 89.1 | 88.4 | Morphology and corpus coverage cause more evident degradation |
| Amharic low-resource | HaluEval | 78.5 | 76.2 | Significantly higher than best baseline (62.3 / 59.8) |
| French high-resource | TriviaQA | 95.5 | 94.9 | Stable in hard negative scenarios |
| Bangla medium-resource | TriviaQA | 87.6 | 86.3 | Retains strong cross-lingual detection signals |
| Amharic low-resource | TriviaQA | 75.8 | 73.4 | Low-resource languages remain the primary challenge |

### Ablation Study

| Configuration | Mistral HaluEval | Mistral TriviaQA | LLaMA2 HaluEval | LLaMA2 TriviaQA | Description |
|---------------|------------------|------------------|-----------------|-----------------|-------------|
| Full | 98.43 | 98.30 | 98.55 | 98.26 | Full model |
| w/o MSA | 91.45 | 90.82 | 92.14 | 91.33 | Removing multi-scale attention drops ~6-8 pts |
| w/o OOF | 88.67 | 87.41 | 89.25 | 88.19 | Largest drop, OOF stacking is key for robustness |
| w/o TP | 93.28 | 92.56 | 93.71 | 93.04 | Using only static last layer loses ~5 pts |

### Key Findings

- Surface probabilistic features are nearly ineffective: $P(\text{True})$, AvgProb, and AvgEnt hover between 41.1%-49.7% AUROC, indicating that "low confidence equals hallucination" is an unreliable heuristic.
- OOF stacking is the most critical component; removing it drops AUROC by over 10 points on TriviaQA, suggesting plausible hard negatives are particularly prone to overfitting.
- Low-resource languages remain a bottleneck: AUROC on Amharic is significantly lower than French/Bangla, attributed to poor representation quality in base models.

## Highlights & Insights

- The most valuable perspective is shifting hallucination detection from "observing output confidence" to "observing hidden state evolution trajectories." This is closer to the model's factual judgment process and explains why trajectory probing ablation leads to performance drops.
- Dynamic layer sampling is a practical engineering design. It does not assume any absolute layer index is most important but uses relative depth to align different models for cross-architecture reuse.
- The combination of multi-scale attention and self-attention pooling is suitable for detection: the former captures local depth anomalies, while the latter allows the model to adaptively select important layers, avoiding signal loss from fixed pooling.
- Multilingual experiments, though based on translated data, clearly demonstrate bottlenecks in representation quality. This suggests future multilingual safety detection cannot report English results alone.

## Limitations & Future Work

- **Heavy White-box Dependency**: The method requires access to hidden states and logits, making it inapplicable to black-box commercial models like GPT-4 or Claude.
- **Higher Computational Cost than Heuristics**: While no language-specific fine-tuning is needed, full-layer state extraction, depth sequence modeling, and 5-fold OOF are more expensive than $P(\text{True})$ or logit entropy.
- **Translation-based Multilingual Evaluation**: Data for French, Bangla, and Amharic comes from translating English benchmarks. Even with manual QA, subtle local contexts or cultural knowledge may be missed.
- **Task Boundary Limited to QA**: Experiments focus on HaluEval and TriviaQA; effectiveness in long-form generation, tool calling, or multi-hop RAG remains unknown.
- Future work could compress full-depth trajectory probing into a few key layers or distill it into a lightweight detector to reduce deployment costs.

## Related Work & Insights

- **vs $P(\text{True})$ / AvgProb / AvgEnt**: These look at output confidence/entropy. They are low cost but perform near random in this study. MultiHaluDet captures high-confidence hallucinations by reading trajectories.
- **vs SAPLMA / MIND / Probe@Exact**: These hidden-state probes are stronger than surface probabilities but rely on single-point or static representations. MultiHaluDet explicitly models the full-depth sequence and reduces leakage via OOF.
- **vs Neural ODE / CDE / SDE hidden trajectory methods**: These also view continuous dynamics. Neural CDEs reach 95.4 AUROC on HaluEval; MultiHaluDet improves this to 98.43 via multi-scale attention, global fusion, and stacking.
- **Insight**: For safety detection, the "path" of internal representations may be more informative than the final state. This can be extended to jailbreak detection, factual consistency, and cross-lingual harmful content detection.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Solid combination of trajectory, multi-scale attention, and OOF stacking, though building on existing hidden-state probing trends.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Comprehensive main and ablation studies; the limitation is the reliance on translated multilingual data.
- Writing Quality: ⭐⭐⭐⭐☆ Clear breakdown of the method; the framework is complex with many components.
- Value: ⭐⭐⭐⭐⭐ Inspiring for multilingual LLM safety, proving that low-resource hallucination detection cannot rely solely on output probabilities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Rethinking Evaluation for LLM Hallucination Detection: A Desiderata, A New RAG-based Benchmark, New Insights](rethinking_evaluation_for_llm_hallucination_detection_a_desiderata_a_new_rag-bas.md)
- [\[ACL 2026\] Logical Consistency as a Bridge: Improving LLM Hallucination Detection via Label Constraint Modeling between Responses and Self-Judgments](logical_consistency_as_a_bridge_improving_llm_hallucination_detection_via_label_.md)
- [\[ACL 2026\] The Reasoning Trap: How Enhancing LLM Reasoning Amplifies Tool Hallucination](the_reasoning_trap_how_enhancing_llm_reasoning_amplifies_tool_hallucination.md)
- [\[ACL 2026\] Enhancing Hallucination Detection via Future Context](enhancing_hallucination_detection_via_future_context.md)
- [\[ICML 2026\] From Out-of-Distribution Detection to Hallucination Detection: A Geometric View](../../ICML2026/hallucination/from_out-of-distribution_detection_to_hallucination_detection_a_geometric_view.md)

</div>

<!-- RELATED:END -->
