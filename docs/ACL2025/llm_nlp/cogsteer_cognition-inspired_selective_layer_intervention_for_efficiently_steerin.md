---
title: >-
  [Paper Note] CogSteer: Cognition-Inspired Selective Layer Intervention for Efficiently Steering Large Language Models
description: >-
  [ACL 2025][LLM (Other)][Selective Layer Intervention] Leveraging eye-tracking data from cognitive science to analyze the behaviors of LLM layers, it is discovered that the middle layers exhibit the highest correlation with human gaze and are most suitable for semantic intervention. The CogSteer framework is proposed—fine-tuning only the single optimal layer (approximately 3% of parameters) achieves or exceeds the performance of full-layer fine-tuning…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Selective Layer Intervention"
  - "Eye-Tracking"
  - "Parameter-Efficient Fine-Tuning"
  - "Semantic Steering"
  - "Detoxification"
date: 2026-05-08
content_hash: 95edd0bb62acb859
---

# CogSteer: Cognition-Inspired Selective Layer Intervention for Efficiently Steering Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2410.17714](https://arxiv.org/abs/2410.17714)  
**Code**: [https://github.com/Ethanscuter/CogSteer](https://github.com/Ethanscuter/CogSteer)  
**Area**: LLM NLP  
**Keywords**: Selective Layer Intervention, Eye-Tracking, Parameter-Efficient Fine-Tuning, Semantic Steering, Detoxification

## TL;DR
Leveraging eye-tracking data from cognitive science to analyze the behaviors of LLM layers, it is discovered that the middle layers exhibit the highest correlation with human gaze and are most suitable for semantic intervention. The CogSteer framework is proposed—fine-tuning only the single optimal layer (approximately 3% of parameters) achieves or exceeds the performance of full-layer fine-tuning, demonstrating effectiveness in GLUE and detoxification tasks.

## Background & Motivation

**Background**: Parameter-Efficient Fine-Tuning (PEFT) methods, such as Adapters and LoRA, are the mainstream approach for adapting LLMs to downstream tasks, where the default practice is to insert trainable modules into **all layers** or the **last layer**.

**Limitations of Prior Work**: Full-layer fine-tuning exhibits parameter redundancy, where some layers contribute minimally or even negatively to specific tasks; fine-tuning only the last layer is not optimal either, as the last layer primarily handles prediction rather than semantic integration. There is a lack of a theoretically grounded layer selection strategy.

**Key Challenge**: Different layers of LLMs perform distinct functions (syntactic processing, semantic integration, reasoning/prediction), but existing PEFT methods make layer selection blindly—either selecting all layers or only the last layer.

**Goal**: (a) To reveal the functional division of labor among LLM layers in an interpretable manner; (b) Based on this, to identify the single layer most suitable for semantic intervention, enabling more efficient fine-tuning and inference-time steering.

**Key Insight**: Eye-tracking research in cognitive science indicates that metrics such as fixation duration and regression during human reading reflect language processing at different levels—from syntax to semantics to reasoning. The authors conduct a correlation analysis between these human cognitive metrics and the hidden states of various LLM layers.

**Core Idea**: Eye-tracking data is utilized to discover that middle layers serve as the core block for semantic processing. Conducting PEFT intervention exclusively on this layer achieves full-layer performance with only 1/N of the parameters.

## Method

### Overall Architecture
CogSteer consists of three steps: (1) Cognition-inspired interpretability analysis—calculating the Pearson correlation between the hidden states of LLM layers and human eye-tracking metrics, finding that the middle layers exhibit the highest correlation; (2) Heuristic optimal layer selection—searching for the best intervention layer $M'$ within the middle bucket (layers from N/3 to 2N/3) using a validation set; (3) Selective layer intervention—inserting Adapter/LoRA for fine-tuning exclusively at the optimal layer, or performing implicit layer contrastive intervention during inference to achieve detoxification.

### Key Designs

1. **Eye-Tracking and Hidden State Correlation Analysis**:

    - **Function**: Measures the depth of human cognitive processing for each word using 5 eye-tracking metrics (SFD: single fixation duration, FFD: first fixation duration, GD: gaze duration, TRT: total reading time, GPT: go-past time), and computes their Pearson correlation with the FFN hidden states of each LLM layer.
    - **Mechanism**: For each layer $l$, the hidden states of all words $\mathbf{h}_{l,i}$ are reduced to a scalar via PCA, which is then used to calculate the correlation $\rho_{l,k}$ with the corresponding eye-tracking metric $e_i^{(k)}$.
    - **Design Motivation**: Eye-tracking metrics represent interpretable measures of human cognitive processing, offering a more generalizable and scalable approach than probing or circuit discovery methods.

2. **Three-Stage Layer Behavior Discovery**:

    - Divide the N layers equally into three buckets: premature (initial), middle (intermediate), and mature (advanced).
    - Finding 1: The middle layers of all models (GPT-2, Llama2-7B) show the highest correlation with eye-tracking metrics, indicating that the middle layers are responsible for deep semantic integration.
    - Finding 2: Task-oriented reading (relation detection) shows higher correlations in the middle and mature layers than natural reading, suggesting these layers are involved in reasoning; Llama2 also exhibits reasoning capabilities in its middle layers due to RLHF training.

3. **Heuristic Optimal Layer Selection**:

    - **Function**: Selects the training/validation layer $M'$ that performs best on the validation set from the candidate layers in the middle bucket.
    - **Formula**: $M' = \arg\max_{l \in J} Score(D; P(\cdot | x_t, l))$, where $J$ is the set of layers in the middle bucket.
    - **Design Motivation**: Based on functional layer priors from cognitive analysis, the search space is narrowed from N layers to approximately N/3 layers, significantly reducing search costs.

4. **Implicit Layer Contrastive Intervention (Inference-Time Steering / Detoxification)**:

    - **Function**: Steers the generation direction during inference without introducing extra parameters, by contrasting the value vectors of a toxic model and the original model at the optimal layer.
    - **Mechanism**: The semantic steering direction is calculated as $\Delta v^M = v_c^M - v_o^M$; the value vector of the original model is then updated as $v'^M = v_o^M - \lambda_{norm}^\alpha \cdot \Delta v^M$, and finally normalized to maintain the vector norm.
    - **Difference from Contrastive Decoding**: Instead of contrasting at the final output layer, an implicit intervention is executed on the attention value vectors of the middle layers, which is more fine-grained.

### Loss & Training
- GLUE Tasks: Insert Adapters (vanilla Adapter for GPT-2, LLaMa-Adapter for Llama2) at the optimal layer, training only the Adapter parameters of that specific layer.
- Toxicity Control: First fine-tune the model on the Jigsaw toxicity dataset to obtain a toxic reference model, and apply layer contrastive intervention during inference for detoxification.

## Key Experimental Results

### Main Results (GLUE Benchmark)

| Model | Intervention Method | MNLI-M | RTE | SST-2 | Avg. (Test) | Parameters |
|---|---|---|---|---|---|---|
| GPT-2 Large | Full-layer FT | 82.6 | 62.6 | 93.5 | 77.1 | 14.8M (100%) |
| GPT-2 Large | Single layer L19 | 79.3 | 64.6 | 92.4 | 75.8 | 0.4M (2.7%) |
| Llama2-7B | Full-layer FT | 89.5 | 58.2 | 93.5 | 78.7 | 1.3M (100%) |
| Llama2-7B | Single layer L14 | 82.9 | 74.7 | 95.2 | **80.5** | 0.04M (3.1%) |
| Mistral-7B | Full-layer FT | 89.7 | 52.3 | 97.3 | 77.3 | 134.5M (100%) |
| Mistral-7B | Single layer L12 | 87.1 | 81.0 | 95.9 | **83.2** | 4.2M (3.1%) |

- Llama2 single-layer intervention outperforms full-layer fine-tuning by **+1.8** in average score, and Mistral outperforms it by **+5.9**, utilizing only 3.1% of the parameters.

### Ablation Study (Toxicity Control)

| Configuration | GPT-2 Toxicity↑ | GPT-2 Detoxification↓ | Llama2 Toxicity↑ | Llama2 Detoxification↓ |
|---|---|---|---|---|
| Full-layer FT | 0.86 | 0.60 | 0.86 | 0.62 |
| Optimal Single Layer | **0.87** (+1.2%) | **0.59** | **0.87** (+1.2%) | **0.59** |
| Last Layer | 0.83 | 0.63 | 0.71 | 0.73 |

### Key Findings
- **Optimal intervention layers consistently reside in the middle bucket**: L19 for GPT-2 (out of 36 layers), L14 for Llama2 (out of 32 layers), and L12 for Mistral (out of 32 layers), aligning perfectly with findings from cognitive analysis.
- **Highly efficient**: On average, requiring only half the training time of full-layer fine-tuning and 3% of the parameters.
- **Significant detoxification**: Llama2 single-layer detoxification outperforms last-layer intervention by +24%, demonstrating that semantic steering in the middle layers is far more effective than in the output layer.
- **Stunning improvement on RTE**: Mistral's single-layer L12 outperforms full-layer fine-tuning by +28.7%, indicating that full-layer fine-tuning can be detrimental to certain reasoning tasks.

## Highlights & Insights
- **Elegant Bridge Between Cognitive Science and AI**: Utilizing eye-tracking data to understand LLM layer behavior offers a novel perspective. Eye-tracking metrics are human processing measures validated by decades of cognitive science; using them to steer model intervention is both theoretically grounded and practically valuable.
- **Intuitive Validation of "Less is More"**: Intuitively, fine-tuning all layers is less efficient than fine-tuning only the most critical layer. This paper validates this intuition through cognitive evidence and extensive experiments. This approach can be directly transferred to any PEFT scenario.
- **Ingenuity of Implicit Layer Contrastive Intervention**: Instead of contrasting at the output layer (like DExps and other methods), the contrast and normalization are conducted on the value vectors of the middle layers, resulting in a more fine-grained intervention without compromising generation fluency.

## Limitations & Future Work
- **Only FFN Hidden States Analyzed**: The relationship between attention blocks and eye movements remains unexplored, which might miss important information.
- **Oversimplified Tri-partition Bucket Division**: Functional boundaries of layers in different models can be uneven; adaptive layer clustering might be a superior alternative.
- **Language Limitations of Eye-Tracking Data**: The utilized eye-tracking datasets are primarily in English, leaving cross-lingual generalizability unverified.
- **Experiments Based on Medium-Scale Models**: Experiments were conducted on GPT-2, Llama2-7B, and Mistral-7B; layer behaviors of larger models (70B+) may differ.
- **Future Directions**: Extending eye-tracking analysis to attention modules; exploring dynamic multi-layer intervention; and combining eye-tracking data from diverse cognitive tasks for fine-grained layer function analysis.

## Related Work & Insights
- **vs. LoRA/Adapter Full-layer Fine-tuning**: Traditional PEFT does not select specific layers; CogSteer selects layers using cognitive analysis, yielding better performance with fewer parameters.
- **vs. Contrastive Decoding (Li et al., 2023)**: Contrastive Decoding performs logit contrast at the output layer, while CogSteer performs contrast on middle-layer value vectors, providing a more reasonable intervention locus.
- **vs. DExperts (Liu et al., 2021)**: DExperts requires additional expert/anti-expert models, whereas CogSteer only employs a single contrastive model to intervene at a single layer.
- The cognitive analysis methodology introduced in this paper can be transferred to scenarios such as layer selection in VLMs and expert routing in MoEs.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The interdisciplinary perspective of eye-tracking paired with LLM layer analysis is highly novel, though the core idea of selective layer fine-tuning is not a massive breakthrough.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Highly comprehensive, covering 3 models × multiple benchmarks (GLUE+toxicity control) + various PEFT methods.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with intuitive visualizations in the cognitive analysis section.
- **Value**: ⭐⭐⭐⭐ High practical value; the middle-layer selection strategy is easily applicable for resource-saving by any researcher or practitioner using PEFT.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Multi-Attribute Steering of Language Models via Targeted Intervention](multi_attribute_steering.md)
- [\[ACL 2025\] SConU: Selective Conformal Uncertainty in Large Language Models](sconu_selective_conformal_uncertainty_in_large_language_models.md)
- [\[ACL 2025\] Steering off Course: Reliability Challenges in Steering Language Models](steering_off_course_reliability_challenges_in_steering_language_models.md)
- [\[ACL 2025\] CogniBench: A Legal-inspired Framework and Dataset for Assessing Cognitive Faithfulness of Large Language Models](cognibench_cognitive_faithfulness.md)
- [\[ACL 2025\] From Data to Knowledge: Evaluating How Efficiently Language Models Learn Facts](from_data_to_knowledge_evaluating_how_efficiently_language_models_learn_facts.md)

</div>

<!-- RELATED:END -->
