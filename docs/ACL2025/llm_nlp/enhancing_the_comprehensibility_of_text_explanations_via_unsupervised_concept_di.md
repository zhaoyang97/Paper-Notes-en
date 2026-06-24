---
title: >-
  [Paper Note] Enhancing the Comprehensibility of Text Explanations via Unsupervised Concept Discovery
description: >-
  [ACL 2025 (Findings)][LLM (Other)][Concept Bottleneck Models] The ECO-Concept framework is proposed to automatically extract textual concepts via a slot attention mechanism and evaluate concept comprehensibility using an LLM as a human proxy. A comprehensibility feedback loss guides model fine-tuning, achieving concept explanations with both high classification accuracy and human comprehensibility without any concept annotations.
tags:
  - "ACL 2025 (Findings)"
  - "LLM (Other)"
  - "Concept Bottleneck Models"
  - "Slot Attention"
  - "LLM as a Human Proxy"
  - "Self-Supervised Concept Discovery"
  - "Comprehensibility"
date: 2026-05-08
content_hash: 10b35285e7c9b162
---

# Enhancing the Comprehensibility of Text Explanations via Unsupervised Concept Discovery

**Conference**: ACL 2025 (Findings)  
**arXiv**: [2505.20293](https://arxiv.org/abs/2505.20293)  
**Code**: [Project Page](https://vicki-sun.github.io/projects/ECO-Concept)  
**Area**: Other  
**Keywords**: Concept Bottleneck Models, Slot Attention, LLM as a Human Proxy, Self-Supervised Concept Discovery, Comprehensibility

## TL;DR

The ECO-Concept framework is proposed to automatically extract textual concepts via a slot attention mechanism and evaluate concept comprehensibility using an LLM as a human proxy. A comprehensibility feedback loss guides model fine-tuning, achieving concept explanations with both high classification accuracy and human comprehensibility without any concept annotations.

## Background & Motivation

Concept-based explanation methods have received increasing attention because they map model decisions to attributes understandable by humans. However, existing methods suffer from significant limitations:

**Supervised methods rely on annotations**: Methods like CBM require extensive pre-defined concept annotations, which are expensive and unable to discover novel concepts. Even if LLMs are used to generate concept sets (such as TBM), the mapping process of the LLM itself remains a black box.

**Unsupervised methods lack comprehensibility**: Methods such as SelfExplain and PROTOTEX can automatically extract concepts, but the extracted concepts are often semantically vague, lack coherence, and even highlight irrelevant or misleading information.

**Comprehensibility is never integrated into the training loop**: Existing methods only measure concept comprehensibility through manual evaluation during the experimental phase, but never integrate this feedback into the training process to optimize concept quality.

**Key Insight**: If concept comprehensibility can be evaluated in real-time during the training process and used as a feedback signal, human comprehensibility of concepts can be significantly improved while maintaining classification performance.

## Method

### Overall Architecture

ECO-Concept consists of three modules:
- **Concept Extractor**: Automatic concept discovery based on slot attention.
- **Classifier**: Concept bottleneck layer + fully connected classification.
- **Concept Evaluator**: Evaluates concept comprehensibility using an LLM and generates a feedback loss.

Training consists of two phases: first training the base model (concept extraction + classification), and then fine-tuning it using the comprehensibility loss.

### Key Designs

1. **Concept Extractor based on Slot Attention**:

    - **Mechanism**: Migrating slot attention from visual object recognition to text concept discovery.
    - The input text is encoded to get $\bm{X} \in \mathbb{R}^{L \times D}$, and $M$ learnable concept prototypes $\bm{C} \in \mathbb{R}^{M \times D}$ serve as queries.
    - The attention matrix $\bm{A} \in [0,1)^{M \times L}$ is calculated via dot-product attention.
    - **Key Difference**: The softmax is normalized along the concept axis (M axis), introducing **sparse competition**—each token is primarily associated with only a few concepts.
    - Concept features $\bm{U} = \frac{\bm{A}}{\sum_l \bm{A}_{:,l}} (\bm{W}_v \bm{X})$.
    - **Design Motivation**: The competitive mechanism among slots naturally promotes concept diversity and concentration.

2. **Concept Regularization**:

    - **Consistency Loss** ($\mathcal{L}_{con}$): Concept features on different samples should be similar. The top-$k$ samples with the highest activation in a mini-batch are selected to calculate the distance between features.
    - **Distinctiveness Loss** ($\mathcal{L}_{dist}$): The average features of different concepts should be as distinct as possible, magnifying the distance between concepts.
    - These two regularizers mutually reinforce each other: consistency ensures each concept has clear semantics, and distinctiveness avoids concept redundancy.

3. **LLM-driven Comprehensibility Evaluation and Enhancement (Core Innovation)**:

    - **Concept Importance**: $\beta_m = \frac{1}{|\mathcal{B}|} \sum_{\bm{t} \in \mathcal{B}} \bm{t}_m \sum_{\omega=1}^{\Omega} |\bm{W}_{m,\omega}|$, considering both the activation value and classifier weights.
    - **Comprehensibility Measurement Process**:
        - (a) Select the highest activation samples from each concept to construct two sets: $\mathcal{D}_{sum}$ and $\mathcal{D}_{high}$.
        - (b) Feed the samples and attention values of $\mathcal{D}_{sum}$ to GPT-4o, asking it to summarize "what this concept focuses on".
        - (c) If the LLM determines that the concept is semantically meaningful, GPT-4o-mini is used to label concept-related tokens on new samples in $\mathcal{D}_{high}$ (0-1 annotation matrix $\bm{S}$).
        - (d) If the concept is not semantically meaningful, set $\bm{S}$ to all zeros to suppress this concept.
    - **Comprehensibility Loss**: $\mathcal{L}_{com} = \frac{1}{M} \sum_{m=1}^{M} \frac{\beta_m \sum \|\bm{A}_{m,:} - \bm{S}_{m,:}\|_2^2}{|\mathcal{B}_m \cap \mathcal{D}_{high}|}$
    - Iterative fine-tuning is performed until the semantics of all concepts stabilize.

### Loss & Training

- **Phase 1**: $\mathcal{L} = \mathcal{L}_{ce} + \lambda_{con}\mathcal{L}_{con} + \lambda_{dist}\mathcal{L}_{dist}$
- **Phase 2 (with Comprehensibility)**: $\mathcal{L} = \mathcal{L}_{ce} + \lambda_{con}\mathcal{L}_{con} + \lambda_{dist}\mathcal{L}_{dist} + \lambda_{com}\mathcal{L}_{com}$
- Hyperparameter settings: $\lambda_{con}=0.1$, $\lambda_{dist}=-0.01$, $\lambda_{com}=1$
- The comprehensibility enhancement phase typically converges within 3 epochs.
- The encoder uses RoBERTa, and the number of concepts is fixed to 20.

## Key Experimental Results

### Main Results: Classification Performance (Table)

| Method | CEBaB | Beer | Hotel | IMDB | AGnews | Twitter | SciCite |
|------|-------|------|-------|------|--------|---------|---------|
| RoBERTa (Black-box) | .682/.797 | .882/.882 | .981/.981 | .937/.937 | .941/.960 | .828/.812 | .858/.879 |
| CBM (Supervised) | .669/.802 | .883/.885 | .979/.979 | - | - | - | - |
| SelfExplain (Unsupervised) | .683/.799 | .873/.872 | .978/.979 | .936/.936 | .925/.949 | .817/.806 | .856/.873 |
| **ECO-Concept** | **.697/.808** | **.885/.885** | **.981/.981** | **.937/.937** | **.941/.961** | **.828/.813** | **.860/.881** |

> ECO-Concept achieves or exceeds black-box models on all datasets, while significantly outperforming unsupervised baselines.

### Concept Comprehensibility Evaluation (Table)

| Method | CEBaB (Sem/Dist/Con) | IMDB (Sem/Dist/Con) | AGnews (Sem/Dist/Con) |
|------|---------------------|---------------------|----------------------|
| Cockatiel | .50/.40/.47 | .35/.40/.41 | .65/.60/.41 |
| Concept-Shap | .25/.30/.42 | .40/.30/.32 | .35/.35/.35 |
| ProtoTEx | .45/.45/.35 | .25/.25/.36 | .40/.45/.41 |
| **ECO-Concept** | **.60/.60/.51** | **.65/.65/.52** | **.70/.65/.54** |

> ECO-Concept leads across all three dimensions: semantic coherence, distinctiveness, and consistency.

### Key Findings

1. **Reaches supervised method standards without concept labels**: It even outperforms CBM on CEBaB.
2. **Comprehensibility enhancement does not harm classification performance**: Ablation studies show that accuracy remains virtually unchanged after adding $\mathcal{L}_{com}$.
3. **Human forward simulation experiment**: Explanations from ECO-Concept helped users achieve the highest accuracy in predicting model outputs (Beer: 98.3%, AGnews: 86.7%), with the highest confidence scores.
4. **Intruder detection experiment**: ECO-Concept achieves the highest accuracy in detecting concept intruders across all tasks (up to 90%).
5. **Two regularizers reinforce each other**: Removing the consistency loss also reduces distinctiveness, and vice versa.

## Highlights & Insights

- **LLM-in-the-loop training paradigm**: Embedding an LLM as a human proxy within the training loop to optimize comprehensibility is a highly novel perspective. Unlike post-hoc evaluation, this is proactive guidance.
- **Successful migration of Slot Attention to NLP**: Proves that object-centric architectures from the visual domain are equally effective for text concept discovery.
- **Operationalized definition of comprehensibility**: "Comprehensibility" is operationalized as "whether the LLM can reconstruct the attention distribution based on the concept summary," which is both quantifiable and intuitively reasonable.
- **Explainability without performance loss**: Overturns the common assumption that "explainability must sacrifice accuracy."

## Limitations & Future Work

1. **Fixed number of concepts**: In the current framework, the number of concepts $M$ cannot be adaptively adjusted during training.
2. **Reliance on API LLM**: Evaluating concepts using GPT-4o/4o-mini is costly and constrained by API dependencies.
3. **Small base encoder**: Only validated on BERT/RoBERTa-scale models; not scaled to large language models like LLaMA.
4. **Subset sampling evaluation**: For cost reasons, comprehensibility evaluation is conducted only on a subset of samples.
5. **Generative tasks unexplored**: Only validated on classification tasks; applicability to generative/QA tasks remains unknown.

## Related Work & Insights

- **BotCL (Wang et al., 2023)** and **CCTs (Hong et al., 2024)** use slot attention for concept explainability in computer vision, which this work directly extends to the text domain.
- The automated neuron explanation method of **Bills et al. (2023)** inspired the design of the comprehensibility metrics in this paper.
- **CBM (Koh et al., 2020)** is the classic framework for Concept Bottleneck Models.
- **Insight**: The "comprehensibility evaluator" utilizing LLMs as human proxies can be extended to more explainability scenarios.

## Rating

| Dimension | Score (1-5) | Explanation |
|------|-----------|------|
| Novelty | 4.5 | LLM-in-the-loop comprehensibility optimization + slot attention transfer to text, which is highly novel. |
| Experimental Thoroughness | 4.5 | 7 datasets + 3 types of human evaluations + thorough ablation study. |
| Writing Quality | 4 | Clear structure with detailed methodology and analysis. |
| Value | 4 | Provides a practical and principled solution for unsupervised concept explainability. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ConSim: Measuring Concept-Based Explanations' Effectiveness with Automated Simulatability](consim_measuring_concept-based_explanations_effectiveness_with_automated_simulat.md)
- [\[ACL 2025\] Self-Foveate: Enhancing Diversity and Difficulty of Synthesized Instructions from Unsupervised Text via Multi-Level Foveation](self-foveate_enhancing_diversity_and_difficulty_of_synthesized_instructions_from.md)
- [\[ACL 2025\] Exploring Explanations Improves the Robustness of In-Context Learning](exploring_explanations_improves_the_robustness_of_in-context_learning.md)
- [\[ACL 2025\] Synergizing Unsupervised Episode Detection with LLMs for Large-Scale News Events](synergizing_unsupervised_episode_detection_with_llms_for_large-scale_news_events.md)
- [\[ACL 2025\] MExGen: Multi-Level Explanations for Generative Language Models](mexgen_multi_level_explanations.md)

</div>

<!-- RELATED:END -->
