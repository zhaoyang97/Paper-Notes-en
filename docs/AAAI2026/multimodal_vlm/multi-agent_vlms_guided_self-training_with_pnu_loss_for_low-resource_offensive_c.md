---
title: >-
  [Paper Note] Multi-Agent VLMs Guided Self-Training with PNU Loss for Low-Resource Offensive Content Detection
description: >-
  [AAAI 2026][Multimodal VLM][Self-training] This paper proposes a multi-agent vision-language model (MA-VLMs) guided self-training framework combined with a novel PNU loss function, achieving high-quality offensive content detection under low-resource settings (as few as 50 labeled samples), with performance approaching that of large-scale models.
tags:
  - "AAAI 2026"
  - "Multimodal VLM"
  - "Self-training"
  - "Multi-agent VLM"
  - "Pseudo-labels"
  - "PNU loss"
  - "Offensive content detection"
date: 2026-05-08
content_hash: 83b608b1a39018da
---

# Multi-Agent VLMs Guided Self-Training with PNU Loss for Low-Resource Offensive Content Detection

**Conference**: AAAI 2026
**arXiv**: [2511.13759](https://arxiv.org/abs/2511.13759)  
**Code**: [github](https://github.com/Social-AI-Studio/MA-VLM.git)  
**Area**: Multimodal VLM
**Keywords**: Self-training, Multi-agent VLM, Pseudo-labels, PNU loss, Offensive content detection

## TL;DR
This paper proposes a multi-agent vision-language model (MA-VLMs) guided self-training framework combined with a novel PNU loss function, achieving high-quality offensive content detection under low-resource settings (as few as 50 labeled samples), with performance approaching that of large-scale models.

## Background & Motivation

Offensive content on social media (hate speech, misogyny, harassment, etc.) poses threats to public safety and democratic discourse. Existing content moderation systems fall short in coverage, fairness, and cross-lingual/cross-cultural adaptability.

**Core Limitations**: Building robust offensive content detection systems requires large amounts of high-quality labeled data, which is extremely scarce. Two reasons account for this: (1) offensive samples are inherently rare and are often removed by platforms; (2) manual annotation is costly and requires understanding of context, sarcasm, and implicit harm.

**Limitations of Prior Work**:
- **LLM prompting**: Effective in few-shot settings but prohibitively expensive for large-scale deployment
- **Data augmentation**: Limited to text, with insufficient semantic diversity
- **Transfer learning**: Still requires moderate-scale target-domain annotations
- **Conventional self-training**: Poor pseudo-label quality when the initial model is weak, leading to severe error propagation

**Key Challenge**: In low-resource settings, a weak initial model yields unreliable pseudo-labels, causing self-training to fail; while VLMs offer strong understanding capabilities, their inference cost precludes direct deployment.

**Key Insight**: Can VLMs serve as "verifiers" rather than "executors" to guide the self-training of lightweight classifiers? Furthermore, since offensive content inherently involves annotation ambiguity (moderators tend toward conservatism while users tend toward freedom of expression), can simulating this social tension improve pseudo-label quality?

## Method

### Overall Architecture
The paper proposes an MA-VLMs-guided self-training pipeline (Figure 2). The core mechanism is to use a lightweight classifier (CLIP-Large + 1-layer MLP) for prediction and a frozen multi-agent VLM (Qwen-2.5-VL-72B) for verification. Consensus or disagreement between the classifier and the VLMs is used to categorize samples, which are then jointly trained with the PNU loss.

### Key Designs

#### 1. **MA-VLMs-Guided Self-Training Pipeline**
Self-training proceeds iteratively in five steps:
- **Training**: Train the classifier on $n$ (e.g., 100) labeled samples
- **Prediction and ranking**: The classifier predicts on unlabeled samples and ranks them by confidence
- **Consensus/disagreement determination**: Top-$k$ ($k=500$) high-confidence samples are submitted to MA-VLMs for verification. Unanimous agreement → **Agreed-Unknown** (assigned positive/negative pseudo-labels); disagreement → **Disagreed-Unknown** (retained as unlabeled)
- **Retraining**: Retrain the classifier on all data types using the PNU loss
- **Validation check**: Continue if development set performance improves; otherwise roll back

**Design Motivation**: Conventional self-training relies solely on the classifier's own judgment, making it prone to error accumulation when the initial model is weak. Introducing a VLM as an external verifier substantially improves pseudo-label quality. The Top-$k$ selection strategy ensures that only the most reliable samples are incorporated each round.

#### 2. **Multi-Agent VLM (MA-VLMs) Prompting Format**
Two distinct VLM roles are designed:
- **Moderator**: Safety-first bias, inclined to label content as offensive
- **User**: Advocates for freedom of expression, inclined to label content as non-offensive

Each agent provides an initial judgment with reasoning, then reviews the other agent's judgment before issuing a final decision. A sample is marked as Agreed-Unknown only when both agents agree with the classifier.

**Design Motivation**: Real-world content moderation is inherently characterized by tension between over-censorship and under-protection. Simulating this social dynamic better captures annotation ambiguity and, through multi-perspective negotiation, surfaces implicit hatred (e.g., memes that disguise denigration of women as compliments).

#### 3. **PNU Loss Function**
The core formulation is as follows:

$$\mathcal{L}_{\text{pnu}} = \begin{cases} (1-\gamma) \cdot (\mathcal{L}_{\text{pn}} + \mathcal{L}_{\text{soft-pn}}) + \gamma \cdot \mathcal{L}_{\text{pu}}, & \text{if } \gamma \geq 0 \\ (1+\gamma) \cdot (\mathcal{L}_{\text{pn}} + \mathcal{L}_{\text{soft-pn}}) - \gamma \cdot \mathcal{L}_{\text{nu}}, & \text{if } \gamma < 0 \end{cases}$$

Three data types correspond to three loss terms:
- **$\mathcal{L}_{\text{pn}}$**: Standard PN loss for ground-truth labeled data
- **$\mathcal{L}_{\text{soft-pn}}$**: Soft PN loss for Agreed-Unknown data (soft labels $\hat{y}_p=0.67, \hat{y}_n=0.33$), mitigating overfitting to pseudo-labels
- **$\mathcal{L}_{\text{pu}}$ / $\mathcal{L}_{\text{nu}}$**: PU/NU learning losses for Disagreed-Unknown data

The parameter $\gamma \in [-1,1]$ controls the strength of PU/NU learning: $\gamma=0$ reduces to PN learning, $\gamma>0$ applies PU learning, and $\gamma<0$ applies NU learning.

**Design Motivation**: Conventional methods discard disagreed samples, wasting valuable training signal. The PNU loss maximizes the utility of all available data by differentiating sample reliability while controlling the influence of pseudo-label noise.

### Loss & Training
- The classification loss $\ell$ uses cross-entropy
- The positive class prior $\pi_p$ is fixed at 0.5 (deviation introduces class bias)
- $\gamma$ is determined via ablation: $\gamma=0.0$ for FHM, $\gamma=0.1$ for other datasets
- Top-$k=500$ high-confidence samples are selected for verification per round
- Training runs for 10 epochs with the best epoch selected on the development set

## Key Experimental Results

### Main Results

| Model | Training Strategy | FHM M-F1 | MAMI M-F1 | HSOL M-F1 | Sent140 M-F1 |
|-------|-------------------|----------|-----------|-----------|--------------|
| Qwen7B | SupOnly | 70.41 | 76.06 | 84.89 | 78.19 |
| CLIP | SupOnly | 59.24 | 62.18 | 85.30 | 64.22 |
| CLIP | SelfTrain(CLIP) | 70.00 | 67.03 | 86.48 | 73.05 |
| CLIP | SelfTrain(Qwen72B) | 65.22 | 67.42 | 81.06 | 75.57 |
| **CLIP** | **SelfTrain(CLIP+Qwen72B)** | **72.68** | **73.49** | **86.69** | **77.11** |

($n=100$, i.e., only 100 labeled samples)

### Experiments under Varying Annotation Budgets (FHM Dataset)

| Annotation size $n$ | Qwen7B SupOnly M-F1 | CLIP SupOnly M-F1 | CLIP SelfTrain M-F1 |
|---------------------|---------------------|-------------------|---------------------|
| 50 | 39.11 | 48.76 | **71.27** |
| 100 | 70.41 | 59.24 | **72.68** |
| 250 | **75.88** | 69.67 | 72.97 |

### Ablation Study

| Configuration | FHM M-F1 | MAMI M-F1 | Note |
|---------------|----------|-----------|------|
| γ = -0.1 | 68.35 | 59.98 | NU learning ineffective |
| γ = 0.0 | **72.68** | 68.84 | Optimal for FHM |
| γ = 0.1 | 71.50 | **73.49** | Optimal for MAMI |
| γ = 0.2 | 71.79 | 72.42 | Excessive PU harmful |

| Prompting Format | FHM M-F1 | MAMI M-F1 | Note |
|------------------|----------|-----------|------|
| Zero-Shot | 74.46 | 79.17 | Baseline |
| Few-Shot | 71.09 | 75.08 | Examples introduce bias |
| CoT | 74.43 | 78.28 | CoT ineffective for social tasks |
| **MA-VLMs** | **74.62** | **81.64** | Multi-agent best |

### Key Findings
1. With only 50 labeled samples, self-trained CLIP (M-F1=71.27) substantially outperforms SupOnly Qwen7B (M-F1=39.11), demonstrating the framework's effectiveness under extreme low-resource conditions.
2. Joint pseudo-labeling with CLIP+Qwen72B outperforms single-model pseudo-labeling across all datasets, validating the value of complementary verification.
3. The MA-VLMs prompting format shows a more pronounced advantage on MAMI, where annotation ambiguity is greater (+2.47 vs. Zero-Shot), indicating that multi-agent negotiation is well-suited for handling ambiguous concepts.
4. The optimal value of $\gamma$ is dataset-dependent: $\gamma=0.1$ performs better when classes are balanced, while $\gamma=0.0$ is preferable under class imbalance.

## Highlights & Insights
1. **Elegantly designed multi-perspective moderation mechanism**: Simulating the social tension between moderators and users not only improves pseudo-label quality but also enhances the fairness and interpretability of the system.
2. **Unified framework via PNU loss**: Elegantly integrates three categories of data with varying reliability, avoiding the wasteful discarding of disagreed samples in conventional methods.
3. **Lightweight deployment**: Inference at test time requires only CLIP-Large (428M parameters); VLMs are used solely for pseudo-label generation during training, balancing performance and efficiency.
4. **Pseudo-label analysis reveals ground-truth annotation errors**: Suggesting that multi-party consensus may be more reliable than human annotation.

## Limitations & Future Work
1. Relies on Qwen72B for pseudo-label verification, requiring large-model inference resources during training.
2. The fixed Top-$k=500$ selection per round does not adaptively adjust the number of selected samples.
3. $\gamma$ requires manual tuning for each dataset, lacking an automatic selection mechanism.
4. Validation is limited to binary classification; fine-grained multi-class offensive content detection remains unexplored.
5. Adaptability to multilingual and multicultural settings is not considered.

## Related Work & Insights
- Applying PU learning to social tasks is a novel perspective, generalizable to other tasks with high annotation ambiguity (e.g., sentiment analysis, stance detection).
- The multi-agent negotiation mechanism can inspire other NLP tasks requiring the balancing of opposing perspectives.
- The knowledge distillation paradigm (large model → small model) warrants further exploration in low-resource settings.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Thinking as Society: Multi-Social-Agent Self-Distillation for Multimodal Misinformation Detection](../../ICLR2026/multimodal_vlm/thinking_as_society_multi-social-agent_self-distillation_for_multimodal_misinfor.md)
- [\[ICLR 2026\] PSP: Prompt-Guided Self-Training Sampling Policy for Active Prompt Learning](../../ICLR2026/multimodal_vlm/psp_prompt-guided_self-training_sampling_policy_for_active_prompt_learning.md)
- [\[ICLR 2026\] Vision-Zero: Scalable VLM Self-Evolution via Multi-Agent Self-Play](../../ICLR2026/multimodal_vlm/vision-zero_scalable_vlm_self-evolution_via_multi-agent_self-play.md)
- [\[CVPR 2026\] VS-Bench: Evaluating VLMs for Strategic Abilities in Multi-Agent Environments](../../CVPR2026/multimodal_vlm/vs_bench_evaluating_vlms_for_strategic_abilities_in_multi_agent_environments.md)
- [\[CVPR 2026\] Hierarchical Attacks for Multi-Modal Multi-Agent Reasoning](../../CVPR2026/multimodal_vlm/hierarchical_attacks_for_multi-modal_multi-agent_reasoning.md)

</div>

<!-- RELATED:END -->
