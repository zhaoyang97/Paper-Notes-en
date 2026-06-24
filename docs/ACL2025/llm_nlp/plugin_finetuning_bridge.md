---
title: >-
  [Paper Note] PiFi: Plug-in and Fine-tuning: Bridging the Gap between Small Language Models and Large Language Models
description: >-
  [ACL 2025][LLM (Other)][Knowledge Distillation] The PiFi framework is proposed, which inserts a single frozen layer of an LLM into an SLM and fine-tunes the combined model, significantly boosting SLM performance on NLU and NLG tasks with minimal computational overhead.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Knowledge Distillation"
  - "Small Language Models"
  - "Large Language Models"
  - "Model Compression"
  - "Transfer Learning"
date: 2026-05-08
content_hash: f42f90733f8ff3d3
---

# PiFi: Plug-in and Fine-tuning: Bridging the Gap between Small Language Models and Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2506.07424](https://arxiv.org/abs/2506.07424)  
**Code**: Not released  
**Area**: LLM/NLP  
**Keywords**: Knowledge Distillation, Small Language Models, Large Language Models, Model Compression, Transfer Learning  

## TL;DR

The PiFi framework is proposed, which inserts a single frozen layer of an LLM into an SLM and fine-tunes the combined model, significantly boosting SLM performance on NLU and NLG tasks with minimal computational overhead.

## Background & Motivation

**Core Problem:** LLMs possess powerful generalization abilities and language knowledge but suffer from high computational overhead, making them difficult to deploy in resource-constrained environments. Although SLMs are highly efficient, their generalization capability is limited. How can SLMs leverage the knowledge advantages of LLMs?

**Limitations of Prior Work:**
- Knowledge distillation methods (parametric or non-parametric) require the complete LLM as a teacher model or to generate synthetic data, yielding high computational costs.
- Directly fine-tuning LLMs demands substantial VRAM and computational resources.
- Existing SLM enhancement methods (such as domain pre-training and data augmentation) fail to introduce LLM-level external knowledge.

**Design Motivation:** Inspired by the computer vision domain where LLM layers are utilized as vision encoders, a single Transformer layer of an LLM already encodes rich linguistic knowledge. Through a "plug-in and freeze" approach, the LLM's knowledge can be effectively transferred to the SLM with almost no increase in trainable parameters.

## Method

### Overall Architecture

The core idea of the PiFi (Plug-in and Fine-tuning) framework is to extract a single Transformer layer from an LLM (e.g., Llama-3.1-8B), freeze it, insert it into a specific position of an SLM, and then fine-tune the combined model. The framework supports both encoder-only architectures (e.g., BERT) and encoder-decoder architectures (e.g., T5).

### Key Designs

1. **Dimension Adaptation Layers:** Since SLMs and LLMs have different hidden dimensions (e.g., BERT 768 vs. Llama 4096), linear projection layers $L_{in}$ (upsampling) and $L_{out}$ (downsampling) are introduced for dimension alignment. For encoder-only models: $h_{enc} = Enc(x)$, $h_{LLM} = L_{LLM}(L_{in}(h_{enc}))$, $\hat{y} = Head(L_{out}(h_{LLM}))$.

2. **LLM Layer Freezing Strategy:** During the fine-tuning phase, the parameters of $L_{LLM}$ are frozen. Only the original parameters of the SLM, $L_{in}$, $L_{out}$, and the classification head are trained, preventing catastrophic forgetting and minimizing extra trainable parameters.

3. **Flexible Architectural Adaptation:** For encoder-only models, the LLM layer is inserted between the encoder and the classification head; for encoder-decoder models, the LLM layer is inserted between the encoder and the decoder.

### Loss & Training

Standard task loss functions are used: cross-entropy loss for classification tasks and autoregressive language modeling loss for generation tasks, without requiring any additional distillation losses.

## Experiments

### Main Results: NLU Task Performance

| Model | SST2 | IMDB | Tweet(Sent) | Tweet(Off) | CoLA | MNLI | SNLI | SQuAD | Avg |
|------|------|------|------|------|------|------|------|------|------|
| BERT_base | 89.41 | 85.10 | 86.90 | 83.15 | 80.10 | 82.00 | 89.10 | 63.81 | 82.45 |
| +PiFi | **91.50** | **87.09** | **92.95** | **86.03** | **82.07** | **82.74** | **89.48** | **66.17** | **84.75** |
| ELECTRA_base | 93.42 | 88.31 | 90.58 | 83.52 | 83.99 | 85.41 | 90.11 | 44.44 | 82.00 |
| +PiFi | **94.13** | **89.40** | **93.31** | **84.99** | **86.26** | **86.47** | **90.48** | **67.99** | **86.71** |
| DeBERTa-V3_base | 93.74 | 89.45 | 91.29 | 83.60 | 84.75 | 87.52 | 90.94 | 69.40 | 86.34 |
| +PiFi | **95.01** | **89.83** | **93.80** | **85.60** | **86.07** | **87.98** | **91.05** | **69.87** | **87.40** |

**NLG Tasks**: T5_base + PiFi improves BLEU from 0.5301 to 0.5413 on Multi30K translation, and BART_base + PiFi improves BLEU from 0.2270 to 0.2331 on CNN/DailyMail summarization.

### Ablation Study

| Ablation Dimension | Key Findings |
|------|------|
| LLM Layer Position (Layer 1/16/32) | The last layer performs the best, as it contains the richest high-level linguistic knowledge |
| Whether to Freeze the LLM Layer | Freezing outperforms unfreezing, as unfreezing leads to catastrophic forgetting |
| Different LLM Sources | Llama-3.1-8B achieves the best overall performance, though all examined LLMs bring improvements |
| Instruction-tuned LLM | Using the instruction-tuned version yields similar results |

### Key Findings

1. PiFi consistently improves performance across all 5 SLM architectures, with BERT_base and ELECTRA_base achieving average gains of 2.3% and 4.7%, respectively.
2. In cross-domain generalization experiments, PiFi improves performance on IMDB $\rightarrow$ Tweet from 70.40 to 83.68 (+13.28%), demonstrating strong domain transfer capability.
3. Multilingual classification experiments demonstrate that using an LLM layer pre-trained on the target language can significantly enhance the multilingual capabilities of SLMs.
4. Only a tiny number of parameters (two linear layers) are added, resulting in negligible computational overhead.

## Highlights & Insights

- **Simple Yet Effective Design**: Simply inserting a single LLM layer with two linear mappings significantly boosts SLM performance.
- **Strong Versatility**: It is applicable to both encoder-only and encoder-decoder architectures, covering a wide range of NLU and NLG tasks.
- **Outstanding Cross-Domain and Multilingual Transfer**: Experiments show that PiFi not only improves in-domain performance but also enhances generalization to unseen domains and languages.
- **Highly Efficient Training**: The LLM layer is frozen during training, introducing minimal extra parameters.

## Limitations & Future Work

- Currently, only the effect of a single LLM layer has been verified; the potential of multi-layer combinations or middle layers has not been fully explored.
- Dimension projection in the linear mapping layer may cause information loss; a more advanced adaptation network (e.g., MLP) might yield better results.
- Whether significant improvements still hold for larger SLMs (e.g., 1-3B scale) has not been fully verified.
- During inference, loading one layer of parameters from the LLM (e.g., ~600M for one layer of Llama-3-8B) is required, increasing storage demands.
- Direct comparison with mainstream PEFT methods like LoRA is lacking.

## Related Work & Insights

- **Knowledge Distillation**: Parametric methods (Zhong et al.) train student models using the teacher's output distribution, while non-parametric methods (Ye et al.) generate synthetic data using LLMs.
- **SLM Enhancement**: Continual pre-training (Gururangan et al.), data augmentation (Gao et al.), etc.
- **Cross-Modal Applications of LLMs**: Utilizing LLM layers as vision encoders (Pang et al.), rewriting text descriptions via LLMs to enhance CLIP (Fan et al.).

## Rating

| Dimension | Score |
|------|------|
| Novelty | 7/10 |
| Effectiveness | 8/10 |
| Experimental Thoroughness | 8/10 |
| Writing Quality | 7/10 |
| Overall Score | 7.5/10 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] HFT: Half Fine-Tuning for Large Language Models](hft_half_fine-tuning_for_large_language_models.md)
- [\[ACL 2025\] Mixture of Small and Large Models for Chinese Spelling Check](mixture_of_small_and_large_models_for_chinese_spelling_check.md)
- [\[ACL 2025\] BehaviorBox: Automated Discovery of Fine-Grained Performance Differences Between Language Models](behaviorbox_automated_discovery_of_fine-grained_performance_differences_between_.md)
- [\[ACL 2025\] Efficient Ensemble for Fine-tuning Language Models on Multiple Datasets](efficient_ensemble_for_fine-tuning_language_models_on_multiple_datasets.md)
- [\[ACL 2025\] GradOT: Training-free Gradient-preserving Offsite-tuning for Large Language Models](gradot_offsite_tuning.md)

</div>

<!-- RELATED:END -->
