---
title: >-
  [Paper Note] TableDART: Dynamic Adaptive Multi-Modal Routing for Table Understanding
description: >-
  [ICLR 2026][Multimodal VLM][Table Understanding] This paper proposes TableDART, which employs a lightweight MLP gating network with only 2.59M parameters to dynamically select the optimal processing path (Text-only / Ima…
tags:
  - "ICLR 2026"
  - "Multimodal VLM"
  - "Table Understanding"
  - "Dynamic Routing"
  - "Multi-modal Fusion"
  - "Gating Network"
  - "LLM Agent"
date: 2026-05-08
content_hash: 65bd63956521fbfc
---

# TableDART: Dynamic Adaptive Multi-Modal Routing for Table Understanding

**Conference**: ICLR 2026
**arXiv**: [2509.14671](https://arxiv.org/abs/2509.14671)  
**Code**: [GitHub](https://github.com/xiaobo-xing/TableDART)  
**Area**: Multimodal VLM
**Keywords**: Table Understanding, Dynamic Routing, Multi-modal Fusion, Gating Network, LLM Agent

## TL;DR

This paper proposes TableDART, which employs a lightweight MLP gating network with only 2.59M parameters to dynamically select the optimal processing path (Text-only / Image-only / Fusion) for each query-table pair. By reusing frozen unimodal expert models and introducing an LLM Agent for cross-modal fusion, TableDART achieves an average improvement of 4.02% over the strongest MLLM baseline HIPPO across 7 table understanding benchmarks, while reducing inference latency by 24.5%.

## Background & Motivation

**Background**: Table understanding is a core task bridging structured data and natural language. Existing approaches fall into three paradigms: (1) Table-as-Text — linearizing tables into text sequences for LLM processing, which is effective but loses spatial structure and is sensitive to serialization format; (2) Table-as-Image — rendering tables as screenshots for VLM processing, which preserves structure but has limited semantic capture capability; (3) Table-as-Multimodality — fusing both textual and visual views, as in HIPPO, which jointly processes both representations within an MLLM.

**Limitations of Prior Work**: Despite promising results, multimodal methods suffer from two critical limitations: (1) **Static fusion introduces redundancy and conflict** — forcing dual-modal processing for all query-table pairs is unnecessary, since text linearization introduces row-order sensitivity while image representation maintains permutation invariance, and conflicting signals from both modalities can mislead the model; (2) **MLLM fine-tuning is prohibitively expensive** — even with parameter-efficient strategies such as LoRA, HIPPO requires 25.87M trainable parameters, approximately 10× that of TableDART.

**Key Challenge**: The benefit of multimodal fusion stems from information complementarity, but its cost lies in redundancy and potential conflict. In 58.7% of test samples, both unimodal paths yield correct answers (i.e., "easy samples"), making forced fusion not only computationally wasteful but also potentially noise-inducing.

**Key Insight**: Since the optimal processing strategy varies across query-table pairs, the system should automatically learn "when to use text, when to use image, and when fusion is needed." An extremely lightweight routing network can make instance-level decisions by fully reusing existing unimodal experts.

**Core Idea**: Replace expensive MLLM fine-tuning with a 2.59M-parameter MLP gating network that dynamically selects the Text-only / Image-only / Fusion path for each query-table pair.

## Method

### Overall Architecture

TableDART consists of five collaborative components: (1) a Table-as-Text model $\mathcal{M}_t$ (TableGPT2-7B, frozen); (2) a Table-as-Image model $\mathcal{M}_v$ (Ovis2-8B, frozen); (3) a query text embedding model; (4) a lightweight MLP gating network (the only trainable component, 2.59M parameters); and (5) an LLM Agent (Gemini 2.0 Flash, used in the Fusion path, training-free). Given a query and table, three encoders extract in parallel the text representation $\mathbf{e}_t$, image representation $\mathbf{e}_v$, and query embedding $\mathbf{e}_q$, which are concatenated as $\mathbf{x} = [\mathbf{e}_q, \mathbf{e}_t, \mathbf{e}_v]$ and fed into the gating network. The path with the highest output logit is then selected for inference.

### Key Designs

1. **Multimodal Encoding and Feature Concatenation**

    - **Function**: Unifies multimodal information from the query and table into input representations for the gating network.
    - **Mechanism**: The table is separately serialized into text (encoded by $\mathcal{M}_t$'s encoder $\mathcal{E}_t$) and rendered as a screenshot (encoded by $\mathcal{M}_v$'s encoder $\mathcal{E}_v$), while the query is encoded by an independent text embedding model $\mathcal{E}_q$. The three feature streams are pooled in a modality-specific manner and concatenated as $\mathbf{x} = [\mathbf{e}_q, \mathbf{e}_t, \mathbf{e}_v]$. Note that $\mathcal{E}_t$ and $\mathcal{E}_v$ activate only a small fraction of the expert models' parameters (7.15% and 7.63%, respectively), incurring minimal computational overhead.
    - **Design Motivation**: The gating network requires access to all modalities to make optimal routing decisions, but only needs feature-level representations rather than full inference outputs; thus, only the early encoder layers are used.

2. **Gating Network and Policy Training**

    - **Function**: Dynamically selects the optimal inference path for each query-table pair.
    - **Mechanism**: The gating network $\mathcal{G}$ is a lightweight MLP that outputs three-way logits $\mathbf{z} = \mathcal{G}(\mathbf{x})$. The training objective $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda \mathcal{L}_{\text{resource}}$ consists of two terms: a task loss that minimizes via KL divergence the predicted distribution against an empirical correctness distribution (a binary vector $\mathbf{s}$ pre-computed for each path, converted to soft targets via temperature-scaled softmax); and a resource regularization term $\mathcal{L}_{\text{resource}} = \text{softmax}(\mathbf{z}/\tau_g)^T \mathbf{c}$ that penalizes high-cost paths (where $\mathbf{c}$ is an empirically measured inference cost vector), preventing over-reliance on the Fusion path.
    - **Design Motivation**: Pure task optimization would route most samples to Fusion (the safest but most expensive option); resource regularization encourages easy samples to be routed to more efficient unimodal paths. Setting $\lambda = 0.15$ achieves the best balance between performance and efficiency.

3. **LLM Agent Fusion Inference**

    - **Function**: Integrates the outputs of two unimodal experts when the gating network selects the Fusion path.
    - **Mechanism**: The two expert models $\mathcal{M}_t$ and $\mathcal{M}_v$ are executed in parallel to obtain their respective results $r_t, r_v$ and auxiliary outputs $a_t, a_v$, which are then submitted to the Fusion Agent (Gemini 2.0 Flash) together with the original table. The Agent operates in two roles: (a) **Arbitrator** — when the two experts' results conflict, it selects the more reliable answer based on confidence; (b) **Rescuer** — when both experts are uncertain, it synthesizes partial evidence from both to reason toward a new answer.
    - **Design Motivation**: Directly training an MLLM for fusion is expensive; using a training-free LLM Agent for post-hoc reasoning leverages strong reasoning capabilities without training costs. Experiments show that the Fusion path successfully rescues 14% of "hard samples" where both unimodal paths fail.

### Loss & Training

The training set comprises 10K mixed samples drawn from 5 table understanding benchmarks. Only the gating network is trained; all large models are frozen. For each training sample, the correctness of all three paths $\mathbf{s} \in \{0,1\}^3$ is pre-computed, and temperature $\tau$ controls the smoothness of the soft label distribution. During inference, the path with the highest logit is selected deterministically.

## Key Experimental Results

### Main Results

| Method | WTQ | TABMWP | TAT-QA | HiTab | FeTaQA | TabFact | InfoTabs | Avg. Acc |
|--------|-----|--------|--------|-------|--------|---------|---------|----------|
| TableGPT2-7B (Text) | 61.42 | 83.87 | 50.39 | 70.27 | 28.97 | 77.80 | 71.07 | 69.14 |
| Ovis2-8B (Image) | 58.76 | 87.00 | 47.67 | 68.59 | 34.70 | 80.80 | 74.11 | 69.49 |
| HIPPO-8B (Multimodal) | 55.77 | 87.50 | 60.75 | 63.00 | 33.18 | 82.27 | 75.74 | 70.84 |
| Gemini 2.0 Flash | 63.56 | 46.29 | 35.62 | 60.41 | 10.57 | 81.33 | 54.31 | 56.92 |
| **TableDART** | **70.58** | **84.54** | **62.05** | **74.37** | **36.11** | **81.37** | **76.22** | **74.86** |

TableDART achieves an average accuracy of 74.86%, surpassing the strongest multimodal baseline HIPPO-8B by **+4.02%**. Generalization on unseen datasets is particularly notable: TableDART 74.37% vs. HIPPO 63.00% (**+18.05%**).

### Ablation Study

| Routing Strategy | WTQ | TABMWP | TAT-QA | HiTab | TabFact | InfoTabs | Note |
|-----------------|-----|--------|--------|-------|---------|---------|------|
| Random Routing | 65.40 | 75.50 | 58.94 | 70.49 | 79.50 | 69.57 | No effective routing |
| Non-adaptive Fusion | 70.97 | 81.47 | 63.34 | 73.35 | 81.56 | 76.83 | All samples routed to Fusion |
| **Dynamic Routing** | **70.58** | **84.54** | **62.05** | **74.37** | **81.37** | **76.22** | Ours |

Dynamic routing outperforms non-adaptive fusion on TABMWP (+3.07) and HiTab (+1.02), demonstrating that forced fusion introduces noise on simpler datasets. In terms of inference efficiency, dynamic routing achieves an average latency of 2.20s vs. 2.92s for non-adaptive fusion, a **24.5% reduction**.

### Key Findings

- **58.7% of samples are "easy samples"**: Both unimodal paths answer correctly, making forced fusion entirely unnecessary.
- **24.0% of samples exhibit cross-modal complementarity**: 17.2% are answered correctly only by the image path and 6.8% only by the text path, validating the necessity of maintaining independent unimodal paths.
- **The Fusion path achieves a 14% "rescue" rate**: Among the 17.3% of hard samples where both unimodal paths fail, the Fusion Agent successfully resolves an additional 2.4%.
- **Routing strategies are interpretable**: 97.2% of samples in simple datasets like TABMWP are routed to Image-only, while 88.7% of hard samples in TAT-QA are routed to Fusion.

## Highlights & Insights

- **Extreme training efficiency**: Training only 2.59M parameters surpasses HIPPO trained with 25.87M parameters. The core insight is that "routing decisions matter more than modal fusion." This "meta-decision + frozen experts" paradigm is transferable to any multi-expert system.
- **Generalizability of the routing strategy**: Performance is nearly consistent across seen and unseen datasets (74.95% vs. 74.37%), whereas HIPPO drops from 72.41% to 63.00%, indicating that the gating network learns a generalizable routing policy rather than overfitting.
- **Elegant design of training signals**: Using "three independently pre-computed correctness labels" as supervision allows multiple paths to be simultaneously correct, and combined with KL divergence soft-label training, this is more principled than hard-label classification.

## Limitations & Future Work

- **Dependency on external Gemini as the Fusion Agent**: The Fusion path requires calls to a closed-source API, increasing cost and privacy concerns; open-source LLMs could be explored as alternatives.
- **Pre-computation of three-path results for training data**: Running inference three times per training sample is non-trivial, limiting scalability of the training set.
- **Gating network relies solely on feature-level information**: The current routing decision is based on shallow encoder features and does not exploit higher-level information such as the semantic complexity of the query.
- **Only three fixed paths are supported**: More flexible routing strategies, such as partial fusion or cascaded inference, have not been explored.

## Related Work & Insights

- **vs. HIPPO**: HIPPO statically fuses text and image representations inside an MLLM, whereas TableDART dynamically routes externally, achieving both better performance and greater training efficiency (2.59M vs. 25.87M parameters).
- **vs. Mixture-of-Experts**: TableDART's design resembles MoE but routes at the model level rather than the layer level; the experts are complete frozen models rather than trainable sub-networks.
- **vs. Table-LLaVA/TabPedia**: These methods train on a single visual modality and cannot leverage the complementary advantages of the text modality.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of instance-level dynamic routing and training-free LLM Agent fusion is novel, though the basic idea of dynamic routing is not new.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluated across 7 benchmarks with comprehensive ablations, routing strategy analysis, efficiency analysis, and generalization verification.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, well-motivated arguments, and rich figures and tables.
- **Value**: ⭐⭐⭐⭐ Provides a training-efficient paradigm for multimodal fusion with reference value for both table understanding and broader multi-expert systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Multi-modal Data Spectrum: Multi-modal Datasets are Multi-dimensional](multi-modal_data_spectrum_multi-modal_datasets_are_multi-dimensional.md)
- [\[CVPR 2026\] AVR: Adaptive VLM Routing for Computer Use Agents](../../CVPR2026/multimodal_vlm/adaptive_vision-language_model_routing_for_computer_use_agents.md)
- [\[AAAI 2026\] TabFlash: Efficient Table Understanding with Progressive Question Conditioning and Token Focusing](../../AAAI2026/multimodal_vlm/tabflash_efficient_table_understanding_with_progressive_question_conditioning_an.md)
- [\[ICLR 2026\] Enhancing Multi-Image Understanding through Delimiter Token Scaling](enhancing_multi-image_understanding_through_delimiter_token_scaling.md)
- [\[ICLR 2026\] Contamination Detection for VLMs using Multi-Modal Semantic Perturbation](contamination_detection_for_vlms_using_multi-modal_semantic_perturbation.md)

</div>

<!-- RELATED:END -->
