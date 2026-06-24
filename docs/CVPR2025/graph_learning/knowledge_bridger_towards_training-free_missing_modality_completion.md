---
title: >-
  [Paper Note] Knowledge Bridger: Towards Training-Free Missing Modality Completion
description: >-
  [CVPR 2025][Graph Learning][Missing Modality Completion] This paper proposes Knowledge Bridger, a training-free framework for missing modality completion. By leveraging Large Multimodal Models (LMMs) to automatically mine multimodal knowledge and construct a knowledge graph, it guides the generation and ranking of missing modalities, surpassing existing methods in both general and medical OOD scenarios.
tags:
  - "CVPR 2025"
  - "Graph Learning"
  - "Missing Modality Completion"
  - "Knowledge Graph"
  - "Training-Free"
  - "Large Multimodal Models"
  - "OOD Generalization"
date: 2026-05-08
content_hash: 226b3ad4e7b1372a
---

# Knowledge Bridger: Towards Training-Free Missing Modality Completion

**Conference**: CVPR 2025  
**arXiv**: [2502.19834](https://arxiv.org/abs/2502.19834)  
**Code**: [https://github.com/Guanzhou-Ke/Knowledge-Bridger](https://github.com/Guanzhou-Ke/Knowledge-Bridger)  
**Area**: Medical Images / Multimodal Learning  
**Keywords**: Missing Modality Completion, Knowledge Graph, Training-Free, Large Multimodal Models, OOD Generalization

## TL;DR
This paper proposes Knowledge Bridger, a training-free framework for missing modality completion. By leveraging Large Multimodal Models (LMMs) to automatically mine multimodal knowledge and construct a knowledge graph, it guides the generation and ranking of missing modalities, surpassing existing methods in both general and medical OOD scenarios.

## Background & Motivation

1. **Background**: Missing Modality Completion (MMC) is a critical problem in multimodal learning. Existing successful methods rely on carefully designed fusion techniques and pre-training on a large amount of complete data, such as MMIN utilizing inter-modality interpolation for recovery, and DiCMoR utilizing cross-modality decoupling for completion.
2. **Limitations of Prior Work**: These methods exhibit limited generalization ability in out-of-distribution (OOD) scenarios, requiring retraining when switching to new domains such as medical imaging or autonomous driving, which incurs significant human effort and computational cost. Although schemes like prompt-learning and missing modality tags reduce adaptation costs, they still depend on a large amount of training data.
3. **Key Challenge**: How to simultaneously achieve low resource dependency (without requiring training data for new domains and massive computation) and strong OOD generalization capability?
4. **Goal**: Two sub-problems: (1) How to constrain the generation process to ensure the fidelity of the missing modality; (2) How to select the best result from multiple generated candidates.
5. **Key Insight**: Large Multimodal Models (e.g., Qwen2-VL) possess powerful OOD capabilities and in-context learning capabilities, enabling them to understand and generate multimodal content without fine-tuning.
6. **Core Idea**: Leverage LMMs to automatically construct a knowledge graph from available modalities, guide missing modality generation in a knowledge-driven manner, and select the best completion through a combination of graph similarity and semantic similarity ranking.

## Method

### Overall Architecture
Knowledge Bridger is a three-stage training-free pipeline: the input consists of available multimodal data and domain prior knowledge, and the output is high-quality missing modality data. The entire process requires no training and fully leverages the capabilities of pre-trained LMMs. The three stages are: Knowledge Graph Modeling -> Knowledge-driven Generation -> Knowledge-based Ranking.

### Key Designs

1. **Knowledge Graph Modeling**:

    - **Function**: Automatically extract structured knowledge from available modalities to construct a knowledge graph for guiding subsequent generation and ranking.
    - **Mechanism**: Leverage the Chain-of-Thought (CoT) reasoning capability of the LMM. By defining extraction rules in the form of {Entity: Reasoning Prompts} (e.g., "identify the main objects in the image"), the LMM is prompted to step-by-step analyze entities, relations, and attributes in the available modalities. CoT consists of two steps: first letting the LMM generate concise answers for each rule, and then extracting unique entity-relation pairs from those answers. For specialized fields like medical domains, domain prior knowledge (such as histology or clinical diagnostic information) can be injected to reduce hallucination.
    - **Design Motivation**: Prompting the LMM to extract all entity relations at once is prone to omissions due to context window limitations. The step-by-step CoT strategy improves the accuracy of answers for individual rules while facilitating better integration of information.

2. **Knowledge-driven Generation**:

    - **Function**: Utilize structured information in the knowledge graph to guide the LMM to precisely generate candidate content for the missing modality.
    - **Mechanism**: Propose a knowledge-driven entity alternation strategy—selecting entities related to the missing modality from the knowledge graph and using a multi-perspective generation method with each entity as the subject, prompting the LMM to generate standardized text descriptions containing all node and attribute information. These descriptions are then passed to a modality generator (SDXL 1.0 when images are missing, Cheff model for medical images, and directly generated by the LMM when text is missing). By default, 5 candidates are generated.
    - **Design Motivation**: Directly prompting the LMM to describe an image introduces substantial randomness—neither knowing the format of the missing text (title/abstract/description) nor precisely specifying the focus of the description. Multi-perspective generation guided by the knowledge graph reduces randomness and enhances controllability.

3. **Knowledge-based Ranking**:

    - **Function**: Automatically evaluate the quality of the generated missing modality and select the best candidate.
    - **Mechanism**: Compute a quality score $QS(x_a, x_m) = \cos_{graph}(f_a(x_a), f_a(x_m)) + [\cos(f_c(x_a), f_c(x_m)) + \cos(f_b(x_a), f_b(x_m))]$ by combining two types of similarity, where $f_a$ extracts adjacency matrices to compute graph similarity (measuring knowledge structure consistency), and $f_c$ and $f_b$ leverage CLIP and BLIP, respectively, to extract semantic embeddings for computing representation similarity (measuring semantic consistency). The candidate with the highest QS is finally selected as the output.
    - **Design Motivation**: Relying solely on generation quality cannot guarantee accuracy. Evaluation must be conducted from two complementary dimensions, structure and semantics, where graph similarity captures alignment at the knowledge level, and semantic similarity captures consistency at the representation level.

### Loss & Training
This method is a training-free framework and does not require any training loss. The entire pipeline relies completely on pre-trained models (Qwen2-VL as the LMM, SDXL/Cheff as the image generators, and CLIP/BLIP as the semantic evaluators), achieving zero-shot completion through prompt engineering and in-context learning.

## Key Experimental Results

### Main Results

| Dataset | Missing Rate | Metric | Knowledge Bridger (7B) | Prev. SOTA | Gain |
|--------|--------|------|----------------------|----------|------|
| COCO-2014 | η=0.7 | F1 | 77.9% | 72.3% (MPLMM) | +5.6% |
| COCO-2014 | η=0.7 | AP | 83.5% | 80.1% (MPLMM) | +3.4% |
| MM-IMDb | η=0.7 | F1 | 55.2% | 49.1% (MPLMM) | +6.1% |
| MM-IMDb | η=0.7 | AP | 61.8% | 56.2% (MPLMM) | +5.6% |
| IU X-ray (OOD) | η=0.7 | F1 | 46.3% | 36.8% (MPMM) | +9.5% |
| IU X-ray (OOD) | η=0.7 | AP | 70.5% | 61.9% (MPLMM) | +8.6% |

The improvement is particularly significant on the OOD medical dataset IU X-ray, with an F1 gain of 9.5% and an AP gain of 8.6% under a high missing rate.

### Ablation Study

| Configuration | F1 (MM-IMDb) | F1 (IU X-ray) | Description |
|------|-------------|---------------|------|
| Full model (Qwen-VL-7B) | 55.2 | 46.3 | Full model |
| w/o Knowledge Modeling | -1.3 | -17.5 | Direct generation without knowledge modeling, OOD performance drops significantly |
| w/o Knowledge + Random Ranking | -1.6 | -19.3 | Same as above with random ranking |
| Random Ranking | -0.5 | -3.8 | Replace ranking with random ranking |
| w/o Knowledge Ranking | -0.2 | -1.9 | Remove graph similarity ranking |
| w/o Semantic Ranking | -0.2 | -3.6 | Remove semantic similarity ranking |

### Key Findings
- **Knowledge modeling is the most critical component**: Removing knowledge modeling in the OOD scenario (IU X-ray) results in a sharp drop in F1 by 17.5%, indicating that domain knowledge is vital for cross-domain generalization.
- **The method scales effectively with LMM size**: All metrics continuously improve from 2B to 7B models, with even better performance when using GPT-4o.
- **Synthetic data enhances downstream tasks**: Generated synthetic modality data not only completes the missing modalities but also boosts the performance of other MMC models.
- Semantic ranking is more critical than graph structure ranking in OOD scenarios (-3.6 vs -1.9), while both contribute comparably in general scenarios.

## Highlights & Insights
- **Training-free paradigm**: Requiring absolutely no training on the target domain, it cleverly utilizes the in-context learning and CoT reasoning capabilities of LMMs to complete cross-domain modalities, breaking the dependency of the MMC field on large amounts of training data.
- **Knowledge graph as a bridge**: Converting the unstructured understanding of LMMs into structured knowledge graphs, which then guide generation and ranking. The design of this "knowledge bridge" makes the entire pipeline both interpretable and controllable.
- **Decoupled generation-and-ranking design**: The strategy of generating multiple candidates and then ranking them for selection can be transferred to other tasks requiring high-precision generation, such as text-to-image and cross-modal retrieval.

## Limitations & Future Work
- **Dependency on LMM quality**: Performance correlates strongly with the LMM. Small models (2B) are visibly weaker than large models (7B), while large models entail high inference costs.
- **Slow generation speed**: Generating 5 candidates per sample, building separate knowledge graphs, and computing similarities limits real-time capability.
- **Scalability of knowledge extraction rules**: Current entity-relation extraction rules require manual definition and may need adjustment for completely new domains.
- Image generation quality is bottlenecked by the underlying generators (SDXL/Cheff), which may lack precision in certain fine-grained medical imaging scenarios.
- An active learning strategy could be considered to allow the model to automatically decide which candidates require additional rounds of generation.

## Related Work & Insights
- **vs MMIN (ACL'21)**: MMIN learns missing representations via inter-modality interpolation, requiring training and performing poorly under OOD. Ours is training-free and exhibits strong OOD generalization, albeit at higher inference costs.
- **vs MPLMM (ACL'24)**: MPLMM dynamically adjusts fusion strategies using prompt-learning but still requires training data. Ours is completely training-free, with the advantage being particularly pronounced under high missing rates.
- **vs DiCMoR (CVPR'23)**: DiCMoR is an interpolation method whose performance degrades severely under high missing rates. Ours utilizes generative models to maintain stable performance.
- The knowledge-graph-guided generation paradigm can inspire other scenarios requiring controllable generation, such as automated medical report generation and multimodal reasoning.

## Rating
- Novelty: ⭐⭐⭐⭐ First to apply LMMs to MMC tasks, with a novel knowledge graph bridge design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covering both general and OOD scenarios with a systematic ablation study.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and systematic method description.
- Value: ⭐⭐⭐⭐ The training-free paradigm holds significant practical value for data-scarce medical fields.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Training-Free Counterfactual Explanation for Temporal Graph Model Inference](../../ICLR2026/graph_learning/training-free_counterfactual_explanation_for_temporal_graph_model_inference.md)
- [\[ACL 2025\] Beyond Completion: A Foundation Model for General Knowledge Graph Reasoning](../../ACL2025/graph_learning/beyond_completion_a_foundation_model_for_general_knowledge_graph_reasoning.md)
- [\[NeurIPS 2025\] Uncertain Knowledge Graph Completion via Semi-Supervised Confidence Distribution Learning](../../NeurIPS2025/graph_learning/uncertain_knowledge_graph_completion_via_semi-supervised_confidence_distribution.md)
- [\[ICML 2026\] GILT: An LLM-Free, Tuning-Free Graph Foundational Model for In-Context Learning](../../ICML2026/graph_learning/gilt_an_llm-free_tuning-free_graph_foundational_model_for_in-context_learning.md)
- [\[AAAI 2026\] MyGram: Modality-aware Graph Transformer with Global Distribution for Multi-modal Entity Alignment](../../AAAI2026/graph_learning/mygram_modality-aware_graph_transformer_with_global_distribution_for_multi-modal.md)

</div>

<!-- RELATED:END -->
