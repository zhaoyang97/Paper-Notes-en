---
title: >-
  [Paper Note] How Do LLMs Acquire New Knowledge? A Knowledge Circuits Perspective on Continual Pre-Training
description: >-
  [ACL 2025][LLM Pretraining][Knowledge Circuits] This work investigates the mechanisms of new knowledge acquisition during continual pre-training of LLMs from the perspective of knowledge circuit evolution. Across GPT-2, Llama, and Phi architectures, the authors find that: (1) new knowledge related to existing knowledge is easier to acquire; (2) knowledge circuits undergo a distinct phase transition of "formation $\rightarrow$ optimization"; (3) circuit evolution follows a dee…
tags:
  - "ACL 2025"
  - "LLM Pretraining"
  - "Knowledge Circuits"
  - "Continual Pre-training"
  - "Knowledge Acquisition"
  - "Circuit Evolution"
  - "Phase Transition"
date: 2026-05-08
content_hash: 1199c8c4dcf4210e
---

# How Do LLMs Acquire New Knowledge? A Knowledge Circuits Perspective on Continual Pre-Training

**Conference**: ACL 2025  
**arXiv**: [2502.11196](https://arxiv.org/abs/2502.11196)  
**Code**: [GitHub](https://github.com/zjunlp/DynamicKnowledgeCircuits)  
**Area**: LLM Pre-training / Mechanistic Interpretability  
**Keywords**: Knowledge Circuits, Continual Pre-training, Knowledge Acquisition, Circuit Evolution, Phase Transition

## TL;DR

This work investigates the mechanisms of new knowledge acquisition during continual pre-training of LLMs from the perspective of knowledge circuit evolution. Across GPT-2, Llama, and Phi architectures, the authors find that: (1) new knowledge related to existing knowledge is easier to acquire; (2) knowledge circuits undergo a distinct phase transition of "formation $\rightarrow$ optimization"; (3) circuit evolution follows a deep-to-shallow pattern, where extraction functions are first established in mid-to-deep layers, followed by the enrichment of knowledge representations in shallow layers.

## Background & Motivation

**Background**: LLMs can capture a vast amount of factual knowledge from the pre-training corpus and encode it as parametric knowledge, but the internal mechanisms of structurally embedding new knowledge remain unclear.

**Limitations of Prior Work**: (a) Prior studies treat knowledge components as isolated units (such as probing or viewing feed-forward layers as key-value memory), ignoring the collaboration among components; (b) Works like Allen-Zhu & Li analyze knowledge storage using probing methods, but do not focus on the dynamic process of knowledge acquisition; (c) Yao et al. proposed the concept of knowledge circuits but only analyzed already stored knowledge.

**Key Challenge**: How do LLMs transition new knowledge from "unknown" to "known" during continual pre-training? How does the internal computational structure evolve to accommodate new knowledge?

**Key Insight**: Track the topology, component roles, and information flow changes of knowledge circuits throughout the entire process of continual pre-training.

## Method

### Overall Architecture

(1) Construct a synthetic knowledge corpus (50k fictitious biographies, with frequencies following an exponential distribution); (2) Conduct continual pre-training on GPT-2 Small/Medium, TinyLlama, and Phi-1.5; (3) Use the EAP-IG method to discover knowledge circuits at each checkpoint; (4) Analyze circuit evolution at three levels: performance, topology, and components.

### Key Designs

1. **Synthetic Knowledge Corpus Construction**:
    - **Function**: Generate 50,000 fictitious person entities, each with 5 relations (birth date, city, profession, university, company).
    - **Core Idea**: Categorize knowledge into "related new knowledge" $K_{rel}$ (using real names from Wikipedia + fictitious attributes) and "completely new knowledge" $K_{compl}$ (completely fictitious names + fictitious attributes), with a ratio of 1:4. Frequencies follow an exponential distribution (1-27) to simulate the long-tail effect.
    - **Design Motivation**: Synthetic data ensures that the knowledge does not exist in the pre-training phase and allows precise control over knowledge types and frequencies.

2. **Knowledge Circuit Discovery and Entropy Metric**:
    - **Function**: Use EAP-IG to score each edge, keeping the top-n edges to form the circuit. Define knowledge circuit entropy to measure topological concentration.
    - **Core Idea**: $H(\mathcal{C}) = -\sum_{e \in E_\mathcal{C}} P(e) \log P(e)$, where $P(e) = S(e)/\sum S(e')$. Decreasing entropy indicates a more concentrated circuit, meaning critical paths are forming.
    - **Design Motivation**: Circuit entropy reflects the organization level of knowledge within the network.

3. **Phase Transition Detection**:
    - **Function**: Identify an inflection point where the rate of circuit entropy decline and the convergence rate of Jaccard similarity simultaneously shift at a specific epoch.
    - **Core Idea**: Before the inflection point = formation stage (circuits form rapidly); after the inflection point = optimization stage (topology stabilizes, but computational efficiency improves). The inflection point occurs at epoch 7 for GPT-2 Small and epoch 1 for Phi-1.5.
    - **Design Motivation**: Reveals that knowledge acquisition is not a linear process but exhibits phased qualitative changes.

### Loss & Training

Standard next-token prediction objective. The learning rate matches the end of the base model's pre-training phase. AdamW optimizer (β₁=0.9, β₂=0.95), weight decay=0.1. Trained on 2 A100 GPUs. Evaluation uses the Hit@10 metric and factual recall tasks for three relation types.

## Key Experimental Results

### Main Results

Influence of knowledge type on acquisition efficiency (GPT-2 Small, Hit@10):

| Knowledge Type | Final Performance | Epochs Required to Reach 80% Performance |
|---------|---------|-------------------|
| Related new knowledge $K_{rel}$ | Higher | Fewer (~5 epochs) |
| Completely new knowledge $K_{compl}$ | Lower | More (~10 epochs) |

Influence of knowledge frequency: high frequency > medium frequency > low frequency (positively correlated).

### Ablation Study

Topology alignment experiment (GPT-2 Small, Hit@10) — aligning circuits from different checkpoints to the topology of specific time points:

| Source of Aligned Topology | Final Hit@10 |
|------------|----------|
| Original (each checkpoint's own) | Highest |
| After Phase Transition (After) | 54% higher than Before |
| Before Phase Transition (Before) | Lower |
| Initialization (Init) | Lowest |

This demonstrates that **the topological evolution at the phase transition point is key to performance improvement**.

### Key Findings

1. **Related knowledge is easier to acquire than completely new knowledge**: The learning curve of $K_{rel}$ is consistently above $K_{compl}$, suggesting that utilizing data curriculum (organizing data similar to the structures in the original corpus) can improve the efficiency of continual pre-training.
2. **Universality of phase transitions**: All four models (ranging from 124M to 1.3B parameters) exhibit clear phase transition points, with larger models reaching the transition point earlier.
3. **Deep-to-shallow evolution pattern**:
    - Formation stage: Mover heads (extraction function) in mid-to-deep layers gradually increase, while relation heads decrease.
    - Optimization stage: The topology stabilizes, but shallow MLPs enrich knowledge representations, leading to the early decoding phenomenon.
4. **Circuits for low-frequency knowledge perform well**: Transfer experiments show that low-frequency circuits perform similarly to high-frequency circuits on high-frequency test sets, indicating that the bottleneck lies in insufficient representation rather than circuit capacity.
5. **Knowledge circuits are elastic**: Even if over 60% of the edges are replaced after learning new knowledge, data replay can reactivate the original circuits.

## Highlights & Insights

- **Three-level analysis framework (performance-topology-component)**: Comprehensively characterizes the knowledge acquisition process from macro to micro.
- **Phase transition discovery**: The inflection point of knowledge circuit entropy can serve as a monitoring signal for continual pre-training.
- **Circuit Elasticity**: Even when knowledge is behaviorally "forgotten," the circuit retains the potential to be reactivated.
- **Practical insights**: Data curriculum design should prioritize content related to existing knowledge; long-tail knowledge can be improved through data augmentation (rather than using larger models).

## Limitations & Future Work

- Experiments were only conducted on decoder-only Transformers, without covering encoder-decoder architectures.
- Model sizes were limited to 1.3B parameters; the behavior of larger models (>7B) remains unverified.
- Only standard next-token prediction training was used, without analyzing the impact of new training techniques such as instruction tuning.
- There is a gap between synthetic data and real-world knowledge distributions.

## Related Work & Insights

- **Yao et al. (2024) Knowledge Circuits**: This study directly extends their concept, pushing from static analysis to dynamic evolution analysis.
- **Allen-Zhu & Li (2024) Physics of LMs**: While they analyze knowledge storage through probing methods, this work provides a complementary perspective from the circuit angle.
- **Tigges et al. (2024)**: They analyze the formation of general circuits during pre-training, whereas this work focuses specifically on the evolution of knowledge circuits during continual pre-training.
- Insights: The state of knowledge circuits can serve as a dynamic indicator for training strategy adjustments (e.g., learning rate, data mixing).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First study on knowledge acquisition from the perspective of knowledge circuit evolution; the three discoveries (relevance effect, phase transition, deep-to-shallow) are all novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across four models, multi-dimensional analysis, transfer experiments, and forgetting analysis, though limited by model scale.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear layer-by-layer analysis, exquisite figures, and highly persuasive visualization of the evolution process.
- Value: ⭐⭐⭐⭐⭐ Holds profound theoretical value for understanding LLM knowledge acquisition mechanisms and offers practical guidance for continual pre-training strategies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Improving Continual Pre-training Through Seamless Data Packing](improving_continual_pre-training_through_seamless_data_packing.md)
- [\[ACL 2025\] Towards Effective and Efficient Continual Pre-training of Large Language Models](towards_effective_and_efficient_continual_pre-training_of_large_language_models.md)
- [\[ACL 2025\] Velocitune: A Velocity-based Dynamic Domain Reweighting Method for Continual Pre-training](velocitune_a_velocity-based_dynamic_domain_reweighting_method_for_continual_pre-.md)
- [\[ACL 2025\] Incorporating Domain Knowledge into Materials Tokenization](incorporating_domain_knowledge_into_materials_tokenization.md)
- [\[ACL 2025\] An Effective Incorporating Heterogeneous Knowledge Curriculum Learning for Sequence Labeling](dual_stage_curriculum_learning_sequence_labeling.md)

</div>

<!-- RELATED:END -->
