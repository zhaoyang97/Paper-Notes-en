---
title: >-
  [Paper Note] Acquisition and Application of Novel Knowledge in Large Language Models
description: >-
  [ACL 2025][LLM (Other)][New Knowledge Acquisition] This paper proposes the PermAR framework, which endows autoregressive models with bidirectional knowledge acquisition capabilities through permuted language modeling. It also constructs NovelHuman, a new knowledge dataset based on simulating biological evolution over knowledge graphs. The authors find that the position of knowledge within a sentence significantly affects the knowledge acquisition performance of LLMs. On new k…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "New Knowledge Acquisition"
  - "Permuted Language Modeling"
  - "Knowledge Graphs"
  - "Bidirectional Knowledge Acquisition"
  - "Autoregressive Models"
date: 2026-05-08
content_hash: b60d1a6824431a24
---

# Acquisition and Application of Novel Knowledge in Large Language Models

**Conference**: ACL 2025  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: New Knowledge Acquisition, Permuted Language Modeling, Knowledge Graphs, Bidirectional Knowledge Acquisition, Autoregressive Models

## TL;DR

This paper proposes the PermAR framework, which endows autoregressive models with bidirectional knowledge acquisition capabilities through permuted language modeling. It also constructs NovelHuman, a new knowledge dataset based on simulating biological evolution over knowledge graphs. The authors find that the position of knowledge within a sentence significantly affects the knowledge acquisition performance of LLMs. On new knowledge injection tasks, PermAR improves performance by 3.3%–38% compared to existing methods.

## Background & Motivation

**Background**: Large Language Models (LLMs) encode a vast amount of world knowledge through their massive parameters, demonstrating impressive capabilities in various generative tasks. However, practical applications require continuously injecting new knowledge into LLMs—such as the latest facts and newly discovered entity relations. Existing knowledge injection methods include continual pre-training, RAG (Retrieval-Augmented Generation), and knowledge editing.

**Limitations of Prior Work**: Existing studies exhibit two core flaws when constructing new knowledge datasets: (1) Timestamp-based methods (which use facts after the training cutoff date as "new knowledge") lack rigor, as this knowledge might have been indirectly learned by the model through other channels. (2) Simple template-based synthesis methods are overly mechanical (e.g., "X is an attribute of Y"), and the generated training data fails to reflect the complex expressions and diversity of real-world knowledge. Furthermore, existing methods overlook a key finding: the position of knowledge within a sentence significantly impacts whether the model can effectively acquire that knowledge.

**Key Challenge**: There is an inherent contradiction between the unidirectional training objective of autoregressive (AR) models and the multi-angle expression of knowledge. AR models predict tokens from left to right sequentially. They can effectively acquire knowledge that appears in the latter half of a sentence and depends on previous context. However, their acquisition efficiency is very low for knowledge that appears at the beginning of a sentence or requires bidirectional context to understand.

**Goal**: (1) Construct a genuinely "novel" knowledge dataset to ensure that the model has never seen these entities; (2) Analyze the impact of knowledge positioning on acquisition performance; (3) Design a training framework that enables bidirectional knowledge acquisition without altering the AR architecture.

**Key Insight**: Starting from knowledge graphs, the authors simulate biological evolution processes to generate entirely new fictitious entities. Completely absent from the training data, these entities possess diverse combinations of attributes, ensuring a strict definition of "novelty." After analyzing the position effect of knowledge, inspiration was drawn from Permutation Language Modeling (such as the strategy used in XLNet).

**Core Idea**: Enhance the knowledge acquisition capabilities of AR models through intra-sentence permutation. By randomly shuffling the order of knowledge units within a knowledge-bearing sentence during training, each knowledge segment has the opportunity to be predicted in different positions, thereby achieving de facto bidirectional knowledge acquisition.

## Method

### Overall Architecture

The framework consists of two core components: (1) The NovelHuman dataset construction pipeline, which utilizes entities and attributes from knowledge graphs to generate entirely new fictional entities and their attribute descriptions by simulating biological evolution. (2) The PermAR training framework, which introduces an intra-sentence permutation mechanism during autoregressive language model training, enabling the model to acquire knowledge from multiple directions. The input is a text corpus containing new knowledge, and the output is an LLM capable of correctly recalling and applying the new knowledge after knowledge injection.

### Key Designs

1. **NovelHuman Dataset Construction (Biological Evolution Simulation)**:

    - **Function**: Generate strictly novel knowledge data, ensuring that these entities and attribute combinations do not exist in the LLM training corpus.
    - **Mechanism**: Extract the attribute schemas of entities (e.g., a person has attributes like name, birthplace, occupation, and achievements) from existing knowledge graphs. Then, through mechanisms akin to biological evolution—crossover (combining attributes of different entities) and mutation (randomly modifying some attribute values)—generate completely new fictional entities (such as a fictitious scientist with an attribute combination that does not exist in the real world). These synthesized entities are subsequently converted into various natural language expressions.
    - **Design Motivation**: Compared to timestamp-based methods and simple templates, evolutionary simulation ensures the absolute novelty of knowledge (precluding its existence in the pre-training data) while producing diverse yet reasonable attribute combinations (conforming to knowledge graph constraints). Consequently, the generated text is closer to real-world knowledge expressions.

2. **Intra-sentence Knowledge Position Analysis**:

    - **Function**: Quantitatively reveal the impact of knowledge positioning within a sentence on the knowledge acquisition performance of AR models.
    - **Mechanism**: Place new knowledge in different positions in the sentence (beginning, middle, end) and test the model's success rate in acquiring that knowledge. The results reveal that knowledge appearing in the latter part of a sentence is more easily acquired by AR models (since the preceding text provides rich contextual predictive signals), whereas knowledge at the beginning of a sentence is acquired with significantly lower efficiency.
    - **Design Motivation**: This analysis provides a direct motivation for the design of PermAR—if position affects acquisition efficiency, allowing each piece of knowledge to have the opportunity to appear in favorable positions through permutation can uniformly enhance knowledge acquisition.

3. **PermAR Permuted Autoregressive Training Framework**:

    - **Function**: Seamlessly integrate into mainstream AR architectures, empowering models with bidirectional knowledge acquisition capabilities.
    - **Mechanism**: During training, multiple permutation orders are randomly generated for each sentence containing new knowledge. Under each permutation, the model still makes predictions autoregressively from left to right. However, because the relative positions of knowledge units are shuffled, the model is forced to learn the same knowledge from context in different directions. Crucially, permutations are used only during training; standard left-to-right generation is maintained during inference. PermAR achieves the permutation effect through a carefully designed attention mask without modifying the model architecture, directly adapting to existing Transformer-based AR models.
    - **Design Motivation**: Distinct from full permutation language models like XLNet, PermAR is optimized specifically for knowledge acquisition scenarios by permuting only segments containing key knowledge, avoiding the training instability and computational overhead of global permutation. At the same time, it resolves potential conflicts between AR training objectives and permutation learning.

### Loss & Training

Standard language modeling loss (cross-entropy) is computed on the permuted sequences. Training is split into two phases: first, performing knowledge injection training on the NovelHuman dataset, followed by evaluating the model's ability to recall and apply the new knowledge. The randomness of permutations and sampling strategies are meticulously controlled during training to balance knowledge acquisition efficiency and language fluency.

## Key Experimental Results

### Main Results

| Method | Knowledge Recall Accuracy | Knowledge Application Accuracy | Relative Gain |
|------|--------------|--------------|---------|
| Standard Continual Pre-training | Baseline | Baseline | - |
| RAG | Medium | Medium | - |
| Knowledge Editing Methods | Medium | Medium | - |
| PermAR (Ours) | Highest | Highest | +3.3%~38% |

### Ablation Study

| Configuration | Knowledge Acquisition Performance | Description |
|------|-----------|------|
| PermAR Full Framework | Best | Permutation + Knowledge Position Optimization |
| No Permutation (Standard AR) | Poor | Obvious Position Bias |
| Permutation without Handling Conflicts | Medium | Interference between AR Objectives and Permutation |
| Knowledge only at Sentence Start | Poor | Verify Position Effect |
| Knowledge only at Sentence End | Good | Verify Position Effect |
| Random Position | Medium | Baseline Position Strategy |

### Key Findings

- The position of knowledge within a sentence has an impact of up to dozens of percentage points on the knowledge acquisition performance of AR models, a finding that is highly valuable in its own right.
- The core advantage of PermAR lies in its "position robustness"—regardless of where the knowledge appears in the sentence, the acquisition efficiency remains stable.
- On the NovelHuman dataset, PermAR outperforms the best existing knowledge enhancement methods by 3.3%–38%, with a more pronounced advantage particularly on knowledge with complex attribute combinations.
- Data generated through biological evolution simulation achieves significantly better training results than template-based and timestamp-based methods.

## Highlights & Insights

- **Discovery of the Knowledge Position Effect**: This insight is highly profound—autoregressive models exhibit a "position bias" when processing knowledge, meaning that the learning outcome varied greatly when the same information is phrased differently. This finding can directly guide the construction and data augmentation strategies of training data.
- **Data Synthesis Approach combining Biological Evolution and Knowledge Graphs**: Introducing biological evolution mechanisms into knowledge data construction is an ingenious design, which ensures absolute novelty while maintaining a reasonable knowledge structure. This approach can be transferred to any scenario that requires synthesized data with guaranteed novelty.
- **Architecture-free Improvement for AR Models**: PermAR achieves the permutation effect through attention masking and is fully compatible with existing frameworks, which greatly reduces the cost of practical deployment.

## Limitations & Future Work

- The method is primarily validated on small to medium-scale models; its performance on larger-scale models (e.g., 70B+) needs further investigation.
- The knowledge types in the NovelHuman dataset lean toward entity-attribute knowledge; its applicability to more complex types, such as procedural knowledge or causal knowledge, remains to be verified.
- The training overhead of PermAR is increased compared to standard continual pre-training (due to the need to generate multiple permutations), and the efficiency trade-off in extremely large-scale data scenarios is worth exploring.
- The durability of knowledge (whether it fades with further training after injection) is not deeply analyzed.

## Related Work & Insights

- **vs ROME/MEMIT (Knowledge Editing Methods)**: Knowledge editing directly modifies factual mappings within model parameters. Although precise, it can only handle a small amount of knowledge at a time. PermAR achieves batch knowledge injection by improving the training method.
- **vs RAG (Retrieval-Augmented Generation)**: RAG externalizes knowledge within a retrieval database and dynamically retrieves it during inference. PermAR internalizes knowledge into model parameters, which achieves faster inference but differs in flexibility.
- **vs XLNet (Permutation Language Modeling)**: XLNet utilizes global permutation during the pre-training phase, whereas PermAR optimizes local permutation specifically for knowledge injection scenarios, avoiding the training instability of global permutation.

## Rating

- Novelty: ⭐⭐⭐⭐ The discovery of the knowledge position effect is novel, and the design of PermAR elegantly addresses the position bias of knowledge acquisition in AR models.
- Experimental Thoroughness: ⭐⭐⭐⭐ Systematic ablation studies and the 3.3%–38% improvement range demonstrate stable performance across different conditions.
- Writing Quality: ⭐⭐⭐⭐ The problem definition is clear, and the logical chain from discovery to solution is coherent.
- Value: ⭐⭐⭐⭐ Provides a fresh perspective of understanding and a practical solution to the critical challenge of LLM knowledge updating.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Contrastive Perplexity for Controlled Generation: An Application in Detoxifying Large Language Models](contrastive_perplexity_controlled_gen.md)
- [\[ACL 2025\] On the Acquisition of Shared Grammatical Representations in Bilingual Language Models](on_the_acquisition_of_shared_grammatical_representations_in_bilingual_language_m.md)
- [\[ACL 2025\] AI as a Novel Ethical Agent: Exploring Moral Judgments by Large Language Models](ai_as_a_novel_ethical_agent_exploring_moral_judgments_by_large_language_models.md)
- [\[ACL 2025\] Knowledge Boundary of Large Language Models: A Survey](knowledge_boundary_survey.md)
- [\[ACL 2025\] Dynamic Knowledge Integration for Evidence-Driven Counter-Argument Generation with Large Language Models](dynamic_knowledge_integration_for_evidence-driven_counter-argument_generation_wi.md)

</div>

<!-- RELATED:END -->
