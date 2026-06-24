---
title: >-
  [Paper Note] ClaimPKG: Enhancing Claim Verification via Pseudo-Subgraph Generation with Lightweight Specialized LLM
description: >-
  [ACL 2025][Model Compression][Knowledge Graph] Proposes the ClaimPKG framework, which utilizes a lightweight specialized LLM to convert textual claims into pseudo-subgraph representations, retrieves relevant subgraphs from a knowledge graph as evidence, and finally performs reasoning and verification using a general LLM, outperforming SOTA by 9%-12% accuracy on the FactKG dataset.
tags:
  - "ACL 2025"
  - "Model Compression"
  - "Knowledge Graph"
  - "Claim Verification"
  - "Pseudo-subgraph Generation"
  - "Trie-constrained Decoding"
  - "LLM Reasoning"
date: 2026-05-08
content_hash: c33dfbe61ba096d1
---

# ClaimPKG: Enhancing Claim Verification via Pseudo-Subgraph Generation with Lightweight Specialized LLM

**Conference**: ACL 2025  
**arXiv**: [2505.22552](https://arxiv.org/abs/2505.22552)  
**Code**: [https://github.com/HoangHoang1408/ClaimPKG](https://github.com/HoangHoang1408/ClaimPKG)  
**Area**: Model Compression  
**Keywords**: Knowledge Graph, Claim Verification, Pseudo-subgraph Generation, Trie-constrained Decoding, LLM Reasoning

## TL;DR

Proposes the ClaimPKG framework, which utilizes a lightweight specialized LLM to convert textual claims into pseudo-subgraph representations, retrieves relevant subgraphs from a knowledge graph as evidence, and finally performs reasoning and verification using a general LLM, outperforming SOTA by 9%-12% accuracy on the FactKG dataset.

## Background & Motivation

**Background**: Claim verification is a critical technology to combat the spread of misinformation, requiring systems to possess the capability of retrieving evidence from external knowledge sources and performing reasoning. Existing methods are primarily based on unstructured text corpora.

**Limitations of Prior Work**:
   - Most verification methods rely on unstructured text, decomposing claims via CoT reasoning, but the inherent limitations of text make it difficult to effectively handle entity ambiguity and multi-hop relationships.
   - Although Knowledge Graphs (KGs) provide structured relations, existing KG-based methods lack an end-to-end solution and usually require pre-extracted entities.
   - Although LLMs have strong reasoning capabilities, they perform poorly on KG-specific tasks such as entity disambiguation and multi-hop reasoning.

**Key Challenge**: How to simultaneously leverage the structural representation advantages of KGs and the reasoning capabilities of LLMs within a unified framework? Existing methods either only use text (lacking structural reasoning) or rely on highly modular KG methods (lacking end-to-end integration).

**Goal**: Address three major limitations: (1) Entity ambiguity: the system must accurately disambiguate entities in the claim; (2) Multi-hop reasoning: complex claims require reasoning across multiple evidence sources; (3) Limited integration between KG and LLMs.

**Key Insight**: Introduce a "pseudo-subgraph" as a bridge, utilizing a lightweight specialized LLM to transform textual claims into graph-structured representations, and then using a retrieval algorithm to find ground-truth evidence subgraphs in the KG.

**Core Idea**: Generate pseudo-subgraphs using a specialized small model + ensure entity correctness via Trie constraints + perform final reasoning with a general LLM, achieving a seamless connection from claims to KG subgraphs.

## Method

### Overall Architecture

ClaimPKG consists of three stages:
1. **Pseudo-Subgraph Generation**: A KG-specialized lightweight LLM generates pseudo-subgraphs under Trie constraints.
2. **Subgraph Retrieval**: A retrieval algorithm uses the pseudo-subgraph as a query to find relevant real subgraphs as evidence in the KG.
3. **General Reasoning**: A general LLM reasons over the claim and the retrieved subgraphs to produce a verdict and explanation.

Mathematically, $p_\theta(v,j|c,\mathcal{G})$ is decomposed as:

$$p_\theta(v,j|c,\mathcal{G}) = \sum_{\mathcal{S}_c} p_\theta(v,j|c,\mathcal{S}_c) \cdot p_\theta(\mathcal{S}_c|c,\mathcal{G})$$

The subgraph selection is further decomposed into a two-step process via the pseudo-subgraph $\mathcal{P}_c$:

$$p_\theta(\mathcal{S}_c|c,\mathcal{G}) = \sum_{\mathcal{P}_c} p_\theta(\mathcal{S}_c|\mathcal{P}_c,\mathcal{G}) \cdot p_\theta(\mathcal{P}_c|c,\mathcal{G})$$

### Key Designs

#### 1. Specialized LLM + Trie-Constrained Decoding

- **Function**: Translates textual claims into pseudo-subgraphs composed of triples $(e, r, e')$.
- **Mechanism**: Fine-tunes a lightweight LLM (e.g., Llama-3.2-3B) for joint entity-relation extraction. For indirectly referenced entities (not explicitly named), it uses an $\text{unknown}_i$ token to signal subsequent disambiguation needs.
- **Trie Constraints**: Constructs a Trie $\mathcal{T}$ of the KG entity set. During entity generation (between `<e>` and `</e>`), token selection is restricted to the Trie path, ensuring that 100% of the generated entities exist in the KG.
- **Multiple Representations**: Employs beam search (beam size=5) to generate multiple pseudo-subgraphs $\mathbb{P}_c = \{\mathcal{P}_c^{(i)}\}_{i=1}^N$, improving triple coverage.
- **Design Motivation**: General LLMs perform poorly on KG entity extraction (experiments show a 70B few-shot entity accuracy of only 86.52%), whereas fine-tuning a 3B model with Trie constraints achieves 100% accuracy.

#### 2. Subgraph Retrieval Algorithm

- **Function**: Matches triples in the pseudo-subgraph to real triples in the KG.
- **Mechanism**:
    - Categorizes pseudo-triples into **incomplete triples** (containing unknown entities) and **complete triples** (both endpoints are known entities).
    - Incomplete triples: For each unknown entity $u$, a candidate set of explicit entities $\mathcal{E}_u$ associated with it is collected, and the best candidate is selected using an entity scoring mechanism (Equation 5).
    - Complete triples: Uses a relation similarity function $\text{Sim}(r_1, r_2)$ to find the $k_2$ most similar relations between the two entities in the KG.
- **Relation Scoring Function**: Computes dot-product similarity using BGE-Large-EN-v1.5 embeddings.
- **Hyperparameter Settings**: $k_1=3$, $k_2=1$.
- **Design Motivation**: The pseudo-subgraph acts as a bridge from text to graph structure, solving the modality mismatch problem.

#### 3. General Reasoning Module

- **Function**: Generates a verdict $v$ and justification $j$ based on the claim $c$ and the retrieved evidence subgraph $\mathcal{S}_c^*$.
- **Mechanism**: Uses a general LLM (e.g., Llama-70B, Qwen-72B) to perform CoT reasoning.
- **Equation**: $p_\theta(v,j|c,\mathcal{S}_c^*) = p_\theta(v|c,j,\mathcal{S}_c^*) \cdot p_\theta(j|c,\mathcal{S}_c^*)$
- **Design Motivation**: Model-agnostic design, allowing flexible integration of different SOTA LLMs.

### Loss & Training

- Specialized LLM Training: Fine-tuned on the FactKG training set using the standard language modeling loss.
- Training Data Volume Analysis: Only 100 samples are needed to achieve satisfactory accuracy (Llama-3.2-3B: 79.35%), and performance saturates after 5K samples.
- The general LLM requires no training and performs zero-shot reasoning.

## Key Experimental Results

### Main Results

Accuracy comparison on the FactKG dataset (%):

| Method | Negation | Existence | Conjunction | Multi-hop | One-hop | Average |
|------|----------|-----------|-------------|-----------|---------|------|
| Zero-shot CoT (Llama-70B) | 64.34 | 64.62 | 72.47 | 65.58 | 78.32 | 69.07 |
| GEAR (Finetuned BERT) | 79.72 | 79.19 | 78.63 | 68.39 | 77.34 | 76.65 |
| KG-GPT (Llama-70B) | 70.91 | 65.06 | 86.64 | 58.87 | 92.02 | 74.70 |
| **ClaimPKG (3B* + Qwen-72B)** | **85.27** | **86.90** | 84.02 | **78.71** | 91.20 | **85.22** |
| **ClaimPKG (3B* + Llama-70B)** | 84.58 | 84.20 | **85.68** | 78.49 | 90.26 | 84.64 |

### Ablation Study

| Configuration | Entity Accuracy | Average Accuracy |
|------|-----------|-----------|
| Full ClaimPKG | 100.0% | 84.64% |
| Without Trie Constraint | 87.50% | 82.74% (-1.90) |
| Few-shot instead of Specialized LLM | 86.52% | 77.63% (-7.01) |
| Without Incomplete Triple Retrieval | 100.0% | 65.08% (-19.56) |

### Key Findings

1. **Evidence Retrieval is Crucial**: Pure LLM CoT achieves a maximum of only 69.07%, which is significantly lower than evidence-based methods.
2. **Specialized Small Models > General Large Models**: Fine-tuning a 1B specialized LLM outperforms few-shot learning of a 70B general LLM (83.91% vs. 77.63%).
3. **Pseudo-Subgraph Brings 12-point Gain**: ClaimPKG achieves 12% higher accuracy than KG-GPT and 9% higher than GEAR.
4. **Zero-shot Transfer**: Performs approximately 4% higher than Llama-70B CoT on HoVer and FEVEROUS.
5. **Error Analysis**: Out of 200 analyzed errors, 0% are structural errors, 28.5% are retrieval errors, and 71.5% are reasoning errors.

## Highlights & Insights

1. **Pseudo-Subgraph is a Key Innovation**: This intermediate representation resolves the modality mismatch between text and graph structures, which is much more effective than forcing LLMs to directly process KGs.
2. **Elegant Design of Trie Constraints**: It guarantees 100% entity correctness while allowing relations to be freely generated, balancing both precision and flexibility.
3. **Excellent Scalability**: When updating the KG, only the Entity-Trie needs to be updated without retraining the models.
4. **High Sample Efficiency**: Highly satisfactory results can be achieved with only 100 training samples, making the training costs extremely low.

## Limitations & Future Work

1. Reasoning errors account for 71.5%, indicating that general LLMs still fall short in complex reasoning scenarios; hence, the reasoning module needs further enhancement.
2. Excessive training samples (>5K) lead to overfitting, requiring regularization strategies.
3. Retrieval errors (28.5%) suggest that direct subgraph retrieval fails to provide complete evidence, necessitating implicit reasoning capabilities.
4. Currently only validated on DBpedia; the generalization ability to other KGs remains to be verified.
5. The inherent biases of LLMs may affect the reliability of the fact-checking system.

## Related Work & Insights

- **ProgramFC & FOLK**: Text-based modular verification pipelines; ClaimPKG unifies these steps.
- **KG-GPT**: A representative of prior KG+LLM methods, but its pipelined design limits performance.
- **StructGPT & RoG**: Combined KG-LLM works on related tasks such as KBQA, which inspired the design of ClaimPKG.
- Insight: The concept of pseudo-subgraphs can be generalized to other tasks requiring text-to-graph structure alignment.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combined design of pseudo-subgraph+Trie constraints is both novel and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Extremely comprehensive, covering multiple baselines, ablation studies, generalization, error analysis, and backbone comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ — The mathematical framework is clear, though some descriptions are slightly verbose.
- **Value**: ⭐⭐⭐⭐ — Offers a beneficial advancement to the field of KG-enhanced fact-checking.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] LightMem: Lightweight and Efficient Memory-Augmented Generation](../../ICLR2026/model_compression/lightmem_lightweight_and_efficient_memory-augmented_generation.md)
- [\[NeurIPS 2025\] Traversal Verification for Speculative Tree Decoding](../../NeurIPS2025/model_compression/traversal_verification_for_speculative_tree_decoding.md)
- [\[ACL 2025\] Predicting Through Generation: Why Generation Is Better for Prediction](predicting_through_generation_why_generation_is_better_for_prediction.md)
- [\[ACL 2025\] Direct Behavior Optimization: Unlocking the Potential of Lightweight LLMs](direct_behavior_optimization_unlocking_the_potential_of_lightweight_llms.md)
- [\[ACL 2025\] Bone Soups: A Seek-and-Soup Model Merging Approach for Controllable Multi-Objective Generation](bone_soups_multi_objective_gen.md)

</div>

<!-- RELATED:END -->
