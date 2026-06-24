---
title: >-
  [Paper Note] How Do Transformers Learn Implicit Reasoning?
description: >-
  [NeurIPS 2025 Spotlight][Interpretability][implicit reasoning] By training Transformers from scratch in a finely controlled symbolic environment, this work reveals that multi-hop implicit reasoning progresses through three stages: "memorization $\to$ in-distribution generalization $\to$ cross-distribution generalization." The core mechanism is not the decodability of intermediate entities, but rather their clustering consistency in the cosine space—reasoning capabilities emer…
tags:
  - "NeurIPS 2025 Spotlight"
  - "Interpretability"
  - "implicit reasoning"
  - "multi-hop"
  - "grokking"
  - "cosine clustering"
  - "semantic patching"
  - "Transformer internals"
date: 2026-05-08
content_hash: 7a0c38fb92a1e62a
---

# How Do Transformers Learn Implicit Reasoning?

**Conference**: NeurIPS 2025 Spotlight  
**arXiv**: [2505.23653](https://arxiv.org/abs/2505.23653)  
**Code**: [GitHub](https://github.com/Jiaran-Ye/ImplicitReasoning)  
**Area**: Interpretability / Multi-hop Reasoning Mechanisms  
**Keywords**: implicit reasoning, multi-hop, grokking, cosine clustering, semantic patching, Transformer internals  

## TL;DR

By training Transformers from scratch in a finely controlled symbolic environment, this work reveals that multi-hop implicit reasoning progresses through three stages: "memorization $\to$ in-distribution generalization $\to$ cross-distribution generalization." The core mechanism is not the decodability of intermediate entities, but rather their clustering consistency in the cosine space—reasoning capabilities emerge only when the representations of the same intermediate entity form tight clusters across different queries.

## Background & Motivation

**Background**: LLMs can perform not only explicit reasoning via Chain-of-Thought (CoT), but also implicit reasoning (directly outputting the final answer without explicitly generating intermediate steps). However, the internal mechanisms of implicit reasoning—specifically, whether the models perform genuine step-by-step reasoning or merely rely on memorization—remain unclear.

**Limitations of Prior Work**: (1) Analyses based on pre-trained LLMs cannot precisely control the training data, making it difficult to distinguish "genuine reasoning" from "memorization/shortcuts." (2) Existing symbolic datasets (such as wang2024grokking) train models from scratch but lack query-level fine control and behavioral granularity, failing to isolate the specific conditions of generalization. (3) Mainstream analysis tools (such as logit lens decoding and causal patching) have inherent limitations—logit lens reveals correlations rather than causation, whereas causal patching tests causal influence but does not verify the actual semantic content of representations.

**Key Challenge**: Correct answers in implicit reasoning can arise from two entirely distinct cognitive processes (step-by-step reasoning vs. memorization), which existing methods fail to distinguish reliably.

**Goal**: To reveal how Transformers acquire and perform implicit multi-hop reasoning in a controlled environment—including the developmental trajectory, necessary conditions, and internal mechanisms.

**Key Insight**: Constructing an expanded symbolic environment that supports query-level ablation, and introducing two new diagnostic tools (cross-query semantic patching + cosine representational lens) to connect behavioral observations with internal mechanisms.

**Core Idea**: The key to implicit reasoning is not whether intermediate entities can be "explicitly decoded," but whether their representations form highly consistent clusters in the cosine space—the emergence of the clustering structure corresponds precisely to the emergence of reasoning capability.

## Method

### Overall Architecture

Constructing a symbolic environment ($2000 \text{ entities} \times 200 \text{ relations} \to 40000 \text{ atomic triples} + \text{2-hop compositional queries}$) $\to$ Training GPT-2 from scratch $\to$ Behavioral analysis (three-stage developmental trajectory + ablation studies) $\to$ Mechanistic analysis (cross-query semantic patching localization + logit lens decodability + cosine clustering lens) $\to$ Closed-loop explanation (explaining behavioral phenomena using mechanisms)

### Key Designs

1. **A Finely Controlled Symbolic Reasoning Environment**
    - Atomic triples $(e_1, r_1) \to e_2$ are divided into ID (in-distribution) and OOD (out-of-distribution), sharing entities and relations.
    - 2-hop queries $(e_1, r_1, r_2) \to e_3$: the model must implicitly reason about the intermediate entity $e_2$.
    - Training set: Train-II (both hops are ID); Testing sets: Test-II / Test-OI / Test-IO / Test-OO.
    - Supports various ablation configurations: removing specific triples, restricting compositional roles, and removing subsets.
    - Data scale: 2000 entities, 200 relations, 38,000 ID triples, 2,000 OOD triples, and 273,600 Train-II queries.
    - **Design Motivation**: Query-level control allows precise isolation of "which training signals are necessary to solve a specific query."

2. **Cross-Query Semantic Patching**
    - Stronger than standard causal patching: it tests not only the causal impact of hidden states but also their semantic content.
    - **Mechanism**: Extract the hidden vectors at candidate positions from a source query $(e_1, r_1, r_2)$ and patch them into the same position of a structurally similar target query $(e_5, r_6, r_7)$.
    - Success Criterion: If the model's prediction after patching changes from $r_7(r_6(e_5))$ to $r_7(r_1(e_1))$, it indicates that the patched representation carries transferable semantic information of the intermediate entity.
    - Result: Effective patching mainly occurs at the intermediate layers (layer 5 of an 8-layer GPT-2) at the $r_1$ token position.
    - **Design Motivation**: To go beyond "correlation" (linear probing) and "surface causality" (standard patching) to test the semantic transferability of representations.

3. **Cosine-Based Representational Lens**
    - **Key Insight**: Instead of asking "can this hidden state be decoded?", it asks "how are these representations organized across different contexts?"
    - **Mechanism**: For all queries sharing the same intermediate entity $e_2$, extract $\mathbf{h}_{r_1}^5$, compute pairwise cosine distances, and visualize using MDS projection.
    - Defines two quantitative metrics:
        - **ID Cohesion Score**: The average cosine similarity between ID-derived representations and their centroid (in-distribution consistency).
        - **OOD Alignment Score**: The average cosine similarity between OOD-derived representations and the ID centroid (cross-distribution alignment).
    - **Design Motivation**: The logit lens decoding success rate does not correspond to the emergence of reasoning capabilities ("decodable $\neq$ reason-capable"), necessitating a new perspective on representational analysis.

### Three-Stage Developmental Trajectory

- **Phase I Memorization**: Fast fitting of training data (atomic facts + 2-hop composition), but fails to generalize to unseen queries.
- **Phase II In-Distribution Generalization**: Begins to generalize to unseen ID-ID compositions (Test-II), resembling the grokking phenomenon.
- **Phase III Cross-Distribution Generalization**: Gradually incorporates OOD triples into the first-hop reasoning (Test-OI), but consistently fails when the second hop is OOD.

### Closed-Loop Mechanistic Explanation

Behavioral phenomena can be fully explained by the cosine clustering mechanism. Training on ID triples shares the prefix structure at the $r_1$ position via the autoregressive causal mask—such that $(e_1, r_1)$ produces identical hidden states in both atomic and 2-hop queries. This constrains the intermediate entity representation to fall within a subspace supporting decoding, accelerating cluster convergence. OOD representations are gradually pulled into ID clusters by frequent OOD atomic triple training, making the first hop appear to generalize; however, this is a byproduct of the ID anchoring effect rather than true cross-distribution reasoning generalization. The second hop, lacking the anchoring effect from causal mask position-sharing, must rely on direct query-level supervision. 3-hop reasoning experiments further validate this: only Test-OII (OOD $\to$ ID $\to$ ID) succeeds, while all configurations with OOD appearing in subsequent hops fail. Experiments also find that the model initially attempts to explicitly decode the intermediate entity but quickly abandons this strategy, shifting to an implicit but geometrically consistent internal representation scheme.

### Key Findings

- **ID Triples are Non-essential but Accelerate Generalization**: Training only with Train-II (without atomic triples) can still generalize to Test-II, but adding ID triples significantly accelerates it (due to the shared hidden state at the $r_1$ position, which constrains the representation space).
- **Second-hop Generalization Requires Query-level Matching**: The model must encounter specific second-hop combinations during training to generalize; moreover, higher exposure frequency of the second hop leads to earlier correct answers on corresponding queries.
- **Decodability $\neq$ Reason-ability**: ID-derived intermediate entity representations achieve a 97%+ decoding success rate in Phase I, yet reasoning capability only emerges in Phase II. Furthermore, there is no significant difference in decoding success rates between ID and OOD representations, despite their vast difference in reasoning performance.
- **First-hop OOD Generalization is Pseudo-Generalization**: It is essentially a byproduct of the ID anchoring effect; removing ID triples causes OOD reasoning to fail completely.

## Key Experimental Results

### Intermediate Entity Decoding Success Rate vs. Reasoning Phases

| Source | Immediate Probing | | | Full-run Probing | | |
|------|---------|---------|---------|---------|---------|---------|
| | Phase I | Phase II | Phase III | Phase I | Phase II | Phase III |
| ID-derived | 92.1% | 98.8% | 99.9% | 97.1% | 99.9% | 99.9% |
| OOD-derived | 67.7% | 81.3% | 99.8% | 83.7% | 98.6% | 99.7% |

### Behavioral Ablation

| Configuration | Test-II Generalization | Test-OI Generalization | Remarks |
|------|-------------|-------------|------|
| Base (Full Data) | ✅ Phase II | ✅ Phase III | Full three stages |
| Train-II Only (No Atomic Triples) | ✅ (Delayed) | ❌ | ID triples accelerate but are non-essential |
| Remove Specific Second-hop Atomic Triples | ✅ (Other queries) | - | Corresponding queries consistently fail |
| ID/OOD=0.3/0.7 | ✅ | ❌ | ID dominance is critical for Phase III |
| ID/OOD=0.8/0.2 | ✅ | ✅ | Sufficient ID exposure drives OOD alignment |

### Correspondence Between Cosine Clustering and Reasoning Capabilities

- Phase I: ID and OOD representations are scattered randomly in the cosine space.
- Phase II: ID representations form tight clusters (ID Cohesion Score increases), co-occurring with the emergence of Test-II generalization.
- Phase III: OOD representations begin to align with ID clusters (OOD Alignment Score increases), co-occurring with the emergence of Test-OI generalization.

## Highlights & Insights

- **"Decodable $\neq$ Reason-capable" is the core insight**—overturning the implicit assumption of many prior works (i.e., that the model performs reasoning if the logit lens can decode intermediate entities).
- Cross-query semantic patching is stronger than standard causal patching: it tests the semantic transferability of representations rather than mere causal influence.
- The cosine clustering lens provides a geometric explanation for the emergence of reasoning capabilities: intermediate entities must form an "abstraction" in the representation space (mapping the same entity in different contexts to nearby vectors) to be reused.
- The "spurious" nature of the first-hop OOD generalization—it is not true generalization, but a byproduct of representational alignment driven by ID supervision—a finding with profound implications for understanding the generalization boundaries of LLMs.
- The hard requirement of second-hop generalization (query-level matching) explains why single-hop knowledge does not automatically transfer to multi-hop reasoning.

## Limitations & Future Work

- Experiments are conducted in a controlled symbolic environment ($2000 \text{ entities} \times 200 \text{ relations}$), which has a significant gap from the complex knowledge bases of real LLMs. Knowledge interaction mechanisms in real LLMs may differ from those observed in symbolic environments; thus, the findings should be regarded as "preliminary guidance" rather than a complete explanation.
- The analysis primarily uses an 8-layer GPT-2. Although basic consistency was verified on Qwen2.5-1.5B, instabilities in Phase III and decoupling between ID Cohesion and Test-II accuracy were observed—suggesting that in larger models, clustering might only be a necessary, but not sufficient, condition for reasoning.
- The causal relationship of the cosine clustering phenomenon has not been rigorously established—clustering is highly correlated with reasoning performance, but whether it is a sufficient condition remains unproven, requiring intervention experiments to establish causal direction.
- Only 2-hop queries are considered (supplemented by 3-hop validation); whether the developmental trajectory and mechanisms remain consistent over longer reasoning chains (5-hop+) remains to be explored.
- The findings are not yet connected to practical application scenarios such as knowledge editing or CoT reasoning optimization in real LLMs, and their practical guidance value remains to be explored.

## Related Work & Insights

- **vs. wang2024grokking**: This work substantially extends their symbolic dataset by adding OOD partitions, query-level ablations, and finer behavioral resolution. It discovers Phase III cross-distribution generalization beyond grokking and demonstrates that atomic triples are not necessary for generalization.
- **vs. hopping-too-late / yang2024large2**: These works study implicit reasoning failures based on pre-trained LLMs but suffer from uncontrollable training data. In contrast, training from scratch provides a cleaner causal analysis, and the query-level matching requirement explains the multi-hop reasoning failures observed in those studies.
- **vs. logit lens series** (sakarvadia2024towards, li2024understanding, zhang2024locate): These works assume "decodable intermediate entity = model is reasoning." This paper refutes this assumption through the three-phase decoupling of decoding and reasoning, providing the cosine clustering lens as an alternative explanatory framework.
- **vs. Balesni et al. (2024)**: They found that knowledge within the same paragraph is easier to compose for reasoning. This work provides a more precise mechanistic explanation from a query-level matching perspective—the key lies not in temporal/physical proximity of knowledge, but in the direct exposure of compositional structures.
- **Insights for Latent CoT**: Results show that implicit reasoning does not require intermediate steps to be decodable in the token space, supporting the direction of performing reasoning in the latent space (e.g., continuous thoughts).
- **Methodological Insights**: Cross-query semantic patching can be generalized to other scenarios requiring tests of representational semantic content (rather than just causal impact), such as testing whether updated representations correctly propagate to downstream reasoning in knowledge editing.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The "decodable $\neq$ reason-capable" insight and the cosine clustering lens are both significant, original contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ The three-layer structure of behavioral ablation + mechanistic analysis + closed-loop explanation is highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain is exceptionally clear, seamlessly linking behavioral observations, mechanistic analyses, and closed-loop explanations.
- Value: ⭐⭐⭐⭐ Provides deep insights into the reasoning mechanisms of LLMs, though generalization from symbolic environments to real-world LLMs still warrants further validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] How Do Transformers Learn to Associate Tokens: Gradient Leading Terms Bring Mechanistic Understanding](../../ICLR2026/interpretability/how_do_transformers_learn_to_associate_tokens_gradient_leading_terms_bring_mecha.md)
- [\[NeurIPS 2025\] Base Models Know How to Reason, Thinking Models Learn When](base_models_know_how_to_reason_thinking_models_learn_when.md)
- [\[NeurIPS 2025\] Uncovering Graph Reasoning in Decoder-only Transformers with Circuit Tracing](uncovering_graph_reasoning_in_decoder-only_transformers_with_circuit_tracing.md)
- [\[ICLR 2026\] How Transformers Learn Causal Structures In-Context: Explainable Mechanism Meets Theoretical Guarantee](../../ICLR2026/interpretability/how_transformers_learn_causal_structures_in-context_explainable_mechanism_meets_.md)
- [\[NeurIPS 2025\] How Intrinsic Motivation Shapes Learned Representations in Decision Transformers: A Cognitive Interpretability Analysis](toward_explainable_offline_rl_analyzing_representations_in_intrinsically_motivat.md)

</div>

<!-- RELATED:END -->
