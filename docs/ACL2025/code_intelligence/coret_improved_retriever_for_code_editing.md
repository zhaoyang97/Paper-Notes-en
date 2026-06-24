---
title: >-
  [Paper Note] CoRet: Improved Retriever for Code Editing
description: >-
  [ACL 2025][Code Intelligence][Code Retrieval] Proposed CoRet, a dense retrieval model tailored for code editing tasks. By integrating code semantics, repository-level file hierarchy, and call graph dependencies, and employing a log-likelihood loss function designed for repository-level retrieval, CoRet improves Recall by at least 15 percentage points over existing models on SWE-bench and Long Code Arena.
tags:
  - "ACL 2025"
  - "Code Intelligence"
  - "Code Retrieval"
  - "Code Editing"
  - "Call Graph"
  - "Repository-level"
  - "SWE-bench"
date: 2026-05-08
content_hash: 4df96d6e99f69428
---

# CoRet: Improved Retriever for Code Editing

**Conference**: ACL 2025  
**arXiv**: [2505.24715](https://arxiv.org/abs/2505.24715)  
**Code**: None  
**Area**: Code Retrieval / Code Edit Retrieval  
**Keywords**: Code Retrieval, Code Editing, Call Graph, Repository-level, SWE-bench

## TL;DR

Proposed CoRet, a dense retrieval model tailored for code editing tasks. By integrating code semantics, repository-level file hierarchy, and call graph dependencies, and employing a log-likelihood loss function designed for repository-level retrieval, CoRet improves Recall by at least 15 percentage points over existing models on SWE-bench and Long Code Arena.

## Background & Motivation

Code editing is a core task in software development: developers modify code repositories based on natural language descriptions (such as GitHub Issues/PRs). Successful code editing first requires accurately **retrieving the code segments that need to be modified**—which is particularly challenging in large, real-world repositories.

**Three Fundamental Problems of Existing Models**:

**Semantic Mismatch**: Existing pre-trained encoders (such as CodeBERT, GraphCodeBERT, and UniXcoder) learn the semantic matching of docstring-to-code, but the semantic relationship between issue descriptions and code snippets to be modified is completely different. Although CodeSage performs exceptionally well in code search, this capability cannot be directly transferred to retrieval for code editing.

**Missing Repository Structural Information**: Existing models treat each code snippet independently, losing repository-level hierarchical info such as file paths. However, the file path is an extremely strong retrieval signal—issue descriptions often contain implicit hints about the location of the file.

**Ignoring Runtime Dependencies**: The invocation relationships between functions (call graphs) are crucial for understanding code semantics—what a function calls and what calls it define its functional role. Existing models do not encode such dependencies.

## Method

### Overall Architecture

CoRet is a dual-encoder dense retrieval model:
- Query Encoder $Q(\cdot; \theta_q)$: Encodes the natural language issue description
- Code Encoder $C(\cdot; \theta_c)$: Encodes the code snippets (including contextual information)
- Retrieval scoring: $f(q, c_i) = \text{sim}(Q(q), C(c_i))$ (cosine similarity)
- The two encoders share weights ($\theta_c = \theta_q$), initialized based on CodeSage Small

### Key Designs

1. **Code Chunking Strategy**: Function -> Decomposes the repository into semantically meaningful atomic units; Mechanism -> Uses functions, classes, and class methods as basic units (chunks), with each chunk prefixed by its file path; Design Motivation -> File-level granularity is too coarse and lacks semantic cohesion, whereas function-level chunks are more suitable for retrieval localization. A repository can contain tens of thousands of chunks.

2. **Repository-level Log-likelihood Loss**: Function -> Uses log-likelihood loss instead of standard contrastive loss to train the retrieval model; Mechanism -> For each query $q_i$, maximizes the average log-likelihood of retrieving all correct code blocks $c^* \in \mathcal{C}_i^*$:

$$\mathcal{L}(\theta) = \frac{1}{N} \sum_i^N \frac{1}{|\mathcal{C}_i^*|} \sum_{c^* \in \mathcal{C}_i^*} \log \frac{\exp(\mathbf{q}_i \cdot \mathbf{c}^* / \tau)}{\Gamma(\mathbf{q}, \mathcal{C}_i)}$$

Since the normalization term involves the entire repository (which can consist of up to tens of thousands of chunks), it is approximated by randomly sampling in-instance negative samples ($\le 1024$); Design Motivation -> Standard contrastive loss utilizes cross-batch negative samples, but the core of retrieval for code editing is to distinguish between relevant and irrelevant code **within the same repository**, making in-instance negative samples more targeted. Temperature $\tau = 0.05$.

3. **Call Graph Context**: Function -> Concatenates the code text of call graph neighbors into each code block; Mechanism -> For each code block $c_i$, identifies its downstream callee neighbors $\mathcal{N}(c_i)$ and concatenates them as $[c_i; \text{[DOWN]}; c_{out}]$, with segment type embeddings added to distinguish the main code from the context; Design Motivation -> Call graphs reflect the runtime dependencies between code entities, and the functionality of potential callees helps understand the semantics of the caller. Only downstream neighbors (invoked functions) are used, prefixed with a special [DOWN] token.

4. **File Path Prefix**: Function -> Prepends the file path to each code block; Mechanism -> The path is directly concatenated as a string to the start of the chunk; Design Motivation -> File paths contain crucial semantic and location information (e.g., `tests/test_api.py` suggests this is API testing-related code), and the model learns to utilize this signal through training.

5. **Mean Pooling Instead of [CLS]**: Function -> Uses mean pooling over all tokens instead of the standard [CLS] token representation; Design Motivation -> Early experiments showed that mean pooling brings modest performance improvements.

### Loss & Training

- Loss function: Repository-level log-likelihood loss (see above), equivalent to multi-class classification cross-entropy
- Negative sampling strategy: Randomly sample up to 1024 in-instance negative samples (irrelevant chunks within the same repository)
- Temperature parameter: $\tau = 0.05$
- Weight sharing: Query and code encoders share parameters
- Training data: Repository-level code editing issues from SWE-bench, where ground truth is derived from parsed pull requests

## Key Experimental Results

### Main Results

**Perfect Recall@k (chunk-level)**

| Model | SWE-bench Verified @5 | @20 | MRR | LCA @5 | @20 | MRR |
|------|----------------------|-----|-----|--------|-----|-----|
| CodeSage S | 0.34 | 0.51 | 0.35 | 0.26 | 0.34 | 0.28 |
| CoRet −CG | 0.52 | 0.69 | 0.52 | 0.32 | 0.41 | 0.45 |
| CoRet −CG +file | 0.54 | 0.69 | 0.52 | 0.29 | 0.38 | 0.44 |
| **CoRet** | **0.54** | **0.71** | **0.53** | **0.32** | **0.47** | **0.47** |

On SWE-bench Verified, recall@5 improves by **52.9%** and recall@20 by approximately **35%** compared to CodeSage S.

### Ablation Study

**Impact of File Path Information (SWE-bench Verified, chunk Accuracy)**

| Model | Without File Path @5 | @20 | MRR | With File Path @5 | @20 | MRR |
|------|-------------|-----|-----|-------------|-----|-----|
| BM25 | 0.15 | 0.21 | 0.14 | 0.16 | 0.22 | 0.16 |
| CodeSage S | 0.40 | 0.58 | 0.33 | 0.40 | 0.57 | 0.35 |
| **CoRet** | 0.42 | 0.58 | 0.42 | **0.53** | **0.70** | **0.53** |

**Impact of the Number of Negative Samples**:
- Increasing from 8 to 1024 negative samples improves recall@20 by nearly 10 percentage points.
- Proves that in-instance negative samples are indeed more effective.

**Ablation of Call Graph Context**:
- CoRet −CG (without Call Graph) to CoRet: LCA @20 increases from 0.41 to 0.47 (+6pt).
- Replacing call graph neighbors with random chunks from the same file (CoRet −CG +file) instead degrades LCA performance, validating the specificity of call graphs.

### Key Findings

1. **Existing Models are Highly Insufficient**: Even the strongest baseline CodeSage S achieves only 34% recall@5 on SWE-bench, indicating a fundamental difference between code edit retrieval and traditional code search.
2. **Loss Function Improvement is the Major Contribution**: Moving from CodeSage S to CoRet −CG (only changing the loss function) immediately boosts recall@5 from 0.34 to 0.52.
3. **File Path is a Strong Signal**: CoRet learns to leverage file path information through training; removing paths leads to a drop in recall@5 from 0.53 to 0.42 (BM25 and CodeSage are unaffected because they fail to capture this signal).
4. **Call Graphs are More Useful for Multi-file Edits**: On LCA (where multi-file edits are more common), the improvement brought by call graphs (+6pt @20) is more pronounced than on SWE-bench (+2pt @20).
5. **In-instance Negative Samples Outperform Cross-instance Negatives**: Because the core of retrieval is resolving differences within the same repository, negative samples from other repositories provide weaker learning signals.

## Highlights & Insights

- **Clear and Practical Problem Definition**: Explicitly distinguishes code edit retrieval from traditional code search, highlighting their different semantic alignment goals (issue $\to$ edit location vs query $\to$ code function).
- **Simple yet Powerful Loss Function Design**: Switching from contrastive learning to log-likelihood represents a paradigm shift from "learning representation" to "learning retrieval," yielding significant gains with minimal modification.
- **Excellent Ablation Study Design**: Each design choice is coupled with its respective ablation—call graph vs same-file random chunk, presence of file path, and type/number of negative samples—proving highly convincing.
- **Lightweight yet Highly Efficient**: Based on CodeSage Small, achieving state-of-the-art results through tailored training strategies rather than expanding resource scale.

## Limitations & Future Work

1. **Limited to Python**: The chunking and call graph extraction strategies are designed for Python. Supporting other languages requires additional engineering (though SWE-PolyBench has been released).
2. **SWE-bench is Dominated by Single-file Edits**: File-level recall can be exceptionally high, making LCA multi-file editing scenarios a more challenging and suitable benchmark.
3. **Encoder-only Architectures**: Modern LLMs can be modified to output embeddings (e.g., LLM2Vec), which might provide better baselines but incur higher training costs.
4. **Call Graph Neighbor Selection Strategy**: Only downstream neighbors are used. Leveraging further graph topological properties (such as centrality or community structures) could yield additional gains.
5. **Lack of End-to-end Evaluation on Code Editing**: The work only assesses retrieval performance without evaluating how retrieval outcomes affect downstream code editing tools (e.g., pass@k on SWE-bench).

## Related Work & Insights

- **CodeSage** (Zhang et al., 2024): The strongest baseline and source of CoRet's backbone.
- **RepoFusion** (Shrivastava et al., 2023): Pioneer of repository-level code understanding.
- **SWE-bench** (Jimenez et al., 2024): Defined evaluation standards for repository-level code editing.
- **The Role of Call Graphs in Code Understanding** (Bansal et al., 2023): Provided theoretical foundations for CoRet's call graph integration.
- **Agentless** (Xia et al., 2024) and **SWE-agent** (Yang et al., 2024): Code-editing agents whose performance is limited by retrieval quality.

## Rating

- **Novelty**: ★★★☆☆ — Individual components (log-likelihood loss, call graph context, file path) are not entirely novel, but their combined application to code edit retrieval is valuable.
- **Experimental Thoroughness**: ★★★★☆ — Extensive ablation studies are conducted, but only two datasets are evaluated, and both are predominantly Python-based.
- **Writing Quality**: ★★★★☆ — Clear problem definition, well-explained motivation, and rigorous methodology description.
- **Value**: ★★★★☆ — Provides a practical retrieval-augmented infrastructure for code editing tasks like SWE-bench.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] OASIS: Order-Augmented Strategy for Improved Code Search](oasis_order-augmented_strategy_for_improved_code_search.md)
- [\[ACL 2025\] GALLa: Graph Aligned Large Language Models for Improved Source Code Understanding](galla_graph_aligned_large_language_models.md)
- [\[ICML 2026\] Pull Requests as a Training Signal for Repo-Level Code Editing](../../ICML2026/code_intelligence/pull_requests_as_a_training_signal_for_repo-level_code_editing.md)
- [\[ACL 2026\] To Diff or Not to Diff? Structure-Aware and Adaptive Output Formats for Efficient LLM-based Code Editing](../../ACL2026/code_intelligence/to_diff_or_not_to_diff_structure-aware_and_adaptive_output_formats_for_efficient.md)
- [\[ACL 2025\] DynaCode: A Dynamic Complexity-Aware Code Benchmark for Evaluating Large Language Models in Code Generation](dynacode_a_dynamic_complexity-aware_code_benchmark_for_evaluating_large_language.md)

</div>

<!-- RELATED:END -->
