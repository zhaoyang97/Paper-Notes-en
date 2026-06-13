---
title: >-
  [Paper Note] Fixed Aggregation Features Can Rival GNNs
description: >-
  [ICML2026][Graph Learning][Fixed aggregation] This paper proposes Fixed Aggregation Features (FAF): compressing multi-hop neighborhoods into tabular features using **non-trainable** aggregation operators (such as mean…
tags:
  - "ICML2026"
  - "Graph Learning"
  - "Fixed aggregation"
  - "Multi-hop feature concatenation"
  - "Kolmogorov-Arnold"
  - "MLP baseline"
  - "Node classification"
date: 2026-05-08
content_hash: c7f5e342e822bed2
---

# Fixed Aggregation Features Can Rival GNNs

**Conference**: ICML2026  
**arXiv**: [2601.19449](https://arxiv.org/abs/2601.19449)  
**Code**: Undisclosed  
**Area**: Graph Learning / GNN / Tabular Learning  
**Keywords**: Fixed aggregation, Multi-hop feature concatenation, Kolmogorov-Arnold, MLP baseline, Node classification

## TL;DR
This paper proposes Fixed Aggregation Features (FAF): compressing multi-hop neighborhoods into tabular features using **non-trainable** aggregation operators (such as mean, sum, max, min, and std) and feeding them into an MLP. In 12 out of 14 node classification benchmarks, FAF matches or surpasses fine-tuned GCN, GAT, GraphSAGE, and even Graph Transformers, thereby systematically questioning the necessity of trainable neighborhood aggregation in GNNs.

## Background & Motivation

**Background**: Node classification is predominantly occupied by message-passing GNNs. From GCN/GAT/GraphSAGE to Graph Transformers and heterophily-specific models, the mainstream narrative posits that "each layer must learn neighborhood aggregation and a linear transformation." The resulting high complexity is tacitly accepted as a necessary cost for expressive power.

**Limitations of Prior Work**: (1) Luo et al. 2024 observed that with hyperparameter fine-tuning, classic GNNs nearly match SOTA Graph Transformers, suggesting that the added complexity of newer models does not yield substantial gains; (2) Attention mechanisms suffer from known learnability issues (vanishing gradients, inability to "silence" neighbors); (3) GNNs generally overfit quickly on training sets, with validation optima often appearing before the aggregation is fully learned.

**Key Challenge**: Expressiveness and learnability have long been conflated. Theoretically, GNNs can learn info-preserving aggregations, but does actual optimization achieve this? Or is a **fixed, or even non-info-preserving aggregation** already sufficient?

**Goal**: Use a minimalist, baseline-like approach to probe two sub-questions: (a) How well can models perform without learning aggregation? (b) Do existing benchmarks truly "require" learned aggregation?

**Key Insight**: Starting from the Kolmogorov-Arnold representation theorem, the authors prove that there exists a **fixed, univariate, lossless** neighborhood aggregation $\Phi$ such that any multiset function can be written as $f = g \circ \Phi^{-1}$, reducing the task from "learning aggregation + learning classifier" to just learning $g$. However, $\Phi$ is discontinuous and numerically brittle, producing "rough" embeddings difficult for MLPs to process. Thus, they step back to simple reducers (mean/sum/max/min/std)—which, while not invertible, are empirically easier to optimize.

**Core Idea**: Replace the "trainable multi-layer aggregation" of GNNs with a "multi-hop fixed aggregation + concatenation + tabular MLP" in the preprocessing stage. This reduces the graph learning problem to a tabular learning problem, gaining benefits in interpretability, tuneability, and efficiency.

## Method

### Overall Architecture

The FAF pipeline is extremely simple and consists of two steps:

1.  **Offline Preprocessing (Parameter-free)**: Given a graph $G=(V,E)$ and node features $x_v$, for each node $v$, each reducer $r \in \mathcal{R} \subset \{\text{mean}, \text{sum}, \text{max}, \text{min}, \text{std}, \ldots\}$, and each hop $k \in \{1, \ldots, K\}$, recursively compute $h_v^{(0,r)} = x_v$ and $h_v^{(k,r)} = r(\{h_u^{(k-1,r)} : u \in N(v)\})$. Finally, concatenate the results from all hops and all reducers with the original features: $z_v = x_v \oplus \bigoplus_{r \in \mathcal{R}} \bigoplus_{k=1}^{K} h_v^{(k,r)}$, resulting in tabular features of dimension $|x_v| \cdot (1 + |\mathcal{R}| \cdot K)$.

2.  **Online Training (MLP only)**: Treat $z_v$ as a standard tabular row and feed it into a **fine-tuned MLP** for node classification. There are **no learnable aggregation parameters** in the entire process; all "graph structure information" is baked into the feature vectors during preprocessing.

The input consists of the graph and node features, and the output is the node category; there are no message-passing layers, no attention, and no trainable propagation matrices.

### Key Designs

1.  **Multi-hop × Multi-reducer Concatenation (Instead of Replacement)**:
    - **Function**: **Concatenates all** reducer outputs from hops $0, 1, \ldots, K$ for the MLP, allowing the classifier to select the combination of hops and operators.
    - **Mechanism**: The authors prove (Thm 4.1) that when node features are orthogonal, 1-hop sum aggregation is **injective** and can losslessly represent any multiset function; mean is equivalent to sum when node degrees are known. However, after $k \geq 2$, orthogonality no longer holds, and a single reducer inevitably loses information. Different reducers capture different aspects of the distribution (sum counts, mean weights by degree, max/min focus on tails). Concatenation allows the MLP to act as a "soft feature selection" mechanism. Ablations (Tab 10/11) show that using only the last hop or a single linear layer significantly degrades performance, proving that concatenation + MLP are both essential.
    - **Design Motivation**: Directly addresses the "info loss in non-injective aggregation" problem—since a single reducer is not invertible, multiple complementary reducers are used to achieve "approximate injectivity" in the tabular space.

2.  **Theoretical Upper Bound via Kolmogorov-Arnold Construction**:
    - **Function**: Provides a **theoretically lossless** fixed aggregation $\Phi(x_1, \ldots, x_d) = 3 \sum_{p=1}^{d} 3^{-p} \phi(x_p)$ (based on ternary expansion of the Cantor set, following Schmidt-Hieber 2021) to define the upper bound of the FAF framework's expressive power.
    - **Mechanism**: $\Phi$ is an injection $[0,1]^d \to \mathbb{R}$, shifting the entire burden of "learning aggregation" to a learnable univariate $g$; any continuous $f$ can be written as $g \circ \Phi^{-1}$ and inherits the approximation rate of $f$. However, $\Phi$ is discontinuous and maps close inputs to distant values, making it hard for downstream MLPs to learn. On Roman-Empire, the KA version of FAF achieved a top score of $80.33$, but KA proved difficult to train on other datasets.
    - **Design Motivation**: Decouples "expressive power" from "actual learnability"—theoretical guarantees show FAF is not inferior to GNNs, while empirical results show simple reducers win because they are "easier to optimize" rather than "more expressive."

3.  **Triple Dividends from the Tabular Perspective**:
    - **Function**: Converting graph problems into tabular problems allows for the **direct application of the entire tabular learning toolbox**, including SHAP interpretability, feature importance, feature selection, handling class imbalance, and noise robustness.
    - **Mechanism**: Running SHAP on Minesweeper revealed the most important signal to be "hop-1 mean of feature 1" (i.e., the proportion of neighbors with a local bomb count of 0), perfectly matching the game's actual logic; similar interpretable attributions were found for Pubmed/Amazon-Ratings. Additionally, extra aggregations after graph rewiring (e.g., edge deletion based on cosine similarity) can be concatenated at the tabular level as comparative diagnostics.
    - **Design Motivation**: The end-to-end structure of GNNs makes it nearly impossible to attribute what each layer has learned. FAF decouples "features," "hops," and "reducer selection," allowing the graph dataset itself to be scrutinized independently—identifying which hops provide signals, which reducers are complementary, and which are redundant.

### Loss & Training
The downstream MLP uses standard CrossEntropy + dropout + LayerNorm. The hyperparameter grid follows Luo et al. 2024 to ensure direct comparability with Graph Transformers. Notably, FAF prefers **larger learning rates** (leading to faster convergence and implicit sparse regularization), while dropout levels are comparable to GNNs—suggesting that dropout gains stem from the dataset itself rather than graph convolution characteristics.

## Key Experimental Results

### Main Results

On 14 standard node classification benchmarks, FAF's best variant vs. classic GNNs (Selected from Table 1):

| Dataset | GCN | GAT | SAGE | FAF$_\text{bestval}$ | Conclusion |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Amazon-Computer | 93.58 | 93.91 | 93.31 | **94.01** | FAF slightly leads |
| Amazon-Photo | 95.77 | 96.45 | 96.17 | **96.54** | FAF slightly leads |
| Amazon-Ratings | 53.86 | 55.51 | 55.26 | 55.09 | Equal |
| Pubmed | 80.00 | 79.80 | 77.42 | **80.96** | FAF slightly leads |
| Questions | 78.44 | 77.72 | 76.75 | **78.69** | FAF slightly leads |
| WikiCS | 80.06 | 81.01 | 80.57 | 80.25 | Equal |
| Coauthor-CS | 95.73 | 96.14 | 96.21 | 95.37 | Slightly lower (~1%) |
| Cora | 84.38 | 83.02 | 83.18 | 82.84 | Slightly lower (~1.5%) |
| **Minesweeper** | 97.48 | 97.00 | 97.72 | **90.00** | **Trailing significantly** |
| **Roman-Empire** | 91.05 | 90.38 | 90.41 | **78.11** | **Trailing significantly** |

Overall performance: Surpassed GNNs on 5 datasets, matched on 5 (gap $\leq 1\%$), and trailed on 4, with Minesweeper/Roman-Empire showing the largest gaps.

### Ablation Study

| Configuration | Key Phenomenon | Explanation |
| :--- | :--- | :--- |
| FAF4 (mean+sum+max+min) | Best across most datasets | Default configuration |
| Single reducer (Tab 7) | mean wins most often | Citeseer prefers sum; Amazon-Ratings prefers max |
| Last hop only (Tab 11) | Significant drop | Confirms necessity of concatenating all hops |
| Linear classifier (Tab 10) | Significantly lower than MLP | Confirms MLP non-linearity is critical |
| KA Aggregation (Tab 12) | Hits peak 80.33 on Roman-Empire | Confirms trailing datasets suffer from info loss, not "must learn aggregation" |
| Increased hops (Tab 8) | Peaks at $k=2$ for most | Signals are concentrated in first two hops |

### Key Findings
- **The best FAF versions generally require only 2-4 hops**, whereas GNNs on Minesweeper/Roman-Empire require 10-15 layers. These specific datasets rely on the linear residual connections of GNNs (Luo et al. 2024); the gap between FAF and GNN matches the gap caused by adding residuals.
- **Mean alone frequently enters the top tier**, suggesting that neighborhood distribution information constitutes the bulk of the task signal, and degree information (implicit in the ratio of mean to sum) is sufficient to distinguish neighbor contributions.
- **Common GNN phenomena like over-smoothing, deep degradation, and dropout sensitivity** also appear in this purely tabular FAF setting—implying these "ailments" are not necessarily faults of message-passing itself but likely stem from dataset characteristics.
- KA aggregation outperformed all simple reducers on Roman-Empire, proving the gap there is due to reducer info loss rather than a "requirement for trainable aggregation."

## Highlights & Insights

- **Strong Counter-example of "SOTA without Trainable Aggregation"**: Challenges the implicit belief that "GNNs must learn aggregation" via a blunt baseline. The method is too simple to be considered a product of overfitting to benchmarks, making it highly persuasive.
- **Synthesized Theoretical and Experimental Evidence**: The Kolmogorov-Arnold construction shows that "theoretically FAF is enough," while simple reducers beating KA aggregation show that "actual learnability is more important than expressiveness."
- **Interpretability as a Free Lunch**: Using SHAP to see features used in Minesweeper decisions provides a granularity of attribution almost impossible in GNNs, offering a reusable tool to diagnose whether graph benchmarks truly contain structural signals.
- **Interrogating Domain-wide Benchmarking**: The authors explicitly call for future work to use "fine-tuned FAF" as a standard baseline. If a new model claiming to utilize complex graph structures cannot beat FAF, then "what it is winning is not the graph structure." This paradigm pressure is the paper's most enduring contribution.

## Limitations & Future Work

- The authors acknowledge that FAF lags on Minesweeper/Roman-Empire due to long-range dependencies and reducer info loss; these are among the few scenarios where GNNs are truly "irreplaceable."
- **Limitations**: When the number of reducers, hops, and original feature dimensions are all large, the concatenated input dimension explodes, increasing the MLP's first-layer parameters and training costs; feature selection is required. A systematic dimensionality reduction scheme was not provided.
- Experiments focused on node classification, without touching link prediction, graph classification, or dynamic graphs; FAF's universality remains unverified for these tasks.
- **Future Directions**: (a) Design aggregations that are both injective and "smooth" (intermediate between mean and $\Phi$); (b) Hybridize FAF with partially learnable aggregations, letting the model decide which hops/reducers to learn; (c) Apply this tabular diagnostic method to design new benchmarks that truly "require learned aggregation."

## Related Work & Insights

- **vs. SGC (Wu et al. 2019)**: SGC also removes non-linearity and fixes propagation but uses only a linear readout and linear diffusion. FAF introduces **non-linear** reducers (max/min/std) and uses an MLP, providing broader coverage and stronger expression.
- **vs. SIGN/GAMLP/HOGA**: These also precompute multi-hop features but still learn complex attention/gating for hop weights, targeting "scalability optimization." FAF treats the MLP as a soft feature selector and learns no propagation parameters, serving as a **baseline and diagnostic tool**.
- **vs. G2T-FM / TabPFN-GN / Hayler et al. 2025**: Parallel "Tabularization + Tabular Foundation Model" routes aimed at adapting Transformers to graphs. FAF's goal is the opposite—using minimalist tabular learning to detect if graph tasks actually need to learn aggregation.
- **vs. PNA (Corso et al. 2020)**: PNA also emphasizes multi-reducer combinations and degree scalers to enhance expressiveness but remains an end-to-end trainable GNN. FAF pushes the "multi-reducer" idea to the extreme—completely freezing the aggregation layers.

## Rating
- Novelty: ⭐⭐⭐⭐ The method itself is minimalist, but the perspective (KA explanation + tabular diagnosis) and systematic counter-example value are high.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 14 standard benchmarks + multiple ablations + SHAP interpretability + KA comparison, covering all bases.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear argumentation, theory and experiments support each other, and limitations are frankly admitted.
- Value: ⭐⭐⭐⭐⭐ Will exert long-term pressure on the GNN evaluation paradigm and will be widely cited as a standard baseline and benchmark design reference.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Logical Characterizations of GNNs with Mean Aggregation](../../AAAI2026/graph_learning/logical_characterizations_of_gnns_with_mean_aggregation.md)
- [\[AAAI 2026\] Enhancing Logical Expressiveness in GNNs via Path-Neighbor Aggregation](../../AAAI2026/graph_learning/enhancing_logical_expressiveness_in_graph_neural_networks_via_path-neighbor_aggr.md)
- [\[AAAI 2026\] Beyond Fixed Depth: Adaptive Graph Neural Networks for Node Classification Under Varying Homophily](../../AAAI2026/graph_learning/beyond_fixed_depth_adaptive_graph_neural_networks_for_node_classification_under_.md)
- [\[ICML 2026\] Identifying and Correcting Label Noise for Robust GNNs via Influence Contradiction](identifying_and_correcting_label_noise_for_robust_gnns_via_influence_contradicti.md)
- [\[ICLR 2026\] GRAPHITE: Graph Homophily Booster — Reimagining the Role of Discrete Features in Heterophilic Graph Learning](../../ICLR2026/graph_learning/graph_homophily_booster_reimagining_the_role_of_discrete_features_in_heterophili.md)

</div>

<!-- RELATED:END -->
