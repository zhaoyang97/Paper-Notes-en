---
title: >-
  [Paper Note] MetaMoE: Diversity-Aware Proxy Selection for Privacy-Preserving Mixture-of-Experts Unification
description: >-
  [ICML 2026][AI Safety][MoE Unification] Domain experts independently fine-tuned by multiple clients on private data are merged into a single deployable MoE model without sharing private data. The core innovation uses rel…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "MoE Unification"
  - "Privacy Preservation"
  - "DPP Diversity"
  - "Proxy Data"
  - "Routing Training"
date: 2026-05-08
content_hash: db7e298b4b984fe2
---

# MetaMoE: Diversity-Aware Proxy Selection for Privacy-Preserving Mixture-of-Experts Unification

**Conference**: ICML 2026  
**arXiv**: [2605.14289](https://arxiv.org/abs/2605.14289)  
**Code**: [GitHub](https://github.com/ws-jiang/MetaMoE)  
**Area**: Privacy-Preserving Learning / Mixture-of-Experts / Model Merging  
**Keywords**: MoE Unification, Privacy Preservation, DPP Diversity, Proxy Data, Routing Training  

## TL;DR
Domain experts independently fine-tuned by multiple clients on private data are merged into a single deployable MoE model without sharing private data. The core innovation uses relevance-weighted DPP to select "both relevant and diverse" proxy samples from public data, followed by proxy-aligned expert training and a context-aware router. This aligns expert behavior with proxy supervision, significantly outperforming methods like FlexOlmo that rely solely on similarity-based proxy selection.

## Background & Motivation

**Background**: In the era of foundation models, different organizations/users often fine-tune domain experts on their respective private data. Model merging methods such as Branch-Train-Merge (BTM), Model Soup, and Branch-Train-MiX (BTX) attempt to fuse these experts into a single deployable model using a Mixture-of-Experts architecture and a router.

**Limitations of Prior Work**: (1) BTM outputs an ensemble rather than a unified model, complicating downstream SFT/RLHF; (2) Model Soup directly averages weights, which leads to performance collapse when experts are highly diverse; (3) BTX requires client private data to train the router, violating privacy constraints; (4) FlexOlmo uses public proxy samples to train the router, but proxies are selected only by similarity, resulting in highly redundant proxies with narrow coverage and misalignment between routing and expert behavior since experts never saw the proxies during training.

**Key Challenge**: Training a router necessitates data representative of each client's domain, yet real client data cannot leave the local environment. Proxy data must simultaneously satisfy "relevance to the client domain" and "coverage of diverse patterns within that domain," which corresponds precisely to the relevance + diversity logic of DPP.

**Goal**: (1) Provide a formal definition of the "Privacy-Preserving MoE Unification" problem; (2) Propose a proxy selection algorithm controlled by both relevance and diversity; (3) Enable experts to see their proxies during training to align with the router's training distribution; (4) Design a router capable of utilizing both token and sequence-scale context; (5) Provide a formal privacy analysis.

**Key Insight**: Selecting proxies by similarity only focuses on "how much a sample resembles the private domain," leading to the selection of redundant samples. DPP naturally generates "negative correlation" through the $\det$ term, avoiding the co-selection of similar samples. By embedding client-specific relevance into the DPP kernel, one can obtain "relevant + diverse" samples simultaneously.

**Core Idea**: Multiply client-specific relevance into the DPP kernel to form a relevance-weighted DPP $\tilde{L}_{ij} = g(x_i, \mathcal{D}_p) \kappa(x_i, x_j) g(x_j, \mathcal{D}_p)$; select $m$ proxies via greedy MAP; allow experts to fine-tune on $\mathcal{D}_p \cup \hat{\mathcal{D}}_p$; and finally train a context-aware router to merge all FFNs into an MoE.

## Method

### Overall Architecture
Input: Seed model $\mathcal{M}_0$, public data $\mathcal{D}_0$, and $K$ clients with private data $\{\mathcal{D}_p\}_{p=1}^K$. Each client first fine-tunes the seed model on local data to obtain expert $\mathcal{M}_p$. The unification phase consists of three steps: (1) Select client-specific proxy sets $\hat{\mathcal{D}}_p$ from $\mathcal{D}_0$ using relevance-weighted DPP; (2) Fine-tune each expert's FFN sublayers on $\mathcal{D}_p \cup \hat{\mathcal{D}}_p$ while freezing other parameters, and compute routing vectors $e_p^{(\ell)}$ as "domain mean representations" for each layer; (3) Merge all expert FFNs into MoE layers and jointly fine-tune a context-aware router on $\bigcup_p \hat{\mathcal{D}}_p$ to obtain the final unified model $\mathcal{M}_\text{MoE}$.

### Key Designs

1. **Relevance-Weighted DPP Proxy Selection**:
    - **Function**: Selects $m$ proxy samples from the public pool for each client that are both relevant to the client domain and diverse among themselves, providing proxy supervision for the router.
    - **Mechanism**: Train a binary classifier $g(x, \mathcal{D}_p)$ on the public pool to distinguish $\mathcal{D}_0$ from $\mathcal{D}_p$ (the score is the relevance); construct the kernel $\tilde{L} = \text{Diag}(r) L \text{Diag}(r)$, where $L_{ij} = \kappa(x_i, x_j)$. The subset selection objective is $\hat{\mathcal{D}}_p = \arg\max_{|S|=m} \log \det(\tilde{L}_S)$, which expands to $2 \sum_{i \in S} \log r_i + \log \det(L_S)$—the first term favors high relevance, and the second favors diversity. Complexity is reduced from $O(nm^3)$ to $O(nm)$ using a top-$n$ candidate pool and greedy MAP with Cholesky updates.
    - **Design Motivation**: Compared to FlexOlmo's similarity-only sorting which results in "relevant but redundant" proxies (clustered in t-SNE plots), the DPP $\det$ term penalizes co-occurrence of similar samples, spreading proxies across the private domain manifold to cover wider routing decision boundaries.

2. **Proxy-Aligned Expert Training**:
    - **Function**: Exposes the model to both private data and corresponding proxies during the expert phase, aligning the expert's output distribution with the proxy distribution encountered by the router.
    - **Mechanism**: Each client fine-tunes only its expert's FFN sublayers using $\mathcal{D}_p \cup \hat{\mathcal{D}}_p$ (instead of just $\mathcal{D}_p$); other layers remain frozen to maintain compatibility with $\mathcal{M}_0$ for MoE assembly. After training, routing representations are computed as $e_p^{(\ell)} = \tfrac{1}{|\mathcal{D}_p \cup \hat{\mathcal{D}}_p|} \sum_x \mathcal{M}_p^{(1:\ell)}(x)$.
    - **Design Motivation**: FlexOlmo trains experts only on private data and the router only on proxies, causing a mismatch between "expert behavior distribution" and "router input distribution"—especially when domain gaps are large. Allowing experts to see proxies aligns these distributions at the source without compromising privacy (proxies are public).

3. **Context-Aware Router + Domain-Aware Initialization**:
    - **Function**: The router considers both token representations and sequence-level representations to avoid routing collisions caused by "lexically similar tokens belonging to different domains."
    - **Mechanism**: Token representations $z_t^{(\ell)}$ and sequence means $z_x^{(\ell)} = \tfrac{1}{T} \sum_t z_t^{(\ell)}$ are combined via a convex sum $\tilde{z}_t^{(\ell)} = (1 - \lambda) z_t^{(\ell)} + \lambda z_x^{(\ell)}$, where $\lambda$ is learnable; the routing distribution is $\pi^{(\ell)}(z_t^{(\ell)}) = \text{softmax}[\tilde{z}_t^{(\ell) \top} e_1^{(\ell)}, \dots, \tilde{z}_t^{(\ell) \top} e_K^{(\ell)}]$. Routing vectors $e_p^{(\ell)}$ are initialized with the "expert domain means" from step (2).
    - **Design Motivation**: Pure token-level routing is easily fooled by literal similarity (e.g., "bank" in finance vs. riverbank). Incorporating whole-sentence context and initializing with expert domain means provides the router with a strong prior regarding each expert's strengths.

### Loss & Training
The expert phase uses standard next-token/classification loss; the router phase jointly fine-tunes the entire MoE on $\bigcup_p \hat{\mathcal{D}}_p$. Clients perform a one-time upload to the server: (i) proxy sample indices; (ii) final expert weights (FFN sublayers); (iii) routing vectors $e_p^{(\ell)}$. Formal analysis proves these artifacts do not leak private information (routing vectors are $N \to \infty$ mean embeddings, where leakage decays with $N$).

## Key Experimental Results

### Main Results
Evaluated on CV (Pets, Cars, CIFAR-100 based on ViT-B/32) and NLP (multi-task benchmarks based on LLMs), MetaMoE is compared against BTM, Model Soup, BTX, and FlexOlmo. Figure 2 in the paper visualizes proxy selection using t-SNE on the Pets dataset: MetaMoE's proxies cover a significantly broader portion of the private domain manifold compared to random or FlexOlmo strategies.

| Method | CV Avg Acc | NLP Avg Acc | Privacy Level | Single Deployable Model |
|------|-------------|--------------|----------|----------------|
| BTM (ensemble) | High | High | Strong | No (multi-expert inference) |
| Model Soup | Weak (heterogeneous) | Weak | Strong | Yes |
| BTX | High | High | Weak (requires private data for router) | Yes |
| FlexOlmo (similarity-only) | Mid-High | Mid-High | Strong | Yes |
| **MetaMoE** | **Highest** | **Highest** | Strong | Yes |

(The main text and appendix show full results; the abstract explicitly states that MetaMoE consistently outperforms the latest baselines on both CV and NLP benchmarks.)

### Ablation Study

| Configuration | Performance |
|------|------|
| Full MetaMoE | Optimal |
| Remove diversity (degrades to similarity-only) | significant drop in accuracy, proxy clustering |
| Remove proxy-aligned training (private data only) | Mismatch between router and expert behavior, higher routing error |
| Remove context-aware blending (pure token routing) | Lexically similar tokens misrouted to wrong experts |
| Remove domain-aware initialization (random init) | Slower convergence, lower final precision |

### Key Findings
- t-SNE visualizations clearly show FlexOlmo's proxies clustered together (narrow coverage), while MetaMoE's proxies spread across the private domain manifold—indicating that "relevance + diversity" is necessary for the router to learn effectively.
- Gains from proxy-aligned expert training are relatively independent of router design, suggesting that "letting experts see proxies" is a critical architectural change. It yields first-order benefits even with a simple router.
- Uploaded artifacts consist only of "indices + weights + mean embeddings," exposing less private information than the per-round gradient uploads in federated learning; formal proof shows privacy leakage decays at $O(1/N)$ as $N$ increases.
- Proxy selection occurs only once (no client polling), resulting in communication complexity an order of magnitude lower than FL.

## Highlights & Insights
- Integrating DPP with client-specific relevance is a natural yet previously unexplored innovation; a few formulas upgrade the quality of router supervision from "relevant" to "relevant + diverse."
- "Proxy-aligned expert training" breaks the traditional isolation between "private experts" and "proxy routers"—treating proxies as expert training data eliminates routing-expert mismatch. This strategy can be transferred to any cross-domain merging tasks (multilingual LMs, multi-modal adapters).
- Initializing routing vectors with expert domain means explicitly tells the router "what each expert is," bypassing the need to search for directions via pure gradients, which is highly beneficial for data-scarce scenarios.
- The privacy analysis provides a specific upper bound for mean embedding leakage of $O(1/N)$, offering a template for broader applications of mean-pooled embeddings in privacy preservation.

## Limitations & Future Work
- The relevance classifier $g(\cdot, \mathcal{D}_p)$ must be trained on $\mathcal{D}_0 \cup \mathcal{D}_p$, which might leak some statistical information about $\mathcal{D}_p$ (classified as classifier output on public data, but strictly speaking, still a private signal).
- DPP uses $O(nm)$ greedy approximation rather than global optimization; the candidate pool limit $n$ is a hyperparameter. When $\mathcal{D}_0$ is much smaller than the private domain, proxies might still fail to cover the domain.
- Experiments are limited to FFN layers of ViT and LLMs; the effectiveness for attention or cross-modal experts remains unverified.
- $\lambda$ is a single scalar in the context-aware router, which may not be optimal across different layers of a transformer—different layers might require different token/sequence balances.

## Related Work & Insights
- **vs. BTM / Model Soup / BTX**: BTM does not output a single model; Model Soup is fragile with heterogeneous experts; BTX requires private data for the router; MetaMoE provides a single model using only public proxies, outperforming all three.
- **vs. FlexOlmo**: FlexOlmo also uses public proxies, but they are selected only by similarity, and experts never see the proxies; MetaMoE upgrades this with DPP diversity, proxy-aligned training, and domain-aware initialization.
- **vs. Federated Learning**: FL requires multiple rounds of gradient exchange and is susceptible to model inversion attacks; Ours uses a one-time upload of expert weights, indices, and mean embeddings, involving less communication and a smaller attack surface.
- **vs. MoE Routing Methods (Switch Transformer, top-k gating)**: While the router is formally top-k softmax, the domain-aware initialization and sequence-blended context adapt routing to the specific scenario of "heterogeneous expert distribution + proxy supervision."

## Rating
- Novelty: ⭐⭐⭐⭐ Systematically combining DPP diversity, relevance weighting, and proxy-aligned training for privacy-preserving MoE is a first.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers both CV and NLP benchmarks, multiple baselines, visualization, and ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Algorithm 1 and privacy analysis are logically sound; formulas and diagrams are clear.
- Value: ⭐⭐⭐⭐ Provides a complete and reproducible pipeline for deploying privacy-sensitive industrial MoE models, accompanied by formal privacy guarantees.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] DP-KFC: Data-Free Preconditioning for Privacy-Preserving Deep Learning](dp-kfc_data-free_preconditioning_for_privacy-preserving_deep_learning.md)
- [\[ICCV 2025\] FedVLA: Federated Vision-Language-Action Learning with Dual Gating Mixture-of-Experts for Robotic Manipulation](../../ICCV2025/ai_safety/fedvla_federated_vision-language-action_learning_with_dual_gating_mixture-of-exp.md)
- [\[CVPR 2026\] FecalFed: Privacy-Preserving Poultry Disease Detection via Federated Learning](../../CVPR2026/ai_safety/fecalfed_privacy-preserving_poultry_disease_detection_via_federated_learning.md)
- [\[ICLR 2026\] Membership Privacy Risks of Sharpness Aware Minimization](../../ICLR2026/ai_safety/sam_membership_privacy_risks.md)
- [\[ICCV 2025\] FedMeNF: Privacy-Preserving Federated Meta-Learning for Neural Fields](../../ICCV2025/ai_safety/fedmenf_privacy-preserving_federated_meta-learning_for_neural_fields.md)

</div>

<!-- RELATED:END -->
