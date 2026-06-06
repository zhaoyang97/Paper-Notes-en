---
title: >-
  [Paper Note] MetaMoE: Diversity-Aware Proxy Selection for Privacy-Preserving Mixture-of-Experts Unification
description: >-
  [ICML 2026][AI Safety][MoE Unification] Multiple client-specific experts, each fine-tuned on private data, can be merged into a deployable MoE model without sharing private data. The core is to select "relevant and diver…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "MoE Unification"
  - "Privacy Preservation"
  - "DPP Diversity"
  - "Proxy Data"
  - "Router Training"
date: 2026-05-08
content_hash: 5ccb22aa7a2b0488
---

# MetaMoE: Diversity-Aware Proxy Selection for Privacy-Preserving Mixture-of-Experts Unification

**Conference**: ICML 2026  
**arXiv**: [2605.14289](https://arxiv.org/abs/2605.14289)  
**Code**: [GitHub](https://github.com/ws-jiang/MetaMoE)  
**Area**: Privacy-Preserving Learning / Mixture-of-Experts / Model Merging  
**Keywords**: MoE Unification, Privacy Preservation, DPP Diversity, Proxy Data, Router Training

## TL;DR
Multiple client-specific experts, each fine-tuned on private data, can be merged into a deployable MoE model without sharing private data. The core is to select "relevant and diverse" proxy samples from public data using relevance-weighted DPP, enabling proxy-aligned expert training followed by context-aware router training. This aligns expert behaviors with proxy supervision and significantly outperforms similarity-only proxy selection methods like FlexOlmo.

## Background & Motivation

**Background**: In the era of foundation models, different organizations/users often fine-tune domain experts on their own private data. Model merging methods such as Branch-Train-Merge (BTM), Model Soup, and Branch-Train-MiX (BTX) attempt to fuse these experts into a deployable model, leveraging Mixture-of-Experts architectures and routers.

**Limitations of Prior Work**: (1) BTM outputs an ensemble, lacking a unified model, which hinders downstream SFT/RLHF; (2) Model Soup averages weights directly, leading to performance collapse when experts are heterogeneous; (3) BTX requires private data for router training, violating privacy constraints; (4) FlexOlmo trains the router using public proxy samples, but proxies are selected solely by similarity, resulting in redundancy and narrow coverage. Experts, having only seen private data, are misaligned with proxies, causing routing-expert behavior mismatch.

**Key Challenge**: Router training requires data representative of each client domain, but real client data cannot be shared. Proxy data must be both "relevant to the client domain" and "cover diverse modes of that domain," which aligns with DPP's relevance + diversity logic.

**Goal**: (1) Formally define the "privacy-preserving MoE unification" problem; (2) Propose a proxy selection algorithm controlling both relevance and diversity; (3) Ensure experts see their proxies during training to align with router training distribution; (4) Design a router leveraging both token- and sequence-level context; (5) Provide formal privacy analysis.

**Key Insight**: Similarity-based proxy selection only considers "how much a sample resembles the private domain," often repeatedly selecting similar samples. DPP's $\det$ term naturally induces "negative correlation," avoiding co-selection of similar samples. Embedding client-specific relevance into the DPP kernel yields both "relevance + diversity."

**Core Idea**: Multiply client-specific relevance into the DPP kernel to form a relevance-weighted DPP $\tilde{L}_{ij} = g(x_i, \mathcal{D}_p) \kappa(x_i, x_j) g(x_j, \mathcal{D}_p)$. Use greedy MAP to select $m$ proxies. Experts are then fine-tuned on $\mathcal{D}_p \cup \hat{\mathcal{D}}_p$, followed by context-aware router training to merge all FFNs into an MoE.

## Method

### Overall Architecture
Input: seed model $\mathcal{M}_0$, public data $\mathcal{D}_0$, $K$ clients with private data $\{\mathcal{D}_p\}_{p=1}^K$. Each client fine-tunes the seed model locally to obtain expert $\mathcal{M}_p$. The unification proceeds in three steps: (1) Use relevance-weighted DPP to select client-specific proxy set $\hat{\mathcal{D}}_p$ from $\mathcal{D}_0$; (2) Fine-tune each expert's FFN sublayer on $\mathcal{D}_p \cup \hat{\mathcal{D}}_p$ (other parameters frozen), and compute routing vector $e_p^{(\ell)}$ per layer as the "domain mean representation"; (3) Merge all experts' FFNs into MoE layers, and jointly fine-tune a context-aware router on $\bigcup_p \hat{\mathcal{D}}_p$ to obtain the unified MoE model $\mathcal{M}_\text{MoE}$.

### Key Designs

1. **Relevance-Weighted DPP Proxy Selection**:

    - **Function**: For each client, select $m$ "domain-relevant and mutually diverse" proxy samples from the public pool to supervise the router.
    - **Mechanism**: Train a binary classifier $g(x, \mathcal{D}_p)$ on the public pool to distinguish $\mathcal{D}_0$ from $\mathcal{D}_p$ (score as relevance); construct kernel $\tilde{L} = \text{Diag}(r) L \text{Diag}(r)$, where $L_{ij} = \kappa(x_i, x_j)$. Subset selection is $\hat{\mathcal{D}}_p = \arg\max_{|S|=m} \log \det(\tilde{L}_S)$, which expands to $2 \sum_{i \in S} \log r_i + \log \det(L_S)$—the first term favors high relevance, the second favors diversity. Top-$n$ candidates are selected by $r$, then greedy MAP + Cholesky incremental update reduces complexity from $O(nm^3)$ to $O(nm)$.
    - **Design Motivation**: Compared to FlexOlmo, which selects "relevant but redundant" proxies (clustered in t-SNE), DPP's $\det$ term penalizes co-selection of similar samples, spreading proxies across the private domain manifold and covering broader routing decision boundaries.

2. **Proxy-Aligned Expert Training**:

    - **Function**: Ensure experts see both private and proxy data during training, aligning expert output distributions with those seen by the router.
    - **Mechanism**: Each client fine-tunes only its expert's FFN sublayer on $\mathcal{D}_p \cup \hat{\mathcal{D}}_p$ (not just $\mathcal{D}_p$); other layers are frozen to maintain compatibility with the seed model $\mathcal{M}_0$, facilitating later MoE merging. After training, compute routing representation per layer $e_p^{(\ell)} = \tfrac{1}{|\mathcal{D}_p \cup \hat{\mathcal{D}}_p|} \sum_x \mathcal{M}_p^{(1:\ell)}(x)$.
    - **Design Motivation**: FlexOlmo trains experts only on private data and routers on proxies, causing "expert behavior distribution" and "router input distribution" mismatch—especially problematic when client domains differ. Having experts see proxies eliminates this mismatch at the source, without compromising privacy (proxies are public).

3. **Context-Aware Router + Domain-Aware Initialization**:

    - **Function**: Router considers both token and sequence-level representations, avoiding routing collisions caused by superficially similar tokens from different domains.
    - **Mechanism**: For each token representation $z_t^{(\ell)}$ and sequence mean $z_x^{(\ell)} = \tfrac{1}{T} \sum_t z_t^{(\ell)}$, form a convex combination $\tilde{z}_t^{(\ell)} = (1 - \lambda) z_t^{(\ell)} + \lambda z_x^{(\ell)}$, with learnable $\lambda$; routing distribution $\pi^{(\ell)}(z_t^{(\ell)}) = \text{softmax}[\tilde{z}_t^{(\ell) \top} e_1^{(\ell)}, \dots, \tilde{z}_t^{(\ell) \top} e_K^{(\ell)}]$. Routing vectors $e_p^{(\ell)}$ are initialized with the "expert domain mean" from step (2), injecting domain priors.
    - **Design Motivation**: Pure token-level routing is easily misled by surface similarity (e.g., "bank" as finance or riverbank); adding sentence context and initializing routing vectors with expert domain means gives the router a strong prior on each expert's specialty.

### Loss & Training
Expert stage uses standard next-token/classification loss; router stage jointly fine-tunes the entire MoE on $\bigcup_p \hat{\mathcal{D}}_p$. Each client uploads only once to the server: (i) proxy sample indices (public data indices); (ii) final expert weights (FFN sublayer); (iii) routing vectors $e_p^{(\ell)}$. The paper provides formal analysis showing these artifacts do not leak private information (routing vectors are mean embeddings over $N \to \infty$ samples, with privacy leakage decaying as $N$ increases).

## Key Experimental Results

### Main Results
Benchmarks on CV (ViT-B/32-based Pets, Cars, CIFAR-100, etc.) and NLP (LLM-based multi-task benchmarks) compare BTM, Model Soup, BTX, FlexOlmo, etc. Figure 2 visualizes, via t-SNE on the Pets dataset, the proxy selection strategies of random, FlexOlmo, and MetaMoE: MetaMoE's proxies cover a much broader private domain manifold.

| Method | CV Avg. Acc | NLP Avg. Acc | Privacy Level | Single Deployable Model |
|--------|-------------|--------------|--------------|------------------------|
| BTM (ensemble) | High | High | Strong | No (multi-expert inference) |
| Model Soup | Weak (when experts are heterogeneous) | Weak | Strong | Yes |
| BTX | High | High | Weak (needs private data for router) | Yes |
| FlexOlmo (similarity-only proxy) | Medium-High | Medium-High | Strong | Yes |
| **MetaMoE** | **Highest** | **Highest** | Strong | Yes |

(The main text and appendix provide full results; the abstract states MetaMoE consistently outperforms latest baselines on both CV and NLP benchmarks.)

### Ablation Study

| Configuration | Effect |
|---------------|--------|
| Full MetaMoE | Optimal |
| Remove diversity (degrades to FlexOlmo-style relevance-only) | Accuracy drops, proxies cluster |
| Remove proxy-aligned expert training (experts see only private data) | Router-expert mismatch, routing error increases |
| Remove context-aware blending (token-only routing) | Surface-similar tokens misrouted |
| Remove routing vector domain-aware initialization (random init) | Slower convergence, lower final accuracy |

### Key Findings
- t-SNE visualizations show FlexOlmo's proxies cluster tightly (narrow coverage), while MetaMoE's proxies spread across the private domain manifold—demonstrating that "relevance + diversity" is necessary for effective router learning, not just "relevance."
- The improvement from proxy-aligned expert training is largely independent of router design, indicating that "having experts see proxies" is itself a key change; even with FlexOlmo's simple router, this yields significant gains.
- Uploaded artifacts are only "indices + weights + mean embeddings," exposing less private information than federated learning's per-round gradient uploads; formal analysis shows privacy leakage decays as $O(1/N)$ with increasing $N$.
- Proxy selection is one-off (no client polling), reducing communication complexity by an order of magnitude compared to FL.

## Highlights & Insights
- Integrating DPP with client-specific relevance is a natural but previously unexplored innovation; a few formulas upgrade router supervision from "relevant" to "relevant + diverse."
- "Proxy-aligned expert training" breaks the traditional separation of "experts on private / router on proxy"—treating proxies as expert training data eliminates routing-expert mismatch, and this idea can transfer to any cross-domain merging task (e.g., multilingual LMs, multimodal adaptation).
- Initializing routing vectors with expert domain mean embeddings directly informs the router of each expert's specialty, avoiding reliance on pure gradient search and benefiting low-data scenarios.
- Privacy analysis provides a concrete upper bound $O(1/N)$ for mean embedding leakage, offering a template for privacy protection in mean-pooled embedding applications.

## Limitations & Future Work
- The relevance classifier $g(\cdot, \mathcal{D}_p)$ must be trained on $\mathcal{D}_0 \cup \mathcal{D}_p$, potentially leaking some statistics of $\mathcal{D}_p$ (the paper treats this as "classifier outputs on public data," but strictly speaking, it is still a private signal).
- DPP uses $O(nm)$ greedy approximation rather than global optimum; candidate pool size $n$ is a hyperparameter. If $\mathcal{D}_0$ is much smaller than the true private domain, proxies may still lack coverage.
- Experiments are limited to FFN layers in ViT and LLMs; effectiveness for attention/cross-modal experts is unverified.
- $\lambda$ in the context-aware router is a single scalar, which may not be optimal for multi-layer transformers—different layers may require different token/sequence balances.

## Related Work & Insights
- **vs BTM / Model Soup / BTX**: BTM does not output a single model; Model Soup is fragile with heterogeneous experts; BTX requires private data for router training. MetaMoE provides a single model using only public proxies, outperforming all three.
- **vs FlexOlmo**: FlexOlmo also uses public proxies, but selects them by similarity only and experts do not see proxies. MetaMoE adds DPP-based diversity, proxy-aligned training, and domain-aware router initialization for a comprehensive upgrade.
- **vs Federated Learning**: FL requires multiple rounds of gradient exchange and is vulnerable to model inversion attacks. This work uploads expert weights + indices + mean embeddings once, reducing communication and attack surface.
- **vs MoE Routing Methods (Switch Transformer, top-k gating)**: The router is still top-k softmax in form, but domain-aware initialization and sequence-blended context adapt routing to the "heterogeneous experts + proxy-only supervision" scenario.

## Rating
- Novelty: ⭐⭐⭐⭐ Systematic combination of DPP diversity, relevance weighting, and proxy-aligned training for privacy-preserving MoE is a first.
- Experimental Thoroughness: ⭐⭐⭐⭐ Benchmarks on both CV and NLP, multiple baselines, visualization, and ablation.
- Writing Quality: ⭐⭐⭐⭐ Algorithm 1 and privacy analysis are logically clear, with clear formulas and illustrations.
- Value: ⭐⭐⭐⭐ Provides a complete, reproducible pipeline for privacy-sensitive industrial MoE deployment, with formal privacy guarantees.

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
