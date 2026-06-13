---
title: >-
  [Paper Note] Gradient Transformer: Learning to Generate Updates for LLMs
description: >-
  [ICML 2026][LLM Safety][update vector] This paper proposes Grad-Transformer, which "translates" the update vectors obtained by clients fine-tuning small models (TinyLM) on private data into update vectors for a target la…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "update vector"
  - "weak-to-strong distillation"
  - "Grad-Transformer"
  - "LoRA"
  - "differential privacy"
date: 2026-05-08
content_hash: 4b870e49fd3cc8d8
---

# Gradient Transformer: Learning to Generate Updates for LLMs

**Conference**: ICML 2026  
**arXiv**: [2605.27591](https://arxiv.org/abs/2605.27591)  
**Code**: TBD  
**Area**: Learned Optimizers / Data-Free Knowledge Distillation / Privacy-Preserving Fine-Tuning  
**Keywords**: update vector, weak-to-strong distillation, Grad-Transformer, LoRA, differential privacy

## TL;DR
This paper proposes Grad-Transformer, which "translates" the update vectors obtained by clients fine-tuning small models (TinyLM) on private data into update vectors for a target large model (LLM) using an encoder-decoder Transformer in an autoregressive manner. This achieves weak-to-strong knowledge distillation without any access to private data. It achieves an average PGR of 91.88% across six reasoning/summarization datasets, a 55.89% improvement over the best baseline (58.94%), and remains robust to differential privacy perturbations.

## Background & Motivation

**Background**: There are two mainstream ways to fine-tune LLMs on corporate private data: (1) clients fine-tune a small model (TinyLM) locally; (2) clients provide data to cloud service providers to fine-tune a large model. The former suffers from poor performance, while the latter violates privacy constraints like GDPR/HIPAA. The academic compromise is data-free knowledge distillation: training a generator to synthesize samples that "look like" private data to distill the student.

**Limitations of Prior Work**: Data-free KD has two major flaws—(a) the generator must be retrained from scratch for every new teacher, and distillation requires massive synthetic samples, leading to high computational costs; (b) synthetic samples may expose privacy-sensitive information through memorization or leakage (Annamalai et al., 2024), contradicting the "data-free" intent. Another approach, weak-to-strong KD (Burns et al., 2024), requires the teacher (weak) and student (strong) to share data, which also fails the "private data stays local" requirement.

**Key Challenge**: Traditional carriers of knowledge distillation are logits or synthetic samples, both of which either require data access or risk leakage. **Is there a "knowledge carrier" that can encode the fine-tuning effects on private data without being reversible into original samples?**

**Goal**: Design a mechanism $\mathcal{M}$ that allows a third-party provider to directly map a client-submitted TinyLM update vector $\Delta\theta_S=\theta_S^*-\theta_S^0$ to a target LLM update vector $\Delta\theta_T$ without **ever touching private data**, while supporting collaborative updates from multiple clients.

**Key Insight**: The authors observe that an update vector itself is a "compressed representation of cumulative gradient steps on a dataset"—it encapsulates the influence of private data as increments in the parameter space, which is more abstract than logits/synthetic samples and does not directly map to specific samples. If the correspondence between "TinyLM update $\leftrightarrow$ LLM update" can be learned on a **public shadow dataset**, this mapping can serve as a reusable "gradient translator."

**Core Idea**: Slice the update vector into token-like sequences according to attention blocks, and use a Flan-T5 encoder-decoder to autoregressively generate block-wise update vectors for the LLM. The entire mapping is trained once on shadow data and used via direct forward pass during deployment.

## Method

### Overall Architecture
The framework consists of three stages: (1) **Update vector curation**—the provider fine-tunes TinyLM and LLM on a public shadow dataset $D_p$ to collect $K$ pairs of $(\Delta\tilde\theta_{S,k}, \Delta\tilde\theta_{T,k})$; (2) **Train Grad-Transformer**—learn a seq2seq model on these pairs; (3) **Deploy**—the client fine-tunes TinyLM locally to obtain $\Delta\theta_{S,i}$ and sends it to the provider; the provider averages updates from multiple clients, feeds them into Grad-Transformer to obtain $\Delta\hat\theta_T$, and adds it to the initial LLM $\hat\theta_T=\theta_T^0+\Delta\hat\theta_T$, which is then returned to the client for inference.

### Key Designs

1.  **Update Vector as Distillation Carrier**:
    - **Function**: Replaces "logits / synthetic samples" with "parameter increments" as the medium for cross-model knowledge transfer.
    - **Mechanism**: Clients only upload $\Delta\theta_S=\theta_S^*-\theta_S^0$ (the difference relative to public initial weights); the provider performs mapping on this increment. Original private samples always stay with the client. LoRA $r=2$ adapters are used to further compress dimensions. Theoretically (Lemma 5.1, Theorem 5.2), generalization and utility bounds are both controlled by $I(w;D_p)$, allowing noisy algorithms like DP-SGD to reduce the dependence of $\Delta\theta_S$ on specific samples, further lowering privacy risks.
    - **Design Motivation**: Update vectors are low-variance, numerically stable "semantic compressions," providing one less leakage channel than synthetic data and being naturally compatible with privacy mechanisms like LoRA and DP. The shadow dataset is only used to learn the "correlation between two parameter spaces," independent of specific client data, allowing one Grad-Transformer to serve all clients.

2.  **Block-wise Tokenization for Scalability**:
    - **Function**: Simplifies the mapping of billion-dimensional parameter spaces into a translation task of token sequences with length $L_T$ and uniform dimensions.
    - **Mechanism**: For each attention block, the weight increments of Q/K/V/output projections are concatenated into a block-wise vector $\delta_{S,k}^j\in\mathbb{R}^{d_S}$, analogous to a token in a Transformer. Embedding layers $W_S^{emb},W_T^{emb}$ project different source/target block dimensions into the same hidden size; the encoder-decoder $\varphi$ processes the entire sequence, and $W_{out}$ projects back to the $d_T$-dimensional LLM block space. A naive approach (projecting all concatenated parameters) would require trillions of parameters; this method reduces the cost to a single Flan-T5-Large.
    - **Design Motivation**: Direct modeling of all parameters is infeasible. Slicing by attention blocks preserves the strong prior of "hierarchical correspondence" while keeping the sequence length within dozens or hundreds, fitting the scale Transformers excel at.

3.  **Teacher-forcing Training + Autoregressive Inference**:
    - **Function**: Enables the decoder to consider all TinyLM blocks and previously generated LLM blocks when generating the $j$-th LLM block update, capturing the coupling between internal LLM blocks.
    - **Mechanism**: Training uses teacher forcing $h_{T,k}^{<j}=W_T^{emb}(\delta_{T,k}^{<j})$ with an MSE objective: $\arg\min_w \tfrac{1}{KL_T}\sum_k\sum_j\|\hat\delta_{T,k}^j-\delta_{T,k}^j\|_2^2$. Inference switches to autoregressive mode using the decoder's own previous predictions (Eq. 11). In multi-client scenarios, $\{\Delta\theta_{S,i}\}$ are pooled (mean or sum) before being fed into $\mathcal{M}$, naturally supporting joint training.
    - **Design Motivation**: Parameter updates in different LLM layers are strongly correlated (deep attention depends on shallow semantics); independent block prediction would lose this structure. Autoregression is the standard paradigm for handling structured output in Transformers.

### Loss & Training
- **Training Goal**: Block-wise MSE (Eq. 10), optimized via Adam for 30 epochs, batch size 32, lr 2e-5 to 8e-5.
- **Data**: For each dataset, the training set is split in half; one half serves as private data $D$, the other as shadow data $D_p$. $D_p$ is randomly sampled into $K=300$ subsets (1024 samples each), each fine-tuned with LoRA $r=2$ until convergence. The adapters from the last 200 steps are collected as update vector pairs, totaling 60k tuples, split 95:5 for training/validation.
- **Models**: TinyLM = Qwen2.5-3B-Instruct, LLM = Qwen2.5-7B-Instruct, $\varphi$ = Flan-T5-Large.

## Key Experimental Results

### Main Results (Single Client, higher PGR % is better)

| Dataset | $P_S$ (TinyLM) | Best Baseline | Grad-Transformer | $P_T$ (LLM Upper Bound) |
|--------|--------------:|--------------:|-----------------:|-----------------:|
| AQuA-RAT (Acc) | 48.43 | 47.64 (W2S Conf) | **61.02** | 58.66 |
| GSM8K (Acc) | 62.62 | 74.30 (W2S Conf) | **73.59** | 73.16 |
| DROP (Acc) | 49.36 | 54.18 (W2S Conf) | **58.26** | 59.01 |
| CommonsenseQA (Acc) | 77.40 | 83.46 | 83.21 | 83.78 |
| SAMSum (R-1) | 47.64 | 49.92 | **50.52** | 50.59 |
| DialogSum (R-1) | 46.43 | 47.70 | 48.37 | 50.92 |

**Key Findings**: On AQuA-RAT, Grad-Transformer's accuracy (61.02%) even **exceeds the upper bound of direct LLM fine-tuning** (58.66%), reaching a PGR of 123%. This suggesting that the "gradient translation" learned on shadow data possesses regularization or ensemble effects. The average PGR of 91.88% significantly outperforms the best baseline at 58.94% (+55.89%). Notably, the three baselines (W2S, Conf, VisSup) **all require access to private data**, while Ours is the **only data-free** method.

### Comparison Table

| Dimension | Data-Free KD Baselines | Weak-to-Strong KD Baselines | Grad-Transformer |
|------|----------------------|---------------------------|------------------|
| Access Private Data | ✗ (but needs generator training) | ✓ | **✗** |
| Retrain per Teacher | ✓ (Expensive) | – | ✗ (One-time mapping) |
| Synthetic Leakage Risk | High | – | **No synthetic samples** |
| Multi-client Aggregation | Difficult | Difficult | ✓ (Pool update vectors) |
| Compatibility | Partial | Partial | ✓ (DP / LoRA) |

### Key Findings
- **Block-wise tokenization is the key to scalability**: It reduces the trillion-scale parameter mapping to a Flan-T5-Large scale; otherwise, the architecture would be untrainable.
- **DP Robustness**: When DP-SGD noise is added on the client side, Grad-Transformer's performance drops significantly less than baselines. Its "translation ability" stems primarily from learned correlations between model spaces on shadow data, rather than exact client $\Delta\theta_S$.
- **Theory-Experiment Alignment**: Theorem 5.2 predicts utility bounds depend on $I(w;D_p)+\mathrm{KL}(\tilde\mu\|\mu)$. Experiments show optimal results when shadow $D_p$ and private $D$ are identically distributed; cross-distribution scenarios lead to performance drops, suggesting careful selection of shadow data before deployment.

## Highlights & Insights
- **New Paradigm of "Gradients as Knowledge"**: Treating update vectors as learnable, translatable "knowledge token sequences" is a critical extension of work like model soup/task arithmetic. While the latter performs arithmetic within the same architecture, this work bridges **cross-architecture and cross-scale** parameter space mapping.
- **Elegant Balance of Privacy-Utility-Cost**: Clients only need to train a 3B model locally and never upload data. Providers train Grad-Transformer once to serve all clients for the same task. This compresses the cost of "repeated gradient communication" in federated learning into a "one-time LoRA adapter upload."
- **Transferable Tricks**: The combination of block-wise serialization and encoder-decoder autoregression can be directly transferred to model merging, cross-architecture adapter transfer, or even "training dynamics prediction"—any task requiring mapping between two high-dimensional parameter spaces.

## Limitations & Future Work
- The authors admit that Grad-Transformer performance depends heavily on the alignment between shadow dataset $D_p$ and client private data (the $\mathrm{KL}(\tilde\mu\|\mu)$ term in Theorem 5.2). Finding suitable $D_p$ for niche client tasks might be difficult.
- This study only validates cross-scale mapping within the **same family (Qwen2.5)**, such as 3B→7B and 7B→14B. Feasibility across model families (e.g., LLaMA→Qwen) remains unknown. Additionally, LoRA $r=2$ is aggressive compression; it is unclear if Grad-Transformer scales when dimensions explode in full fine-tuning scenarios.
- Future improvements could involve hierarchical block-wise sequences (layer-group then intra-layer) or introducing "model architecture embeddings" as prompts to allow one Grad-Transformer to serve multiple teacher-student combinations.

## Related Work & Insights
- **vs Burns et al. 2024 (Weak-to-Strong)**: W2S uses weak teacher outputs (logits/labels) to supervise a strong student, requiring both to see the same data. This paper uses the weak teacher's **parameter increments**, and data is only seen by the weak teacher; the strong student is entirely data-free.
- **vs Data-Free KD (Tran et al., 2024; Wei et al., 2025)**: These train generators to synthesize data for distillation, requiring retraining per teacher and risking leakage. This paper uses no generators; the "translator" is pre-trained and reusable.
- **vs Task Arithmetic / Model Soup**: The latter performs addition/subtraction of $\Delta\theta$ within the same architecture. This paper learns a **non-linear cross-architecture mapping** $\Delta\theta_S\mapsto\Delta\theta_T$, serving as a "cross-scale superset" of task arithmetic.
- **vs LoRA Adapter Hub**: While adapter hubs reuse others' trained adapters, this work functions as an "adapter translator"—translating small model adapters into large model adapters, allowing resource-constrained users to benefit from large models.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ "Translating gradients with Transformer" is a truly novel perspective, turning cross-scale/cross-architecture parameter mapping into a seq2seq task.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 6 datasets, 3 baselines, single/multi-client, and DP settings, but lacks validation across different model families (e.g., LLaMA/Mistral).
- Writing Quality: ⭐⭐⭐⭐ The three-stage framework is clearly explained, and the link between theory (Lemma 5.1/Theorem 5.2), method, and experiments is tight.
- Value: ⭐⭐⭐⭐⭐ Directly addresses real-world pain points of private LLM fine-tuning. It is engineering-ready (LoRA/DP compatible) and has the potential to become a new baseline for privacy-preserving LLM services.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Geminio: Language-Guided Gradient Inversion Attacks in Federated Learning](../../ICCV2025/llm_safety/geminio_language-guided_gradient_inversion_attacks_in_federated_learning.md)
- [\[ICML 2026\] Decoupled Training with Local Reinforcement Fine-Tuning in Federated Learning](decoupled_training_with_local_reinforcement_fine-tuning_in_federated_learning.md)
- [\[AAAI 2026\] Ghost in the Transformer: Detecting Model Reuse with Invariant Spectral Signatures](../../AAAI2026/llm_safety/ghost_in_the_transformer_detecting_model_reuse_with_invariant_spectral_signature.md)
- [\[AAAI 2026\] FedP²EFT: Federated Learning to Personalize PEFT for Multilingual LLMs](../../AAAI2026/llm_safety/fedp2eft_federated_learning_to_personalize_peft_for_multilingual_llms.md)
- [\[ICML 2026\] Understanding Generalization and Forgetting in In-Context Continual Learning](understanding_generalization_and_forgetting_in_in-context_continual_learning.md)

</div>

<!-- RELATED:END -->
