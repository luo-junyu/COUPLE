# COUPLE - Cross-Domain Diffusion with Progressive Alignment for Efficient Adaptive Retrieval
The codes and checkpoints of our paper "Cross-Domain Diffusion with Progressive Alignment for Efficient Adaptive Retrieval".

## Request

```bash
pytorch
sklearn
scipy
faiss
networkx
``` 
## Files

```bash
.
├── src
│   ├── codes of our method
|── ckpts
│   ├── checkpoints of our method

```

## Usage

```bash
python -u main.py --nbit $BITS --dataset $DATASET --domain $DOMAIN ...

```

## Abstract

Unsupervised efficient domain adaptive retrieval aims to transfer knowledge from a labeled source domain to an unlabeled target domain, while maintaining low storage cost and high retrieval efficiency. However, existing methods typically fail to address potential noise in the target domain, and directly align high-level features across domains, thus result in suboptimal retrieval performance. Solutions to the noise-related challenges are underexplored. In this paper, we propose a novel Cross-Domain Diffusion with Progressive Alignment method~(COUPLE). This approach revisits unsupervised efficient domain adaptive retrieval from a graph diffusion perspective, simulating cross-domain adaptation dynamics to achieve a stable adaptation process in the target domain. First, we construct a cross-domain relationship graph and leverage noise-robust graph flow diffusion to simulate the transfer dynamics from the source domain to the target domain, identifying high-confidence local clusters. We then leverage the graph diffusion results for discriminative hash code learning, effectively learning from the target domain while reducing the negative impact of noise. Furthermore, we employ a hierarchical Mixup operation for progressive domain alignment, which is performed along the cross-domain random walk paths. Utilizing target domain discriminative hash learning and progressive domain alignment, COUPLE enables effective domain adaptive hash learning. Extensive experiments on several benchmark datasets demonstrate the effectiveness of our proposed COUPLE.
