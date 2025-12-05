<div align="center">
  <!-- <h1><b> BasicTS </b></h1> -->
  <!-- <h2><b> BasicTS </b></h2> -->
  <h2><b> Awesome Mixture-of-Experts for Time Series Analysis (MoE4TS) </b></h2>
</div>

<div align="center">

[![Awesome](https://awesome.re/badge.svg)](https://github.com/syrGitHub/Awesome-MoE4TS)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![](https://img.shields.io/github/last-commit/syrGitHub/Awesome-MoE4TS?color=green)
![](https://img.shields.io/github/stars/syrGitHub/Awesome-MoE4TS?color=yellow)
![](https://img.shields.io/github/forks/syrGitHub/Awesome-MoE4TS?color=lightblue)
![](https://img.shields.io/badge/PRs-Welcome-red)

</div>

<div align="center">

**[<a href="">Paper Page</a>]**

</div>

<p align="center">

<img src="./assets/Figure 1.png" width="600">

</p>

---

> 🔥 Abundant resources related to [**MoE for time series analysis (MoE4TS)**]() by _[Yanru Sun](https://syrgithub.github.io/), [Dilfira Kudrat](), [Zongxia Xie](), [Qinghua Hu](https://cic.tju.edu.cn/faculty/huqinghua/index.html), [Emadeldeen Eldele](https://emadeldeen24.github.io/), [Zhenghua Chen](https://zhenghuantu.github.io/), [Xiaoli Li](https://www.sutd.edu.sg/profile/li-xiaoli/), [Chee Keong Kwoh](https://scholar.google.com/citations?hl=zh-CN&user=jVn0wDMAAAAJ), [Min Wu](https://sites.google.com/site/wumincf/)_
>
> 🙋 Please let us know if you find out a mistake or have any suggestions!
> 
> 🌟 If you find this resource helpful, please consider to star this repository and cite our survey paper:

```
@article{sun2025moe4ts,
  title={Mixture-of-Experts for Time Series Analysis: Taxonomy, Progress, and Prospects},
  author={Sun, Yanru and Kudrat, Dilfira and Xie, Zongxia and Hu, Qinghua and Eldele, Emadeldeen and Chen, Zhenghua and Li, Xiaoli and Kwoh, Chee Keong and Wu, Min},
  year={2025}
}
```

Time series analysis plays a central role in numerous real-world applications, including finance, healthcare, and transportation. In recent years, mixture-of-experts (MoE) models have been increasingly adopted for time series tasks. This repository provides a curated collection of resources related to MoE for time series analysis (MoE4TS).

时间序列分析在金融、医疗健康、交通运输等众多真实场景中发挥着关键作用。近年来，混合专家（Mixture-of-Experts, MoE）模型在各类时间序列任务中得到越来越广泛的应用。本仓库旨在系统整理与时间序列混合专家方法（MoE4TS）相关的资源。

<p align="center">
<img src="./assets/taxonomy.png" width="600">
</p>

We provide a unified taxonomy of MoE4TS that consists of two major dimensions. The first dimension, **routing strategy**, categorizes MoE architectures based on how inputs are dynamically dispatched to experts, including ungated routing, dense-gated mechanisms, and sparse-gated schemes. The second dimension, **expert design**, characterizes the architectural forms of the experts themselves, covering homogeneous, heterogeneous, and shared-expert paradigms.

我们提出了一个统一的 MoE4TS 分类体系，由两个主要维度组成。第一个维度是**路由策略**，根据输入如何被动态分配给专家对 MoE 架构进行划分，包括无路由机制、密集路由机制以及稀疏路由机制。第二个维度是**专家设计**，用于刻画专家本身的结构形式，涵盖同质专家、异质专家以及共享专家等范式。

## ✨ News
- [2025-12-05] 📮 We have released this repository that collects the resources related to MoE for time series analysis (MoE4TS). We will keep updating this repository, and welcome to **STAR🌟** and **WATCH** to keep track of it.

## 🔭 Table of Contents
- [Awesome-MoE4TS](#awesome-MoE4ts)
  - [News](#news)
  - [Collection of Papers](#collection-of-papers)
    - [Gating Strategy](#Gating-Strategy)
        - [Ungated Models](#Ungated-Models)
        - [Dense-gated Models](#Dense-gated-Models)
        - [Sparse-gated Models](#Sparse-gated-Models)
    - [Expert Design](#Expert-Design)
        - [Homogeneous Experts](#Homogeneous-Experts)
        - [Heterogeneous Experts](#Heterogeneous-Experts)
        - [Shared Experts](#Shared-Experts)
    
## 📚 Collection of Papers

### Gating Strategy
 
#### Ungated Models

| Title | Venue   | Month   | Code |
| ------- | ------- | ------- | ------- |
|Film [Film: Frequency improved legendre memory model for long-term time series forecasting](https://proceedings.neurips.cc/paper_files/paper/2022/file/524ef58c2bd075775861234266e5e020-Paper-Conference.pdf) | NeurIPS | 2022 | [Code](https://github.com/tianzhou2011/FiLM/) |
|LeMoLE [Lemole: Llm-enhanced mixture of linear experts for time series forecasting](https://arxiv.org/pdf/2412.00053) | Arxiv | 2024.11 | None |
|PTN [Learned Data Transformation: A Data-centric Plugin for Enhancing Time Series Forecasting](https://openreview.net/pdf?id=6hJ3khuJY4) | None | 2025 | None|
|COST [CoST: Contrastive Learning of Disentangled Seasonal-Trend Representations for Time Series Forecasting](https://openreview.net/pdf?id=PilZY3omXV2) | ICLR | 2022 | [Code](https://github.com/salesforce/CoST) |
|VPF-MoE [A vegetable-price forecasting method based on mixture of experts](https://www.mdpi.com/2077-0472/15/2/162) | Agriculture | 2025 | None |


#### Dense-gated Models

-  Linear-based

| Title | Venue   | Month   | Code |
| ------- | ------- | ------- | ------- |
|Fedformer [Fedformer: Frequency enhanced decomposed transformer for long-term series forecasting](https://proceedings.mlr.press/v162/zhou22g/zhou22g.pdf) | ICML | 2022 | [Code](https://github.com/MAZiqing/FEDformer) |
|TCNet [Temporal chain network with intuitive attention mechanism for long-term series forecasting](https://ieeexplore.ieee.org/abstract/document/10273738) | IEEE TIM | 2023 | None |
|MoE-Traffic [Interpretable mixture of experts for time series prediction under recurrent and non-recurrent conditions](https://arxiv.org/pdf/2409.03282) | Arxiv | 2024.09 | None|
|MoPE [Mixture of Projection Experts for Multivariate Long-Term Time Series Forecasting](https://ieeexplore.ieee.org/abstract/document/10903405) | ICMLA | 2024 | None |
|VE [VE: Modeling Multivariate Time Series Correlation with Variate Embedding](https://ieeexplore.ieee.org/abstract/document/10818803) | ICAMechS | 2024 | [Code](https://github.com/swang-song/VE) |
|FreqMoE [FreqMoE: Enhancing Time Series Forecasting through Frequency Decomposition Mixture of Experts](https://proceedings.mlr.press/v258/liu25i.html) | ICAIS | 2025 | [Code](https://github.com/sunbus100/FreqMoE-main) |
|SMETimes [Small but Mighty: Enhancing Time Series Forecasting with Lightweight LLMs](https://arxiv.org/pdf/2503.03594) | Arxiv | 2025.03 | [Code](https://github.com/xiyan1234567/SMETimes) |
|WaveTS-M [Wavelet Mixture of Experts for Time Series Forecasting](https://arxiv.org/pdf/2508.08825) | Arxiv | 2025.08 | None |
|TSEPMoE [Time Series-Based Electric Load Forecasting with Mixture of Expert System](https://ieeexplore.ieee.org/abstract/document/10277743) | ICHCI | 2023 | None |
|IMMOE [Improved multi-gate mixture-of-experts framework for multi-step prediction of gas load](https://www.sciencedirect.com/science/article/abs/pii/S0360544223017383) | Energy | 2023 | None |
|MoE-KAN [Interpretable mixture of experts for time series prediction under recurrent and non-recurrent conditions](https://arxiv.org/pdf/2409.03282) | Arxiv | 2024.09 | None |
|DeepUnifiedMoM [DeepUnifiedMom: Unified Time-series Momentum Portfolio Construction via Multi-Task Learning with Multi-Gate Mixture of Experts](https://arxiv.org/pdf/2406.08742) | Arxiv | 2024.06 | [Code](https://github.com/joelowj/unified_mom_mmoe) |
|DFMH [Dynamic fusion of multi-source heterogeneous data using MOE mechanism for stock prediction](https://link.springer.com/article/10.1007/s10489-025-06330-7) | Applied Intelligence | 2025 | None |
|MambaMoE [From news to trends: a financial time series forecasting framework with LLM-driven news sentiment analysis and selective state spaces](https://link.springer.com/article/10.1007/s10844-025-00971-3) | Journal of Intelligent Information Systems | 225 | None |
|CAD [Beyond sharing: Conflict-aware multivariate time series anomaly detection](https://dl.acm.org/doi/pdf/10.1145/3611643.3613896) | ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering | 2023 | [Code](https://github.com/dawnvince/MTS_CAD) |
|STGCN-MoE [Mixture of Experts based Model Integration for Traffic State Prediction](https://www.ece.nus.edu.sg/stfpage/eletck/papers/ChattoTham-IEEE-VTC-Spring-2022.pdf) | VTC | 2022 | None |
|AME [Attention with Mixture Experts model for Multivariate Time Series Imputation](https://ieeexplore.ieee.org/abstract/document/10929537) | EIECC | 2024 | None |

- Atention

| Title | Venue   | Month   | Code |
| ------- | ------- | ------- | ------- |
|MAES [Model-Attentive Ensemble Learning for Sequence Modeling](https://arxiv.org/pdf/2102.11500) | Arxiv | 2021.02 | None |
|DynaMix [True zero-shot inference of dynamical systems preserving long-term statistics](https://arxiv.org/pdf/2505.13192) | Arxiv | 2025.05 | [Code](https://github.com/DurstewitzLab/DynaMix-python) |
|MoME [MoME: Mixture of Multi-Domain Experts for Multivariate Long-Term Series Forecasting](https://ieeexplore.ieee.org/abstract/document/10887716/) | ICASSP | 2025 | [Code](https://github.com/lxy-PhD2022/MoME) |
|Graph-MoE [Graph mixture of experts and memory-augmented routers for multivariate time series anomaly detection](https://ojs.aaai.org/index.php/AAAI/article/view/33921) | AAAI | 2025 | [Code](https://github.com/dearlexie1128/Graph-MoE) |
|TITAN [A time series is worth five experts: Heterogeneous mixture of experts for traffic flow prediction](https://arxiv.org/pdf/2409.17440) | Arxiv | 2024.09 | [Code](https://github.com/sqlcow/TITAN) |
|CICADA [CICADA: Cross-Domain Interpretable Coding for Anomaly Detection and Adaptation in Multivariate Time Series](https://arxiv.org/pdf/2505.00415) | Arxiv | 2025.05 | None |

- LSTM

| Title | Venue   | Month   | Code |
| ------- | ------- | ------- | ------- |
|Dychem [Dynamic combination of heterogeneous models for hierarchical time series](https://ieeexplore.ieee.org/abstract/document/10031198) | ICDMW | 2022 | [Code](https://github.com/aaronhan223/htsf) |
|TMMOE [A temporal multi-gate mixture-of-experts approach for vehicle trajectory and driving intention prediction](https://ieeexplore.ieee.org/abstract/document/10330032/) | IEEE Transactions on Intelligent Vehicles | 2023 | None |


- Non-parameteric

| Title | Venue   | Month   | Code |
| ------- | ------- | ------- | ------- |
|MSGNet [Msgnet: Learning multi-scale inter-series correlations for multivariate time series forecasting](https://ojs.aaai.org/index.php/AAAI/article/view/28991) | AAAI | 2024 | [Code](https://github.com/YoZhibo/MSGNet) |
|MoE-F [Filtered not Mixed: Filtering-Based Online Gating for Mixture of Large Language Models](https://openreview.net/pdf?id=ecIvumCyAj) | ICLR | 2025 | [Code](https://github.com/raeidsaqur/moe-f) |
|MOOE [Mixture of Online and Offline Experts for Non-Stationary Time Series](https://ojs.aaai.org/index.php/AAAI/article/view/34448) | AAAI | 2025 | [Code](https://github.com/Lawliet-zzl/MOOE) |


#### Sparse-gated Models
 
- Parameteric

| Title | Venue   | Month   | Code |
| ------- | ------- | ------- | ------- |
|MoE-AMC [Moe-amc: Enhancing automatic modulation classification performance using mixture-of-experts](https://arxiv.org/pdf/2312.02298) | Arxiv | 2023.12 | None |
|ARTEMIS [Artemis: Autoregressive end-to-end trajectory planning with mixture of experts for autonomous driving](https://arxiv.org/pdf/2504.19580) | Arxiv | 2025.04 | [Code](None) |
|xTime [xTime: Extreme Event Prediction with Hierarchical Knowledge Distillation and Expert Fusion](https://arxiv.org/pdf/2510.20651) | Arxiv | 2025.10 | None |
|MGTS-Net [MGTS-Net: Exploring Graph-Enhanced Multimodal Fusion for Augmented Time Series Forecasting](https://arxiv.org/pdf/2510.16350) | Arxiv | 2025.10 | None |
|Pathformer [Pathformer: Multi-scale Transformers with Adaptive Pathways for Time Series Forecasting](https://openreview.net/pdf?id=lJkOCMP2aW) | ICLR | 2024 | [Code](https://github.com/decisionintelligence/pathformer) |
|Super-Linear [Super-Linear: A Lightweight Pretrained Mixture of Linear Experts for Time Series Forecasting](https://arxiv.org/pdf/2509.15105) | Arxiv | 2025.09 | [Code](https://github.com/azencot-group/SuperLinear) |
|TransMoE [Transformer with Sparse Mixture of Experts for Time-Series Data Prediction in Industrial IoT Systems](https://www.scirp.org/journal/paperinformation?paperid=141557) | Engineering | 2025 | None |
|MoU [Semantics-Aware Patch Encoding and Hierarchical Dependency Modeling for Long-Term Time Series Forecasting](https://dl.acm.org/doi/abs/10.1145/3711896.3737123) | SIGKDD | 2025 | [Code](https://github.com/lunaaa95/mou/) |
|SoftShape [Learning Soft Sparse Shapes for Efficient Time-Series Classification](https://openreview.net/pdf?id=B9DOjtj9xK) | ICML | 2025 | [Code](https://github.com/qianlima-lab/SoftShape) |
|BP-MoE [BP-MoE: Behavior Pattern-aware Mixture-of-Experts for temporal graph representation learning](https://www.sciencedirect.com/science/article/abs/pii/S0950705124006907) | KBS | 2024 | [Code](https://github.com/rekaBelbA/BP-MoE) |
|TopK Dense [Advancing accuracy in energy forecasting using mixture-of-experts and federated learning](https://dl.acm.org/doi/abs/10.1145/3632775.3661945) | International Conference on Future and Sustainable Technologies | 2024 | None |
|Time-MoE [Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts](https://openreview.net/pdf?id=e1wDDFmlVu) | ICLR | 2025 | [Code](https://github.com/time-moe/time-moe) |
|ContexTST [Unify and Anchor: A Context-Aware Transformer for Cross-Domain Time Series Forecasting](https://arxiv.org/pdf/2503.01157?) | Arxiv | 2025.03 | None |
|MoFE-Time [MoFE-Time: Mixture of Frequency Domain Experts for Time-Series Forecasting Models](https://arxiv.org/pdf/2507.06502) | Arxiv | 2025.07 | None |
|CAPTime [Context-Aware Probabilistic Modeling with LLM for Multimodal Time Series Forecasting](https://arxiv.org/pdf/2505.10774) | Arxiv | 2025.05 | None |
|MMK [Are KANs Effective for Multivariate Time Series Forecasting?](https://arxiv.org/pdf/2408.11306?) | Arxiv | 2024.08 | [Code](https://github.com/smilehanCN/EasyTSF) |
|TALON [Adapting LLMs to Time Series Forecasting via Temporal Heterogeneity Modeling and Semantic Alignment](https://arxiv.org/pdf/2508.07195?) | Arxiv | 2025.08 | [Code](https://github.com/syrGitHub/TALON) |
|MixMamba [MixMamba: Time series modeling with adaptive expertise](https://www.sciencedirect.com/science/article/abs/pii/S1566253524003671) | Information Fusion | 2024 | None |
|TimeFilter [TimeFilter: Patch-Specific Spatial-Temporal Graph Filtration for Time Series Forecasting](https://openreview.net/pdf?id=490VcNtjh7) | ICML | 2025 | [Code](https://github.com/TROUBADOUR000/TimeFilter) |
 
- Similarity-deiven

| Title | Venue   | Month   | Code |
| ------- | ------- | ------- | ------- |
|TFPS [Learning pattern-specific experts for time series forecasting under patch-level distribution shift](https://openreview.net/pdf?id=CtoIG9Iwas) | NeurIPS | 2025 | [Code](https://github.com/syrGitHub/TFPS) |
|Moirai-MoE [Moirai-MoE: Empowering Time Series Foundation Models with Sparse Mixture of Experts](https://openreview.net/pdf?id=SrEOUSyJcR) | ICML | 2025 | [Code](https://github.com/liuxu77/uni2ts) |
|Time Tracker [Time Tracker: Mixture-of-Experts-Enhanced Foundation Time Series Forecasting Model with Decoupled Training Pipelines](https://arxiv.org/pdf/2505.15151) | Arxiv | 2025.05 | None |
|ProtoN-FM [Bridging Distribution Gaps in Time Series Foundation Model Pretraining with Prototype-Guided Normalization](https://arxiv.org/pdf/2504.10900) | Arxiv | 2025.04 | None |
|FuseMoE [Fusemoe: Mixture-of-experts transformers for fleximodal fusion](https://proceedings.neurips.cc/paper_files/paper/2024/file/7d62a85ebfed2f680eb5544beae93191-Paper-Conference.pdf) | NeurIPS | 2024 | None |
|FT-MoE [FT-MoE: Sustainable-learning Mixture of Experts Model for Fault-Tolerant Computing with Multiple Tasks](https://arxiv.org/pdf/2504.20446?) | Arxiv | 2025.04 | None |
|TimeExpert [TimeExpert: Boosting Long Time Series Forecasting with Temporal Mix of Experts](https://arxiv.org/pdf/2509.23145?) | Arxiv | 2025.09 | [Code](https://github.com/xwmaxwma/TimeExpert) |

### Expert Design

#### Homogeneous Experts

- FFN

| Title | Venue   | Month   | Code |
| ------- | ------- | ------- | ------- |
|MoPE [Mixture of Projection Experts for Multivariate Long-Term Time Series Forecasting](https://ieeexplore.ieee.org/abstract/document/10903405) | ICMLA | 2024 | None |
|VE [VE: Modeling Multivariate Time Series Correlation with Variate Embedding](https://ieeexplore.ieee.org/abstract/document/10818803) | ICAMechS | 2024 | [Code](https://github.com/swang-song/VE) |
|FuseMoE [Fusemoe: Mixture-of-experts transformers for fleximodal fusion](https://proceedings.neurips.cc/paper_files/paper/2024/file/7d62a85ebfed2f680eb5544beae93191-Paper-Conference.pdf) | NeurIPS | 2024 | None |
|TFPS [Learning pattern-specific experts for time series forecasting under patch-level distribution shift](https://openreview.net/pdf?id=CtoIG9Iwas) | NeurIPS | 2025 | [Code](https://github.com/syrGitHub/TFPS) |
|FreqMoE [FreqMoE: Enhancing Time Series Forecasting through Frequency Decomposition Mixture of Experts](https://proceedings.mlr.press/v258/liu25i.html) | ICAIS | 2025 | [Code](https://github.com/sunbus100/FreqMoE-main) |
|AMD [Adaptive multi-scale decomposition framework for time series forecasting](https://ojs.aaai.org/index.php/AAAI/article/view/33908) | AAAI | 2025 | [Code](https://github.com/TROUBADOUR000/AMD) |
|MoME [MoME: Mixture of Multi-Domain Experts for Multivariate Long-Term Series Forecasting](https://ieeexplore.ieee.org/abstract/document/10887716/) | ICASSP | 2025 | [Code](https://github.com/lxy-PhD2022/MoME) |
|MOOE [Mixture of Online and Offline Experts for Non-Stationary Time Series](https://ojs.aaai.org/index.php/AAAI/article/view/34448) | AAAI | 2025 | [Code](https://github.com/Lawliet-zzl/MOOE) |
|WaveTS-M [Wavelet Mixture of Experts for Time Series Forecasting](https://arxiv.org/pdf/2508.08825) | Arxiv | 2025.08 | None |
|MoU [Semantics-Aware Patch Encoding and Hierarchical Dependency Modeling for Long-Term Time Series Forecasting](https://dl.acm.org/doi/abs/10.1145/3711896.3737123) | SIGKDD | 2025 | [Code](https://github.com/lunaaa95/mou/) |
|FT-MoE [FT-MoE: Sustainable-learning Mixture of Experts Model for Fault-Tolerant Computing with Multiple Tasks](https://arxiv.org/pdf/2504.20446?) | Arxiv | 2025.04 | None |
|TimeExpert [TimeExpert: Boosting Long Time Series Forecasting with Temporal Mix of Experts](https://arxiv.org/pdf/2509.23145?) | Arxiv | 2025.09 | [Code](https://github.com/xwmaxwma/TimeExpert) |
|MGTS-Net [MGTS-Net: Exploring Graph-Enhanced Multimodal Fusion for Augmented Time Series Forecasting](https://arxiv.org/pdf/2510.16350) | Arxiv | 2025.10 | None |
|LeMoLE [Lemole: Llm-enhanced mixture of linear experts for time series forecasting](https://arxiv.org/pdf/2412.00053) | Arxiv | 2024.11 | None |
|Soft/TopK Dense [Advancing accuracy in energy forecasting using mixture-of-experts and federated learning](https://dl.acm.org/doi/pdf/10.1145/3632775.3661945) | ACM International Conference on Future and Sustainable Energy Systems | 2024 | None |
|TransMoE [Transformer with Sparse Mixture of Experts for Time-Series Data Prediction in Industrial IoT Systems](https://www.scirp.org/journal/paperinformation?paperid=141557) | Engineering | 2025 | None |
|93 [Financial time series prediction using mixture of experts](https://link.springer.com/chapter/10.1007/978-3-540-39737-3_69) | International Symposium on Computer and Information Sciences | 2003 | None |
|ANFIS [Mixture of MLP-experts for trend forecasting of time series: A case study of the Tehran stock exchange]() | International Journal of Forecasting | 2011 | None |
|SMETimes [Small but Mighty: Enhancing Time Series Forecasting with Lightweight LLMs](https://arxiv.org/pdf/2503.03594) | Arxiv | 2025.03 | [Code](https://github.com/xiyan1234567/SMETimes) |
|CAPTime [Context-Aware Probabilistic Modeling with LLM for Multimodal Time Series Forecasting](https://arxiv.org/pdf/2505.10774) | Arxiv | 2025.05 | None |
|Moirai-MoE [Moirai-MoE: Empowering Time Series Foundation Models with Sparse Mixture of Experts](https://openreview.net/pdf?id=SrEOUSyJcR) | ICML | 2025 | [Code](https://github.com/liuxu77/uni2ts) |
|Time-MoE [Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts](https://openreview.net/pdf?id=e1wDDFmlVu) | ICLR | 2025 | [Code](https://github.com/time-moe/time-moe) |
|ULoRA-MoE [Uncertainty-aware Fine-tuning on Time Series Foundation Model for Anomaly Detection](https://openreview.net/pdf?id=W1wlE4bPqP) | None | 2025 | None |



- LSTM


| Title | Venue   | Month   | Code |
| ------- | ------- | ------- | ------- |
|Soft/TopK LSTM [Advancing accuracy in energy forecasting using mixture-of-experts and federated learning](https://dl.acm.org/doi/pdf/10.1145/3632775.3661945) | ACM International Conference on Future and Sustainable Energy Systems | 2024 | None |
|TMMOE [A temporal multi-gate mixture-of-experts approach for vehicle trajectory and driving intention prediction](https://ieeexplore.ieee.org/abstract/document/10330032/) | IEEE Transactions on Intelligent Vehicles | 2023 | None |
|DeepUnifiedMoM [DeepUnifiedMom: Unified Time-series Momentum Portfolio Construction via Multi-Task Learning with Multi-Gate Mixture of Experts](https://arxiv.org/pdf/2406.08742) | Arxiv | 2024.06 | [Code](https://github.com/joelowj/unified_mom_mmoe) |

- Attention


| Title | Venue   | Month   | Code |
| ------- | ------- | ------- | ------- |
|Pathformer [Pathformer: Multi-scale Transformers with Adaptive Pathways for Time Series Forecasting](https://openreview.net/pdf?id=lJkOCMP2aW) | ICLR | 2024 | [Code](https://github.com/decisionintelligence/pathformer) |
|TITAN [A time series is worth five experts: Heterogeneous mixture of experts for traffic flow prediction](https://arxiv.org/pdf/2409.17440) | Arxiv | 2024.09 | [Code](https://github.com/sqlcow/TITAN) |
|DFMH [Dynamic fusion of multi-source heterogeneous data using MOE mechanism for stock prediction](https://link.springer.com/article/10.1007/s10489-025-06330-7) | Applied Intelligence | 2025 | None |
|POND [POND: Multi-source time series domain adaptation with information-aware prompt tuning]() | SIGKDD | 2024 | None |
|Graph-MoE [Graph mixture of experts and memory-augmented routers for multivariate time series anomaly detection](https://ojs.aaai.org/index.php/AAAI/article/view/33921) | AAAI | 2025 | [Code](https://github.com/dearlexie1128/Graph-MoE) |


- Mamba


| Title | Venue   | Month   | Code |
| ------- | ------- | ------- | ------- |
|MixMamba [MixMamba: Time series modeling with adaptive expertise](https://www.sciencedirect.com/science/article/abs/pii/S1566253524003671) | Information Fusion | 2024 | None |
|MambaMoE [From news to trends: a financial time series forecasting framework with LLM-driven news sentiment analysis and selective state spaces](https://link.springer.com/article/10.1007/s10844-025-00971-3) | Journal of Intelligent Information Systems | 225 | None |

- Others

| Title | Venue   | Month   | Code |
| ------- | ------- | ------- | ------- |
|COST [CoST: Contrastive Learning of Disentangled Seasonal-Trend Representations for Time Series Forecasting](https://openreview.net/pdf?id=PilZY3omXV2) | ICLR | 2022 | [Code](https://github.com/salesforce/CoST) |
|TCNet [Temporal chain network with intuitive attention mechanism for long-term series forecasting](https://ieeexplore.ieee.org/abstract/document/10273738) | IEEE TIM | 2023 | None |
|MSGNet [Msgnet: Learning multi-scale inter-series correlations for multivariate time series forecasting](https://ojs.aaai.org/index.php/AAAI/article/view/28991) | AAAI | 2024 | [Code](https://github.com/YoZhibo/MSGNet) |
|TimeFilter [TimeFilter: Patch-Specific Spatial-Temporal Graph Filtration for Time Series Forecasting](https://openreview.net/pdf?id=490VcNtjh7) | ICML | 2025 | [Code](https://github.com/TROUBADOUR000/TimeFilter) |
|STGCN-MoE [Mixture of Experts based Model Integration for Traffic State Prediction](https://ieeexplore.ieee.org/abstract/document/9860682) | VTC | 2022 | None |
|TFMoE [Continual traffic forecasting via mixture of experts](https://arxiv.org/pdf/2406.03140) | Arxiv | 2024.06 | None |
|MMK [Are KANs Effective for Multivariate Time Series Forecasting?](https://arxiv.org/pdf/2408.11306?) | Arxiv | 2024.08 | [Code](https://github.com/smilehanCN/EasyTSF) |


- Custom

| Title | Venue   | Month   | Code |
| ------- | ------- | ------- | ------- |
|CAD [Beyond sharing: Conflict-aware multivariate time series anomaly detection](https://dl.acm.org/doi/pdf/10.1145/3611643.3613896) | ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering | 2023 | [Code](https://github.com/dawnvince/MTS_CAD) |
|MoE-Traffic [Interpretable mixture of experts for time series prediction under recurrent and non-recurrent conditions](https://arxiv.org/pdf/2409.03282) | Arxiv | 2024.09 | None|
|xTime [xTime: Extreme Event Prediction with Hierarchical Knowledge Distillation and Expert Fusion](https://arxiv.org/pdf/2510.20651) | Arxiv | 2025.10 | None |
|DynaMix [True zero-shot inference of dynamical systems preserving long-term statistics](https://arxiv.org/pdf/2505.13192) | Arxiv | 2025.05 | [Code](https://github.com/DurstewitzLab/DynaMix-python) |
|MoFE-Time [MoFE-Time: Mixture of Frequency Domain Experts for Time-Series Forecasting Models](https://arxiv.org/pdf/2507.06502) | Arxiv | 2025.07 | None |
|PTN [Learned Data Transformation: A Data-centric Plugin for Enhancing Time Series Forecasting](https://openreview.net/pdf?id=6hJ3khuJY4) | None | 2025 | None|
|ProtoN-FM [Bridging Distribution Gaps in Time Series Foundation Model Pretraining with Prototype-Guided Normalization](https://arxiv.org/pdf/2504.10900) | Arxiv | 2025.04 | None |
|Fedformer [Fedformer: Frequency enhanced decomposed transformer for long-term series forecasting](https://proceedings.mlr.press/v162/zhou22g/zhou22g.pdf) | ICML | 2022 | [Code](https://github.com/MAZiqing/FEDformer) |
|Film [Film: Frequency improved legendre memory model for long-term time series forecasting](https://proceedings.neurips.cc/paper_files/paper/2022/file/524ef58c2bd075775861234266e5e020-Paper-Conference.pdf) | NeurIPS | 2022 | [Code](https://github.com/tianzhou2011/FiLM/) |
|MoE-KAN [Interpretable mixture of experts for time series prediction under recurrent and non-recurrent conditions](https://arxiv.org/pdf/2409.03282) | Arxiv | 2024.09 | None |



#### Heterogeneous Experts

- Statistical + neural

| Title | Venue   | Month   | Code |
| ------- | ------- | ------- | ------- |
|Dychem [Dynamic combination of heterogeneous models for hierarchical time series](https://ieeexplore.ieee.org/abstract/document/10031198) | ICDMW | 2022 | [Code](https://github.com/aaronhan223/htsf) |
|TSEPMoE [Time Series-Based Electric Load Forecasting with Mixture of Expert System](https://ieeexplore.ieee.org/abstract/document/10277743) | ICHCI | 2023 | None |

- Neural models


| Title | Venue   | Month   | Code |
| ------- | ------- | ------- | ------- |
|TALON [Adapting LLMs to Time Series Forecasting via Temporal Heterogeneity Modeling and Semantic Alignment](https://arxiv.org/pdf/2508.07195?) | Arxiv | 2025.08 | [Code](https://github.com/syrGitHub/TALON) |
|IMMOE [Improved multi-gate mixture-of-experts framework for multi-step prediction of gas load](https://www.sciencedirect.com/science/article/abs/pii/S0360544223017383) | Energy | 2023 | None |
|98 [Time series forecasting with high stakes: A field study of the air cargo industry](https://arxiv.org/pdf/2407.20192?) | Arxiv | 2024.07 | None |
|MoME [MoME: Mixture of Multi-Domain Experts for Multivariate Long-Term Series Forecasting](https://ieeexplore.ieee.org/abstract/document/10887716/) | ICASSP | 2025 | [Code](https://github.com/lxy-PhD2022/MoME) |
|99 [A Dynamic Approach to Stock Price Prediction: Comparing RNN and Mixture of Experts Models Across Different Volatility Profiles](https://arxiv.org/pdf/2410.07234) | Arxiv | 2024.10 | None |
|MoE-AMC [Moe-amc: Enhancing automatic modulation classification performance using mixture-of-experts](https://arxiv.org/pdf/2312.02298) | Arxiv | 2023.12 | None |
|MAES [Model-Attentive Ensemble Learning for Sequence Modeling](https://arxiv.org/pdf/2102.11500) | Arxiv | 2021.02 | None |
|BP-MoE [BP-MoE: Behavior Pattern-aware Mixture-of-Experts for temporal graph representation learning](https://www.sciencedirect.com/science/article/abs/pii/S0950705124006907) | KBS | 2024 | [Code](https://github.com/rekaBelbA/BP-MoE) |
|MGTS-Net [MGTS-Net: Exploring Graph-Enhanced Multimodal Fusion for Augmented Time Series Forecasting](https://arxiv.org/pdf/2510.16350) | Arxiv | 2025.10 | None |


- Foundation models


| Title | Venue   | Month   | Code |
| ------- | ------- | ------- | ------- |
|MoE-F [Filtered not Mixed: Filtering-Based Online Gating for Mixture of Large Language Models](https://openreview.net/pdf?id=ecIvumCyAj) | ICLR | 2025 | [Code](https://github.com/raeidsaqur/moe-f) |
|VPF-MoE [A vegetable-price forecasting method based on mixture of experts](https://www.mdpi.com/2077-0472/15/2/162) | Agriculture | 2025 | None |
|MSE-ITT [Multimodal Language Models with Modality-Specific Experts for Financial Forecasting from Interleaved Sequences of Text and Time Series](https://arxiv.org/pdf/2509.19628) | Arxiv | 2025.09 | [Code](https://github.com/rosskoval/mlm_text_ts) |

#### Shared Experts


| Title | Venue   | Month   | Code |
| ------- | ------- | ------- | ------- |
|MoLE [Mixture-of-linear-experts for long-term time series forecasting](https://proceedings.mlr.press/v238/ni24a/ni24a.pdf) | ICAIS | 2024 | [Code](https://github.com/RogerNi/MoLE) |
|SoftShape [Learning Soft Sparse Shapes for Efficient Time-Series Classification](https://openreview.net/pdf?id=B9DOjtj9xK) | ICML | 2025 | [Code](https://github.com/qianlima-lab/SoftShape) |
|AME [Attention with Mixture Experts model for Multivariate Time Series Imputation](https://ieeexplore.ieee.org/abstract/document/10929537) | EIECC | 2024 | None |
|Time-MoE [Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts](https://openreview.net/pdf?id=e1wDDFmlVu) | ICLR | 2025 | [Code](https://github.com/time-moe/time-moe) |
|Time Tracker [Time Tracker: Mixture-of-Experts-Enhanced Foundation Time Series Forecasting Model with Decoupled Training Pipelines](https://arxiv.org/pdf/2505.15151) | Arxiv | 2025.05 | None |
|ContexTST [Unify and Anchor: A Context-Aware Transformer for Cross-Domain Time Series Forecasting](https://arxiv.org/pdf/2503.01157?) | Arxiv | 2025.03 | None |
|PatchMoE [Unlocking the Power of Mixture-of-Experts for Task-Aware Time Series Analytics](https://arxiv.org/pdf/2509.22279?) | Arxiv | 2025.09 | [Code](https://anonymous.4open.science/r/PatchMoE-BD38/README.md) |
|ARTEMIS [Artemis: Autoregressive end-to-end trajectory planning with mixture of experts for autonomous driving](https://arxiv.org/pdf/2504.19580) | Arxiv | 2025.04 | [Code](None) |
