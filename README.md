# 🚦 PalemTraffic V3
## A Hybrid Spatio-Temporal Benchmark Dataset for Urban Traffic Forecasting

[![License](https://img.shields.io/badge/license-CC%20BY%204.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-green.svg)]()
[![Dataset](https://img.shields.io/badge/Dataset-Spatio--Temporal-orange.svg)]
[![Nodes](https://img.shields.io/badge/Nodes-113-red.svg)]
[![Timesteps](https://img.shields.io/badge/Timesteps-8640-purple.svg)]

---

## Overview

**PalemTraffic V3** is a hybrid spatio-temporal traffic forecasting dataset representing an urban road network in **Palembang, Indonesia**.

The dataset is designed for research on:

- Traffic forecasting
- Graph Neural Networks (GNNs)
- Spatio-Temporal Graph Neural Networks (STGNNs)
- Graph Transformers
- Intelligent Transportation Systems (ITS)
- Urban computing
- Smart-city applications

PalemTraffic V3 combines an urban road-network topology with traffic-demand patterns designed to reproduce important characteristics of real-world urban traffic, including:

- Different traffic regimes between weekdays and weekends
- Road-type-dependent traffic volumes
- Strong morning and evening commuter peaks
- Day-to-day demand variation
- Node-specific traffic characteristics
- Spatial correlation between connected roads
- Traffic/congestion propagation through neighboring roads
- Different vehicle-type compositions
- Low nighttime traffic on residential roads
- Correlated but non-identical dynamics between neighboring road segments

The dataset is **semi-synthetic**: the road network and node definitions represent the Palembang urban environment, while the traffic time series are generated using a calibrated traffic-demand model informed by observed traffic characteristics and domain assumptions. 20 days of traffic is generated manually, while 10 days of traffic is syntheticaly generated.

This design provides a controlled benchmark while maintaining realistic temporal and spatial traffic behavior.

---

# Dataset Highlights

| Property | Value |
|---|---|
| Dataset | **PalemTraffic** |
| City | Palembang, Indonesia |
| Road Nodes | **113** |
| Main Road Nodes | **31** |
| Residential Road Nodes | **82** |
| Time Interval | **5 minutes** |
| Duration | **30 consecutive days** |
| Total Timesteps | **8,640** |
| Observations | **976,320 node-time observations** |
| Vehicle Categories | **8** |
| Tensor Features | **11** |
| Graph | Urban road connectivity graph |
| Data Type | `float32` |
The total number of node-time observations is:

```text
8,640 timesteps × 113 nodes
= 976,320 observations
# Citation

If you use PalemTraffic in your research, please cite:

```bibtex
@article{fajri2026palemtraffic,
  title={PalemTraffic: A Novel Hybrid Benchmark Dataset for Spatio-Temporal
Graph Neural Networks in Urban Traffic Forecasting},
  author={Fajri, Ricky Maulana and others},
  journal={Under Review},
  year={2026}
}
```

---

# License

This dataset is released under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** License.

You are free to:

- Share
- Adapt
- Redistribute

provided appropriate credit is given.

---

# Contact

**Ricky Maulana Fajri**

Faculty of Computer Science, Universitas Indo Global Mandiri

Palembang, Indonesia

Email: *rickymaulanafajri@uigm.ac.id*

---

# Acknowledgement

This dataset was developed as part of ongoing research on spatio-temporal graph neural networks for intelligent transportation systems.

The authors gratefully acknowledge the support of **Direktorat Penelitian dan Pengabdian kepada Masyarakat (DPPM), Direktorat Jenderal Riset dan Pengembangan, Kementerian Pendidikan Tinggi, Sains, dan Teknologi Republik Indonesia**, for supporting this research.
