# 🚦 PalemTraffic: A Novel Hybrid Benchmark Dataset for Spatio-Temporal Graph Neural Networks in Urban Traffic Forecasting

[![License](https://img.shields.io/badge/license-CC%20BY%204.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-green.svg)]()
[![Dataset](https://img.shields.io/badge/Dataset-Spatio--Temporal-orange.svg)]()

## Overview

**PalemTraffic** is a hybrid spatio-temporal traffic forecasting dataset collected from the urban road network of **Palembang, Indonesia**. The dataset was developed to support research on **traffic forecasting**, **Graph Neural Networks (GNNs)**, **Spatio-Temporal Graph Neural Networks (STGNNs)**, and **Intelligent Transportation Systems (ITS)**.

Unlike conventional benchmark datasets collected from expensive loop detectors or traffic sensors, PalemTraffic combines:

- Field observations
- Video-based vehicle detection
- Automatic vehicle counting
- Official transportation statistics
- Expert interviews
- Semi-synthetic simulation

to produce a realistic traffic dataset suitable for deep learning research.

---

# Dataset Highlights

| Property | Value |
|----------|------|
| City | Palembang, Indonesia |
| Road Segments | **113** |
| Time Interval | **5 minutes** |
| Duration | **30 consecutive days** |
| Total Timesteps | **8,640** |
| Graph Type | Directed road network |
| Data Type | Multivariate Traffic Flow |
| Vehicle Categories | **8** |
| Spatial Representation | Graph |
| Forecasting Task | Multistep Traffic Forecasting |

---

# Vehicle Categories

Traffic flow is recorded separately for eight vehicle classes.

| Channel | Vehicle Type |
|----------|-------------|
| 1 | Motorcycle |
| 2 | Passenger Car |
| 3 | Pickup |
| 4 | Minibus |
| 5 | Bus |
| 6 | Truck |
| 7 | Trailer |
| 8 | Bicycle / Non-motorized Vehicle |

---

# Data Shape

Traffic tensor:

```python
(T, N, F)

where

T = 8640 timesteps
N = 113 road segments
F = traffic features
```

Example

```python
traffic.shape

(8640, 113, 8)
```

where

- 8640 = 30 days × 24 hours × 12 intervals/hour
- 113 = road segments
- 8 = vehicle categories

---

# Graph Representation

Each node represents a road segment.

Edges represent road connectivity.

```
Node
   │
Road Segment

Edge
   │
Physical road connection
```

The graph is represented by an adjacency matrix

```python
A ∈ ℝ^(113 × 113)
```

which can be directly used in Graph Neural Networks.

---

# Forecasting Task

Given historical observations

```
X(t−11), ..., X(t)
```

predict future traffic

```
X(t+1), ..., X(t+12)
```

The dataset supports

- Short-term forecasting
- Multi-step forecasting
- Node-level prediction
- Network-wide prediction

---

# Supported Models

PalemTraffic can be used with most modern traffic forecasting models, including

- GraphWaveNet
- DCRNN
- STGCN
- ASTGCN
- AGCRN
- STAEformer
- STEP
- STID
- STGAT
- GCN-LSTM
- GraphSAGE-LSTM
- Any custom GNN or Transformer model

---

# Example Loading

```python
import numpy as np

data = np.load("traffic_data.npz")

traffic = data["traffic"]

print(traffic.shape)
```

---

# Benchmark Split

A commonly used split is

| Split | Percentage |
|--------|-----------|
| Training | 70% |
| Validation | 10% |
| Testing | 20% |

Researchers may use alternative splits depending on their experiments.

---

# Research Applications

PalemTraffic is suitable for research in

- Traffic Forecasting
- Graph Neural Networks
- Spatio-Temporal Graph Learning
- Intelligent Transportation Systems
- Smart Cities
- Urban Computing
- Transportation Analytics
- Time Series Forecasting

---

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
