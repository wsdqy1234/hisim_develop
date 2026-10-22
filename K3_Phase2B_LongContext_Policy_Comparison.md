# Phase 2B Long-Context Policy Comparison

## 1. Trace Configuration

### B1: Long Context (8K / 32K)
| Parameter | B1-8K | B1-32K |
|-----------|-------|--------|
| Requests | 56 | 56 |
| Sessions | 28 (8 hot + 8 cold + 16 churn) | 28 |
| Tokens/session | 8192 | 32768 |
| Total tokens | 458K | 1.84M |
| Checkpoints | 32 | 32 |
| KDA | 16 | 16 |
| Evictions | 16 (50%) | 16 (50%) |
| Replay cost/miss | 8K | 32K |

Structure: Build(16) → Re-access hot(8) → Churn(16) → Reuse(16)

### B2: Fork Prefix
| Parameter | Value |
|-----------|-------|
| Requests | 56 |
| Sessions | 17 (1 shared + 4 L1 + 2 L2 + 16 churn - 6 overlap) |
| Shared prefix | 32K tokens |
| L1 fork point | 32K |
| L2 fork point | 36K (from L1[0]) |
| Total tokens | 2.24M |
| Checkpoints | 23 |
| KDA | 16 |
| Evictions | 7 (30%) |

### B3: Hotspot Migration
| Parameter | Value |
|-----------|-------|
| Requests | 56 |
| Sessions | 17 (8 A + 8 B + 8 churn - 7 overlap) |
| Phase A | Build A0-A7, re-access A0-A3 × 2 |
| Phase B | Build B0-B7, re-access B0-B3 × 2 |
| Churn | 8 sessions |
| Phase C | Re-access all A and B |
| Checkpoints | 24 |
| KDA | 16 |
| Evictions | 8 (33%) |

## 2. Results

(To be filled after experiments complete)

## 3. Analysis Questions

### Q1: Hotness 相对 Native 的收益在什么工作负载下最大？

### Q2: 哪些情况下 Hotness 会退化？

### Q3: Oracle 与 Hotness 的差距主要来自什么？

### Q4: Phase 2C 应优先研究什么？
