## Single user

| approach / config | prompt | weights in RAM (GB) | stage-pod RAM (GB) | coordinator RAM (GB) | total RAM (GB) | vs baseline | TTFT (s) | TTFT × | per token (ms) | per token × | output |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline: all 36 layers in one pod | 128 | 6.17 | 5.96 | 1.01 | 6.98 | +0% | 0.44 | 1.00 | 122 | 1.00 | identical |
| baseline: all 36 layers in one pod | 512 | 6.17 | 6.05 | 1.02 | 7.07 | +0% | 1.40 | 1.00 | 133 | 1.00 | identical |
| baseline: all 36 layers in one pod | 2048 | 6.17 | 6.88 | 1.06 | 7.94 | +0% | 6.80 | 1.00 | 214 | 1.00 | identical |
| baseline: all 36 layers in one pod [non-repeating prompt] | 512 | 6.17 | 6.05 | 1.02 | 7.07 | +0% | 1.26 | 1.00 | 133 | 1.00 | identical |
| cyclic: 2 pods, 1 layer each | 128 | 0.93 | 0.99 | 1.02 | 2.00 | -71% | 1.23 | 2.81 | 1001 | 8.22 | identical |
| cyclic: 3 pods, 1 layer each | 128 | 1.08 | 1.40 | 1.02 | 2.42 | -65% | 0.95 | 2.17 | 824 | 6.77 | identical |
| cyclic: 4 pods, 1 layer each | 128 | 1.24 | 1.81 | 1.01 | 2.82 | -60% | 0.91 | 2.09 | 783 | 6.43 | identical |
| cyclic: 6 pods, 1 layer each | 128 | 1.55 | 2.64 | 1.02 | 3.65 | -48% | 1.01 | 2.30 | 771 | 6.33 | identical |
| cyclic: 9 pods, 1 layer each | 128 | 2.01 | 3.86 | 1.02 | 4.88 | -30% | 0.95 | 2.18 | 769 | 6.32 | identical |
| cyclic: 12 pods, 1 layer each | 128 | 2.47 | 5.10 | 1.01 | 6.12 | -12% | 1.01 | 2.31 | 769 | 6.31 | identical |
| cyclic: 18 pods, 1 layer each | 128 | 3.40 | 7.57 | 1.02 | 8.59 | +23% | 1.01 | 2.31 | 785 | 6.45 | identical |
| cyclic: 2 pods, 1 layer each | 512 | 0.93 | 1.24 | 1.02 | 2.26 | -68% | 1.88 | 1.34 | 1013 | 7.61 | identical |
| cyclic: 3 pods, 1 layer each | 512 | 1.08 | 1.71 | 1.02 | 2.73 | -61% | 1.64 | 1.18 | 822 | 6.18 | identical |
| cyclic: 4 pods, 1 layer each | 512 | 1.24 | 2.23 | 1.02 | 3.25 | -54% | 2.06 | 1.47 | 781 | 5.87 | identical |
| cyclic: 6 pods, 1 layer each | 512 | 1.55 | 3.24 | 1.02 | 4.26 | -40% | 2.08 | 1.49 | 772 | 5.80 | identical |
| cyclic: 9 pods, 1 layer each | 512 | 2.01 | 4.71 | 1.02 | 5.73 | -19% | 2.00 | 1.43 | 770 | 5.79 | identical |
| cyclic: 12 pods, 1 layer each | 512 | 2.47 | 6.22 | 1.02 | 7.23 | +2% | 2.07 | 1.48 | 774 | 5.82 | identical |
| cyclic: 18 pods, 1 layer each | 512 | 3.40 | 9.22 | 1.02 | 10.24 | +45% | 2.05 | 1.47 | 769 | 5.78 | identical |
| cyclic: 2 pods, 1 layer each | 2048 | 0.93 | 2.87 | 1.04 | 3.91 | -51% | 6.66 | 0.98 | 1069 | 4.99 | identical |
| cyclic: 3 pods, 1 layer each | 2048 | 1.08 | 4.07 | 1.04 | 5.11 | -36% | 7.02 | 1.03 | 860 | 4.02 | identical |
| cyclic: 4 pods, 1 layer each | 2048 | 1.24 | 5.30 | 1.06 | 6.36 | -20% | 7.36 | 1.08 | 791 | 3.69 | identical |
| cyclic: 6 pods, 1 layer each | 2048 | 1.55 | 7.74 | 1.04 | 8.79 | +11% | 7.54 | 1.11 | 776 | 3.62 | identical |
| cyclic: 9 pods, 1 layer each | 2048 | 2.01 | 11.47 | 1.05 | 12.52 | +58% | 7.62 | 1.12 | 767 | 3.58 | identical |
| cyclic: 12 pods, 1 layer each | 2048 | 2.47 | 15.00 | 1.04 | 16.03 | +102% | 7.47 | 1.10 | 769 | 3.59 | identical |
| cyclic: 18 pods, 1 layer each | 2048 | 3.40 | 22.48 | 1.04 | 23.52 | +196% | 7.67 | 1.13 | 765 | 3.57 | identical |
| cyclic: 2 pods, 1 layer each [non-repeating prompt] | 512 | 0.93 | 1.23 | 1.02 | 2.24 | -68% | 1.82 | 1.44 | 1029 | 7.75 | identical |
| cyclic: 4 pods, 1 layer each [non-repeating prompt] | 512 | 1.24 | 2.20 | 1.02 | 3.22 | -54% | 1.68 | 1.33 | 786 | 5.92 | identical |
| single pod: all 36 layers streamed in order through 1 slot | 128 | 0.78 | 0.57 | 1.02 | 1.58 | -77% | 1.99 | 4.56 | 1646 | 13.51 | identical |
| single pod: all 36 layers streamed in order through 2 slots | 128 | 0.93 | 0.72 | 1.01 | 1.73 | -75% | 1.25 | 2.85 | 1036 | 8.51 | identical |
| single pod: all 36 layers streamed in order through 3 slots | 128 | 1.08 | 0.87 | 1.01 | 1.89 | -73% | 1.01 | 2.32 | 866 | 7.11 | identical |
| single pod: all 36 layers streamed in order through 4 slots | 128 | 1.24 | 1.03 | 1.02 | 2.04 | -71% | 0.98 | 2.25 | 798 | 6.55 | identical |
| single pod: all 36 layers streamed in order through 6 slots | 128 | 1.24 | 1.27 | 1.02 | 2.29 | -67% | 0.94 | 2.16 | 777 | 6.38 | identical |
| single pod: all 36 layers streamed in order through 9 slots | 128 | 1.86 | 1.76 | 1.02 | 2.77 | -60% | 0.97 | 2.22 | 781 | 6.41 | identical |
| single pod: all 36 layers streamed in order through 1 slot | 512 | 0.78 | 0.66 | 1.03 | 1.69 | -76% | 2.85 | 2.03 | 1664 | 12.51 | identical |
| single pod: all 36 layers streamed in order through 2 slots | 512 | 0.93 | 0.83 | 1.02 | 1.85 | -74% | 1.88 | 1.35 | 1044 | 7.85 | identical |
| single pod: all 36 layers streamed in order through 3 slots | 512 | 1.08 | 0.97 | 1.02 | 1.99 | -72% | 1.90 | 1.36 | 868 | 6.52 | identical |
| single pod: all 36 layers streamed in order through 4 slots | 512 | 1.24 | 1.11 | 1.02 | 2.14 | -70% | 2.00 | 1.43 | 798 | 6.00 | identical |
| single pod: all 36 layers streamed in order through 6 slots | 512 | 1.55 | 1.42 | 1.02 | 2.44 | -66% | 1.94 | 1.39 | 778 | 5.85 | identical |
| single pod: all 36 layers streamed in order through 9 slots | 512 | 2.01 | 1.91 | 1.02 | 2.93 | -59% | 2.02 | 1.44 | 784 | 5.89 | identical |
| single pod: all 36 layers streamed in order through 1 slot | 2048 | 0.78 | 1.43 | 1.05 | 2.48 | -69% | 8.11 | 1.19 | 1714 | 8.00 | identical |
| single pod: all 36 layers streamed in order through 2 slots | 2048 | 0.93 | 1.60 | 1.04 | 2.63 | -67% | 6.97 | 1.03 | 1075 | 5.02 | identical |
| single pod: all 36 layers streamed in order through 3 slots | 2048 | 1.08 | 1.74 | 1.04 | 2.78 | -65% | 7.71 | 1.13 | 924 | 4.31 | identical |
| single pod: all 36 layers streamed in order through 4 slots | 2048 | 1.24 | 1.88 | 1.05 | 2.92 | -63% | 7.79 | 1.15 | 809 | 3.78 | identical |
| single pod: all 36 layers streamed in order through 6 slots | 2048 | 1.55 | 2.19 | 1.04 | 3.22 | -59% | 7.69 | 1.13 | 780 | 3.64 | identical |
| single pod: all 36 layers streamed in order through 9 slots | 2048 | 2.01 | 2.68 | 1.04 | 3.72 | -53% | 7.72 | 1.14 | 786 | 3.67 | identical |
| partial: 18 layers pinned, 18 streamed through 4 slots (−14 layers) | 128 | 4.01 | 3.80 | 1.02 | 4.82 | -31% | 0.64 | 1.47 | 460 | 3.77 | identical |
| partial: 24 layers pinned, 12 streamed through 4 slots (−8 layers) | 128 | 4.94 | 4.73 | 1.02 | 5.74 | -18% | 0.61 | 1.39 | 332 | 2.73 | identical |
| partial: 27 layers pinned, 9 streamed through 3 slots (−6 layers) | 128 | 5.25 | 5.04 | 1.01 | 6.05 | -13% | 0.55 | 1.26 | 288 | 2.37 | identical |
| partial: 30 layers pinned, 6 streamed through 3 slots (−3 layers) | 128 | 5.71 | 5.50 | 1.02 | 6.51 | -7% | 0.51 | 1.17 | 226 | 1.86 | identical |
| partial: 33 layers pinned, 3 streamed through 3 slots (−0 layers) | 128 | 6.17 | 5.96 | 1.02 | 6.98 | +0% | 0.53 | 1.21 | 160 | 1.32 | identical |
| partial: 18 layers pinned, 18 streamed through 4 slots (−14 layers) | 512 | 4.01 | 3.90 | 1.02 | 4.92 | -30% | 1.67 | 1.19 | 478 | 3.59 | identical |
| partial: 24 layers pinned, 12 streamed through 4 slots (−8 layers) | 512 | 4.94 | 4.82 | 1.02 | 5.84 | -17% | 1.51 | 1.08 | 339 | 2.55 | identical |
| partial: 27 layers pinned, 9 streamed through 3 slots (−6 layers) | 512 | 5.25 | 5.12 | 1.02 | 6.14 | -13% | 1.39 | 1.00 | 298 | 2.24 | identical |
| partial: 30 layers pinned, 6 streamed through 3 slots (−3 layers) | 512 | 5.71 | 5.58 | 1.02 | 6.60 | -7% | 1.52 | 1.09 | 235 | 1.76 | identical |
| partial: 33 layers pinned, 3 streamed through 3 slots (−0 layers) | 512 | 6.17 | 6.06 | 1.02 | 7.08 | +0% | 1.45 | 1.04 | 176 | 1.33 | identical |
| partial: 18 layers pinned, 18 streamed through 4 slots (−14 layers) | 2048 | 4.01 | 4.63 | 1.04 | 5.67 | -29% | 7.36 | 1.08 | 517 | 2.41 | identical |
| partial: 24 layers pinned, 12 streamed through 4 slots (−8 layers) | 2048 | 4.94 | 5.59 | 1.04 | 6.63 | -17% | 6.44 | 0.95 | 398 | 1.86 | identical |
| partial: 27 layers pinned, 9 streamed through 3 slots (−6 layers) | 2048 | 5.25 | 5.89 | 1.04 | 6.93 | -13% | 6.45 | 0.95 | 376 | 1.76 | identical |
| partial: 30 layers pinned, 6 streamed through 3 slots (−3 layers) | 2048 | 5.71 | 6.42 | 1.04 | 7.46 | -6% | 6.51 | 0.96 | 296 | 1.38 | identical |
| partial: 33 layers pinned, 3 streamed through 3 slots (−0 layers) | 2048 | 6.17 | 6.83 | 1.04 | 7.87 | -1% | 6.66 | 0.98 | 241 | 1.13 | identical |
| mmap + cgroup limit 1.5Gi | 128 | mapped; cgroup peak 1.61 | 1.68 | 1.02 | 2.69 | -61% | 1.69 | 3.85 | 1399 | 11.48 | identical |
| mmap + cgroup limit 2Gi | 128 | mapped; cgroup peak 2.15 | 2.21 | 1.02 | 3.23 | -54% | 1.61 | 3.68 | 1377 | 11.30 | identical |
| mmap + cgroup limit 3Gi | 128 | mapped; cgroup peak 3.22 | 3.28 | 1.02 | 4.30 | -38% | 1.63 | 3.72 | 1327 | 10.90 | identical |
| mmap + cgroup limit 4Gi | 128 | mapped; cgroup peak 4.29 | 4.36 | 1.02 | 5.37 | -23% | 1.56 | 3.58 | 1069 | 8.78 | identical |
| mmap + cgroup limit 6Gi | 128 | mapped; cgroup peak 5.93 | 5.96 | 1.02 | 6.98 | -0% | 0.42 | 0.95 | 122 | 1.00 | identical |
| mmap + cgroup limit 1.5Gi | 512 | mapped; cgroup peak 1.61 | 1.68 | 1.02 | 2.70 | -62% | 2.73 | 1.95 | 1455 | 10.94 | identical |
| mmap + cgroup limit 2Gi | 512 | mapped; cgroup peak 2.15 | 2.21 | 1.02 | 3.23 | -54% | 2.63 | 1.88 | 1413 | 10.62 | identical |
| mmap + cgroup limit 3Gi | 512 | mapped; cgroup peak 3.22 | 3.28 | 1.03 | 4.31 | -39% | 2.66 | 1.90 | 1399 | 10.51 | identical |
| mmap + cgroup limit 4Gi | 512 | mapped; cgroup peak 4.29 | 4.36 | 1.02 | 5.38 | -24% | 2.24 | 1.60 | 1029 | 7.73 | identical |
| mmap + cgroup limit 6Gi | 512 | mapped; cgroup peak 6.00 | 6.05 | 1.02 | 7.07 | -0% | 1.28 | 0.91 | 131 | 0.99 | identical |
| mmap + cgroup limit 1.5Gi | 2048 | mapped; cgroup peak 1.61 | 1.68 | 1.06 | 2.73 | -66% | 8.45 | 1.24 | 1872 | 8.74 | identical |
| mmap + cgroup limit 2Gi | 2048 | mapped; cgroup peak 2.15 | 2.21 | 1.06 | 3.27 | -59% | 8.72 | 1.28 | 1834 | 8.56 | identical |
| mmap + cgroup limit 3Gi | 2048 | mapped; cgroup peak 3.22 | 3.29 | 1.06 | 4.35 | -45% | 8.26 | 1.22 | 1877 | 8.76 | identical |
| mmap + cgroup limit 4Gi | 2048 | mapped; cgroup peak 4.30 | 4.36 | 1.04 | 5.40 | -32% | 8.36 | 1.23 | 1690 | 7.89 | identical |
| mmap + cgroup limit 6Gi | 2048 | mapped; cgroup peak 6.44 | 6.50 | 1.04 | 7.54 | -5% | 7.18 | 1.06 | 220 | 1.03 | identical |
| speculative γ=5 on all-resident pod | 128 | 7.16 | 5.96 | 2.09 | 8.05 | +15% | 0.45 | 1.03 | 98 | 0.81 | differs from token 15 |
| speculative γ=5 on cyclic 2 pods | 128 | 1.92 | 0.99 | 2.08 | 3.06 | -56% | 1.35 | 3.09 | 400 | 3.29 | differs from token 15 |
| speculative γ=5 on cyclic 4 pods | 128 | 2.23 | 1.81 | 2.08 | 3.89 | -44% | 0.95 | 2.17 | 327 | 2.69 | differs from token 15 |
| speculative γ=8 on cyclic 4 pods | 128 | 2.23 | 1.81 | 2.08 | 3.90 | -44% | 0.97 | 2.22 | 318 | 2.61 | differs from token 15 |
| speculative γ=5 on all-resident pod | 512 | 7.16 | 6.05 | 2.26 | 8.31 | +17% | 1.27 | 0.91 | 63 | 0.47 | identical |
| speculative γ=5 on cyclic 2 pods | 512 | 1.92 | 1.26 | 2.23 | 3.49 | -51% | 2.22 | 1.59 | 221 | 1.66 | identical |
| speculative γ=5 on cyclic 4 pods | 512 | 2.23 | 2.25 | 2.28 | 4.53 | -36% | 2.00 | 1.43 | 179 | 1.34 | identical |
| speculative γ=8 on cyclic 4 pods | 512 | 2.23 | 2.25 | 2.31 | 4.56 | -36% | 2.15 | 1.54 | 136 | 1.02 | identical |
| speculative γ=5 on all-resident pod | 2048 | 7.16 | 6.80 | 2.90 | 9.70 | +22% | 7.20 | 1.06 | 139 | 0.65 | identical |
| speculative γ=5 on cyclic 2 pods | 2048 | 1.92 | 2.80 | 2.75 | 5.55 | -30% | 7.73 | 1.14 | 296 | 1.38 | identical |
| speculative γ=5 on cyclic 4 pods | 2048 | 2.23 | 5.26 | 2.95 | 8.20 | +3% | 7.65 | 1.12 | 247 | 1.15 | identical |
| speculative γ=8 on cyclic 4 pods | 2048 | 2.23 | 5.34 | 2.90 | 8.24 | +4% | 7.70 | 1.13 | 205 | 0.96 | identical |
| speculative γ=5 on all-resident pod [non-repeating prompt] | 512 | 7.16 | 6.05 | 2.30 | 8.35 | +18% | 1.32 | 1.05 | 120 | 0.91 | identical |
| speculative γ=5 on cyclic 2 pods [non-repeating prompt] | 512 | 1.92 | 1.23 | 2.25 | 3.48 | -51% | 2.09 | 1.66 | 441 | 3.33 | identical |
| speculative γ=5 on cyclic 4 pods [non-repeating prompt] | 512 | 2.23 | 2.21 | 2.26 | 4.47 | -37% | 2.03 | 1.61 | 365 | 2.75 | identical |
| speculative γ=8 on cyclic 4 pods [non-repeating prompt] | 512 | 2.23 | 2.23 | 2.30 | 4.53 | -36% | 1.96 | 1.55 | 371 | 2.79 | identical |

## Speculative decoding details (medians per cycle)

| config | prompt | accepted drafts / cycle | tokens / cycle | draft (ms) | verify (ms) | cycle (ms) | per token (ms) |
|---|---|---|---|---|---|---|---|
| speculative γ=5 on all-resident pod | 128 | 2.00 | 3.00 | 155 | 126 | 283 | 98 |
| speculative γ=5 on cyclic 2 pods | 128 | 2.00 | 3.00 | 178 | 1012 | 1192 | 400 |
| speculative γ=5 on cyclic 4 pods | 128 | 2.00 | 3.00 | 191 | 778 | 973 | 327 |
| speculative γ=8 on cyclic 4 pods | 128 | 2.40 | 3.40 | 287 | 782 | 1076 | 318 |
| speculative γ=5 on all-resident pod | 512 | 5.00 | 6.00 | 166 | 140 | 309 | 63 |
| speculative γ=5 on cyclic 2 pods | 512 | 5.00 | 6.00 | 215 | 1042 | 1260 | 221 |
| speculative γ=5 on cyclic 4 pods | 512 | 5.00 | 6.00 | 215 | 793 | 1006 | 179 |
| speculative γ=8 on cyclic 4 pods | 512 | 8.00 | 9.00 | 381 | 785 | 1166 | 136 |
| speculative γ=5 on all-resident pod | 2048 | 5.00 | 6.00 | 249 | 200 | 459 | 139 |
| speculative γ=5 on cyclic 2 pods | 2048 | 5.00 | 6.00 | 308 | 1094 | 1404 | 296 |
| speculative γ=5 on cyclic 4 pods | 2048 | 5.00 | 6.00 | 314 | 800 | 1105 | 247 |
| speculative γ=8 on cyclic 4 pods | 2048 | 8.00 | 9.00 | 480 | 805 | 1294 | 205 |
| speculative γ=5 on all-resident pod [non-repeating prompt] | 512 | 1.82 | 2.82 | 167 | 137 | 309 | 120 |
| speculative γ=5 on cyclic 2 pods [non-repeating prompt] | 512 | 1.82 | 2.82 | 194 | 1008 | 1209 | 441 |
| speculative γ=5 on cyclic 4 pods [non-repeating prompt] | 512 | 1.82 | 2.82 | 212 | 778 | 994 | 365 |
| speculative γ=8 on cyclic 4 pods [non-repeating prompt] | 512 | 2.10 | 3.10 | 321 | 786 | 1114 | 371 |

## Batching (128-token prompt)

| config | batch | total RAM (GB) | TTFT (s) | per-user ms/token | vs baseline at same batch | tokens/s | output (row 0) |
|---|---|---|---|---|---|---|---|
| baseline: all 36 layers in one pod | 1 | 6.98 | 0.44 | 122 | 1.00× | 8 | identical |
| cyclic: 2 pods, 1 layer each | 1 | 2.00 | 1.23 | 1001 | 8.22× | 1 | identical |
| cyclic: 4 pods, 1 layer each | 1 | 2.82 | 0.91 | 783 | 6.43× | 1 | identical |
| batch 16: baseline | 16 | 7.30 | 4.25 | 188 | 1.00× | 85 | differs from token 15 |
| batch 64: baseline | 64 | 8.36 | 16.36 | 541 | 1.00× | 118 | differs from token 15 |
| batch 256: baseline | 256 | 12.78 | 63.30 | 1897 | 1.00× | 135 | differs from token 15 |
| batch 512: baseline | 512 | 18.66 | 124.39 | 3418 | 1.00× | 150 | differs from token 15 |
| batch 64: cyclic 2 pods | 64 | 4.37 | 17.32 | 1366 | 2.53× | 47 | differs from token 15 |
| batch 256: cyclic 2 pods | 256 | 11.49 | 64.67 | 2582 | 1.36× | 99 | differs from token 15 |
| batch 16: cyclic 4 pods | 16 | 4.26 | 4.80 | 795 | 4.22× | 20 | differs from token 15 |
| batch 64: cyclic 4 pods | 64 | 7.08 | 17.23 | 1015 | 1.88× | 63 | differs from token 15 |
| batch 256: cyclic 4 pods | 256 | 19.59 | 64.56 | 2475 | 1.30× | 103 | differs from token 15 |
| batch 512: cyclic 4 pods | 512 | 36.23 | 126.87 | 4058 | 1.19× | 126 | differs from token 15 |
