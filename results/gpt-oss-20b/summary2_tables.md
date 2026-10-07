## Single user

| approach / config | prompt | weights in RAM (GB) | stage-pod RAM (GB) | coordinator RAM (GB) | total RAM (GB) | vs baseline | TTFT (s) | TTFT × | per token (ms) | per token × | output |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline: all 36 layers in one pod | 128 | 41.83 | 40.08 | 2.90 | 42.97 | +0% | 1.14 | 1.00 | 159 | 1.00 | identical |
| baseline: all 36 layers in one pod | 512 | 41.83 | 40.40 | 2.91 | 43.31 | +0% | 2.90 | 1.00 | 168 | 1.00 | identical |
| baseline: all 36 layers in one pod | 2048 | 41.83 | 43.21 | 2.94 | 46.15 | +0% | 10.46 | 1.00 | 176 | 1.00 | identical |
| cyclic: 2 pods, 1 layer each | 128 | 5.61 | 4.18 | 2.88 | 7.06 | -84% | 7.78 | 6.80 | 7075 | 44.38 | identical |
| cyclic: 3 pods, 1 layer each | 128 | 7.26 | 6.13 | 2.89 | 9.01 | -79% | 6.46 | 5.64 | 6142 | 38.53 | identical |
| cyclic: 4 pods, 1 layer each | 128 | 8.90 | 8.07 | 2.89 | 10.96 | -74% | 5.66 | 4.95 | 5502 | 34.52 | identical |
| cyclic: 6 pods, 1 layer each | 128 | 12.19 | 11.93 | 2.89 | 14.82 | -66% | 5.44 | 4.75 | 5493 | 34.46 | identical |
| cyclic: 8 pods, 1 layer each | 128 | 15.49 | 15.76 | 2.89 | 18.65 | -57% | 5.41 | 4.73 | 5509 | 34.56 | identical |
| cyclic: 12 pods, 1 layer each | 128 | 22.07 | 23.45 | 2.89 | 26.34 | -39% | 5.27 | 4.60 | 5509 | 34.56 | identical |
| cyclic: 2 pods, 1 layer each | 512 | 5.61 | 4.74 | 2.89 | 7.63 | -82% | 8.89 | 3.07 | 7087 | 42.12 | identical |
| cyclic: 3 pods, 1 layer each | 512 | 7.26 | 6.99 | 2.89 | 9.88 | -77% | 7.62 | 2.63 | 5772 | 34.30 | identical |
| cyclic: 4 pods, 1 layer each | 512 | 8.90 | 9.15 | 2.90 | 12.05 | -72% | 6.42 | 2.22 | 5636 | 33.49 | identical |
| cyclic: 6 pods, 1 layer each | 512 | 12.19 | 13.39 | 2.90 | 16.29 | -62% | 6.47 | 2.23 | 5499 | 32.68 | identical |
| cyclic: 8 pods, 1 layer each | 512 | 15.49 | 17.64 | 2.91 | 20.54 | -53% | 6.23 | 2.15 | 5463 | 32.47 | identical |
| cyclic: 12 pods, 1 layer each | 512 | 22.07 | 26.02 | 2.90 | 28.92 | -33% | 6.78 | 2.34 | 5533 | 32.89 | identical |
| cyclic: 2 pods, 1 layer each | 2048 | 5.61 | 9.91 | 2.92 | 12.83 | -72% | 14.32 | 1.37 | 7107 | 40.29 | identical |
| cyclic: 3 pods, 1 layer each | 2048 | 7.26 | 14.13 | 2.92 | 17.05 | -63% | 14.42 | 1.38 | 5975 | 33.87 | identical |
| cyclic: 4 pods, 1 layer each | 2048 | 8.90 | 18.85 | 2.92 | 21.77 | -53% | 14.37 | 1.37 | 5548 | 31.45 | identical |
| cyclic: 6 pods, 1 layer each | 2048 | 12.19 | 27.85 | 2.93 | 30.77 | -33% | 14.47 | 1.38 | 5445 | 30.86 | identical |
| cyclic: 8 pods, 1 layer each | 2048 | 15.49 | 36.44 | 2.94 | 39.38 | -15% | 14.78 | 1.41 | 5516 | 31.27 | identical |
| cyclic: 12 pods, 1 layer each | 2048 | 22.07 | 54.13 | 2.95 | 57.08 | +24% | 14.78 | 1.41 | 5470 | 31.01 | identical |
| single pod: all 24 layers streamed in order through 1 slot | 128 | 3.96 | 2.21 | 2.89 | 5.10 | -88% | 13.19 | 11.52 | 12271 | 76.98 | identical |
| single pod: all 24 layers streamed in order through 2 slots | 128 | 5.61 | 4.06 | 2.89 | 6.95 | -84% | 7.83 | 6.84 | 7145 | 44.83 | identical |
| single pod: all 24 layers streamed in order through 3 slots | 128 | 7.26 | 5.50 | 2.89 | 8.39 | -80% | 6.59 | 5.76 | 6221 | 39.03 | identical |
| single pod: all 24 layers streamed in order through 4 slots | 128 | 5.61 | 7.01 | 2.89 | 9.90 | -77% | 5.86 | 5.12 | 5522 | 34.64 | identical |
| single pod: all 24 layers streamed in order through 6 slots | 128 | 7.26 | 7.82 | 2.89 | 10.71 | -75% | 5.75 | 5.02 | 5634 | 35.35 | identical |
| single pod: all 24 layers streamed in order through 1 slot | 512 | 3.96 | 2.55 | 2.89 | 5.44 | -87% | 15.15 | 5.23 | 12362 | 73.47 | identical |
| single pod: all 24 layers streamed in order through 2 slots | 512 | 5.61 | 4.44 | 2.89 | 7.34 | -83% | 9.03 | 3.12 | 7179 | 42.67 | identical |
| single pod: all 24 layers streamed in order through 3 slots | 512 | 7.26 | 5.83 | 2.89 | 8.73 | -80% | 7.70 | 2.66 | 5945 | 35.33 | identical |
| single pod: all 24 layers streamed in order through 4 slots | 512 | 5.61 | 7.64 | 2.89 | 10.53 | -76% | 6.51 | 2.25 | 5563 | 33.06 | identical |
| single pod: all 24 layers streamed in order through 6 slots | 512 | 8.90 | 9.67 | 2.90 | 12.57 | -71% | 6.84 | 2.36 | 5526 | 32.84 | identical |
| single pod: all 24 layers streamed in order through 1 slot | 2048 | 3.96 | 5.13 | 2.92 | 8.05 | -83% | 21.77 | 2.08 | 12427 | 70.44 | identical |
| single pod: all 24 layers streamed in order through 2 slots | 2048 | 5.61 | 7.08 | 2.92 | 10.00 | -78% | 15.21 | 1.45 | 7190 | 40.76 | identical |
| single pod: all 24 layers streamed in order through 3 slots | 2048 | 7.26 | 8.18 | 2.92 | 11.10 | -76% | 14.36 | 1.37 | 6258 | 35.47 | identical |
| single pod: all 24 layers streamed in order through 4 slots | 2048 | 8.90 | 9.59 | 2.92 | 12.51 | -73% | 14.93 | 1.43 | 5657 | 32.07 | identical |
| single pod: all 24 layers streamed in order through 6 slots | 2048 | 10.55 | 13.15 | 2.93 | 16.08 | -65% | 14.63 | 1.40 | 5554 | 31.48 | identical |
| partial: 12 layers pinned, 12 streamed through 4 slots (−8 layers) | 128 | 28.66 | 26.91 | 2.89 | 29.80 | -31% | 3.43 | 3.00 | 2799 | 17.56 | identical |
| partial: 16 layers pinned, 8 streamed through 4 slots (−4 layers) | 128 | 35.24 | 33.49 | 2.89 | 36.38 | -15% | 2.50 | 2.18 | 1867 | 11.71 | identical |
| partial: 18 layers pinned, 6 streamed through 3 slots (−3 layers) | 128 | 36.89 | 35.13 | 2.89 | 38.02 | -12% | 2.38 | 2.08 | 1545 | 9.69 | identical |
| partial: 20 layers pinned, 4 streamed through 2 slots (−2 layers) | 128 | 38.54 | 36.78 | 2.89 | 39.67 | -8% | 2.13 | 1.86 | 1228 | 7.71 | identical |
| partial: 22 layers pinned, 2 streamed through 2 slots (−0 layers) | 128 | 41.83 | 40.07 | 2.89 | 42.96 | -0% | 1.61 | 1.41 | 643 | 4.03 | identical |
| partial: 12 layers pinned, 12 streamed through 4 slots (−8 layers) | 512 | 28.66 | 27.25 | 2.90 | 30.15 | -30% | 4.78 | 1.65 | 2755 | 16.37 | identical |
| partial: 16 layers pinned, 8 streamed through 4 slots (−4 layers) | 512 | 35.24 | 33.85 | 2.90 | 36.75 | -15% | 4.36 | 1.50 | 1851 | 11.00 | identical |
| partial: 18 layers pinned, 6 streamed through 3 slots (−3 layers) | 512 | 36.89 | 35.53 | 2.90 | 38.42 | -11% | 4.05 | 1.40 | 1472 | 8.75 | identical |
| partial: 20 layers pinned, 4 streamed through 2 slots (−2 layers) | 512 | 38.54 | 37.16 | 2.89 | 40.05 | -8% | 3.91 | 1.35 | 1229 | 7.30 | identical |
| partial: 22 layers pinned, 2 streamed through 2 slots (−0 layers) | 512 | 41.83 | 40.44 | 2.89 | 43.34 | +0% | 3.33 | 1.15 | 648 | 3.85 | identical |
| partial: 12 layers pinned, 12 streamed through 4 slots (−8 layers) | 2048 | 28.66 | 29.85 | 2.92 | 32.77 | -29% | 12.52 | 1.20 | 2758 | 15.64 | identical |
| partial: 16 layers pinned, 8 streamed through 4 slots (−4 layers) | 2048 | 35.24 | 36.50 | 2.92 | 39.42 | -15% | 11.99 | 1.15 | 1861 | 10.55 | identical |
| partial: 18 layers pinned, 6 streamed through 3 slots (−3 layers) | 2048 | 36.89 | 38.01 | 2.92 | 40.93 | -11% | 11.55 | 1.10 | 1521 | 8.62 | identical |
| partial: 20 layers pinned, 4 streamed through 2 slots (−2 layers) | 2048 | 38.54 | 39.76 | 2.92 | 42.68 | -8% | 11.64 | 1.11 | 1225 | 6.94 | identical |
| partial: 22 layers pinned, 2 streamed through 2 slots (−0 layers) | 2048 | 41.83 | 42.94 | 2.95 | 45.89 | -1% | 10.91 | 1.04 | 657 | 3.72 | identical |
| mmap + cgroup limit 16Gi | 128 | mapped; cgroup peak 17.18 | 16.76 | 2.89 | 19.65 | -54% | 12.23 | 10.69 | 1122 | 7.04 | identical |
| mmap + cgroup limit 24Gi | 128 | mapped; cgroup peak 25.77 | 25.02 | 2.89 | 27.91 | -35% | 9.33 | 8.15 | 562 | 3.53 | identical |
| mmap + cgroup limit 32Gi | 128 | mapped; cgroup peak 34.33 | 32.28 | 2.89 | 35.17 | -18% | 1.14 | 1.00 | 160 | 1.00 | identical |
| mmap + cgroup limit 48Gi | 128 | mapped; cgroup peak 34.34 | 32.28 | 2.89 | 35.17 | -18% | 1.16 | 1.01 | 169 | 1.06 | identical |
| mmap + cgroup limit 8Gi | 128 | mapped; cgroup peak 8.59 | 8.43 | 2.89 | 11.32 | -74% | 14.09 | 12.30 | 2457 | 15.41 | identical |
| mmap + cgroup limit 16Gi | 512 | mapped; cgroup peak 17.18 | 16.90 | 2.89 | 19.79 | -54% | 16.63 | 5.74 | 314 | 1.87 | identical |
| mmap + cgroup limit 24Gi | 512 | mapped; cgroup peak 25.77 | 25.31 | 2.90 | 28.20 | -35% | 15.93 | 5.50 | 266 | 1.58 | identical |
| mmap + cgroup limit 32Gi | 512 | mapped; cgroup peak 34.36 | 33.81 | 2.90 | 36.71 | -15% | 11.46 | 3.96 | 206 | 1.22 | identical |
| mmap + cgroup limit 48Gi | 512 | mapped; cgroup peak 37.47 | 36.19 | 2.90 | 39.08 | -10% | 2.76 | 0.95 | 167 | 0.99 | identical |
| mmap + cgroup limit 8Gi | 512 | mapped; cgroup peak 8.59 | 8.48 | 2.90 | 11.37 | -74% | 17.63 | 6.09 | 786 | 4.67 | identical |
| mmap + cgroup limit 16Gi | 2048 | mapped; cgroup peak 17.18 | 17.00 | 2.92 | 19.93 | -57% | 26.22 | 2.51 | 413 | 2.34 | identical |
| mmap + cgroup limit 24Gi | 2048 | mapped; cgroup peak 25.77 | 25.53 | 2.92 | 28.45 | -38% | 24.87 | 2.38 | 252 | 1.43 | identical |
| mmap + cgroup limit 32Gi | 2048 | mapped; cgroup peak 34.36 | 33.86 | 2.95 | 36.80 | -20% | 20.46 | 1.96 | 193 | 1.09 | identical |
| mmap + cgroup limit 48Gi | 2048 | mapped; cgroup peak 40.35 | 39.12 | 2.95 | 42.07 | -9% | 9.66 | 0.92 | 173 | 0.98 | identical |
| mmap + cgroup limit 8Gi | 2048 | mapped; cgroup peak 8.59 | 8.50 | 2.92 | 11.42 | -75% | 27.04 | 2.59 | 1250 | 7.08 | identical |

## Speculative decoding details (medians per cycle)

| config | prompt | accepted drafts / cycle | tokens / cycle | draft (ms) | verify (ms) | cycle (ms) | per token (ms) |
|---|---|---|---|---|---|---|---|

## Batching (128-token prompt)

| config | batch | total RAM (GB) | TTFT (s) | per-user ms/token | vs baseline at same batch | tokens/s | output (row 0) |
|---|---|---|---|---|---|---|---|
| baseline: all 36 layers in one pod | 1 | 42.97 | 1.14 | 159 | 1.00× | 6 | identical |
| cyclic: 2 pods, 1 layer each | 1 | 7.06 | 7.78 | 7075 | 44.38× | 0 | identical |
| cyclic: 4 pods, 1 layer each | 1 | 10.96 | 5.66 | 5502 | 34.52× | 0 | identical |
| batch 16: baseline | 16 | 43.66 | 7.27 | 568 | 1.00× | 28 | identical |
| batch 64: baseline | 64 | 46.09 | 22.86 | 976 | 1.00× | 66 | identical |
| batch 256: baseline | 256 | 51.72 | 81.52 | 2368 | 1.00× | 108 | identical |
| batch 512: baseline | 512 | 59.74 | 155.15 | 3780 | 1.00× | 135 | identical |
| batch 64: cyclic 2 pods | 64 | 11.88 | 26.00 | 7733 | 7.93× | 8 | identical |
| batch 256: cyclic 2 pods | 256 | 21.26 | 85.07 | 8767 | 3.70× | 29 | identical |
| batch 16: cyclic 4 pods | 16 | 13.43 | 10.89 | 5523 | 9.72× | 3 | identical |
| batch 64: cyclic 4 pods | 64 | 17.98 | 26.13 | 5581 | 5.72× | 11 | identical |
| batch 256: cyclic 4 pods | 256 | 35.14 | 84.42 | 5995 | 2.53× | 43 | identical |
| batch 512: cyclic 4 pods | 512 | 56.33 | 159.61 | 6744 | 1.78× | 76 | identical |
