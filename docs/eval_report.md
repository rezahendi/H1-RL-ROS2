# Evaluation report

policy `policies/h1_walk.npz` · iteration 9800 · full mode · 2026-09-24 15:38:23

## Command tracking

| command vx, vy, yaw | achieved | abs error |
|---|---|---|
| +0.00, +0.00, +0.00 | +0.00, -0.00, +0.02 | 0.001, 0.004, 0.018 |
| +0.50, +0.00, +0.00 | +0.47, -0.03, +0.00 | 0.031, 0.030, 0.003 |
| +1.00, +0.00, +0.00 | +0.89, +0.01, +0.06 | 0.107, 0.006, 0.061 |
| +0.00, +0.00, +0.00 | +0.01, +0.00, +0.04 | 0.012, 0.001, 0.036 |
| +0.00, +0.00, +0.80 | +0.02, +0.02, +0.70 | 0.017, 0.017, 0.095 |
| +0.60, +0.00, -0.60 | +0.53, -0.04, -0.53 | 0.067, 0.038, 0.067 |
| +0.00, +0.00, +0.00 | -0.03, -0.02, -0.03 | 0.026, 0.020, 0.030 |
| +0.00, +0.40, +0.00 | +0.03, +0.29, +0.03 | 0.034, 0.111, 0.029 |
| +0.00, -0.40, +0.00 | +0.02, -0.29, -0.05 | 0.016, 0.109, 0.045 |
| -0.50, +0.00, +0.00 | -0.40, -0.05, +0.04 | 0.100, 0.048, 0.042 |
| +0.00, +0.00, +0.00 | +0.01, +0.01, +0.04 | 0.007, 0.005, 0.037 |

falls: 0 · mean abs error vx 0.038 m/s, vy 0.035 m/s, yaw 0.042 rad/s

## Stopping

| from speed | failures |
|---|---|
| 0.3 m/s | 0 / 12 |
| 0.5 m/s | 0 / 12 |
| 0.7 m/s | 0 / 12 |
| 1.0 m/s | 0 / 12 |

total 0 / 48 (0.0%), of which 0 falls

## Standing still

| condition | falls |
|---|---|
| sensor noise, 12 s | 4 / 192 robots |
| randomized dynamics, 12 s | 0 / 192 robots |

## Push recovery

| case | survived |
|---|---|
| 0.4 m/s shove while standing | 7 / 12 |
| 0.8 m/s shove while standing | 3 / 12 |
| 0.4 m/s shove while walking | 12 / 12 |
| 0.8 m/s shove while walking | 12 / 12 |

## Randomized episodes

64 robots x 20 s with randomized dynamics, sensor noise and pushes: 8 falls (0.38 per robot-minute), mean |v_xy error| 0.197 m/s

## Gates

| metric | value | limit | |
|---|---|---|---|
| track/falls | 0.000 | <= 0.0 | pass |
| track/err_vx | 0.038 | <= 0.08 | pass |
| track/err_vy | 0.035 | <= 0.08 | pass |
| track/err_yaw | 0.042 | <= 0.08 | pass |
| stop/failure_rate | 0.000 | <= 0.05 | pass |
| stand/fall_rate_noise | 0.021 | <= 0.03 | pass |
| stand/fall_rate_random | 0.000 | <= 0.03 | pass |
| push/walking_survival_rate | 1.000 | >= 0.75 | pass |
