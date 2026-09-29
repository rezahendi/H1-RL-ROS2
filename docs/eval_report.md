# Evaluation report

policy `policies/h1_walk.npz` · iteration 13500 · full mode · 2026-09-26 18:18:12

## Command tracking

| command vx, vy, yaw | achieved | abs error |
|---|---|---|
| +0.00, +0.00, +0.00 | -0.00, +0.00, -0.01 | 0.001, 0.001, 0.009 |
| +0.50, +0.00, +0.00 | +0.43, +0.00, -0.03 | 0.066, 0.003, 0.030 |
| +1.00, +0.00, +0.00 | +0.71, +0.02, -0.02 | 0.095, 0.024, 0.023 |
| +0.00, +0.00, +0.00 | -0.02, -0.01, -0.01 | 0.021, 0.006, 0.008 |
| +0.00, +0.00, +0.80 | +0.02, -0.02, +0.74 | 0.019, 0.021, 0.055 |
| +0.60, +0.00, -0.60 | +0.51, -0.01, -0.58 | 0.087, 0.008, 0.017 |
| +0.00, +0.00, +0.00 | -0.01, +0.00, -0.03 | 0.010, 0.003, 0.032 |
| +0.00, +0.40, +0.00 | -0.04, +0.26, -0.01 | 0.037, 0.137, 0.014 |
| +0.00, -0.40, +0.00 | -0.02, -0.28, -0.01 | 0.020, 0.117, 0.015 |
| -0.50, +0.00, +0.00 | -0.45, -0.05, -0.01 | 0.046, 0.050, 0.013 |
| +0.00, +0.00, +0.00 | +0.00, -0.00, -0.01 | 0.002, 0.002, 0.008 |

falls: 0 · mean abs error vx 0.037 m/s, vy 0.034 m/s, yaw 0.020 rad/s

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
| sensor noise, 12 s | 0 / 192 robots |
| randomized dynamics, 12 s | 0 / 192 robots |

stepping during quiet standing (false recovery triggers): 0.00% of the time

## Push recovery

| case | survived |
|---|---|
| 0.4 m/s shove while standing | 12 / 12 |
| 0.8 m/s shove while standing | 8 / 12 |
| 0.4 m/s shove while walking | 12 / 12 |
| 0.8 m/s shove while walking | 12 / 12 |

## Randomized episodes

64 robots x 20 s with randomized dynamics, sensor noise and pushes: 5 falls (0.23 per robot-minute), mean |v_xy error| 0.193 m/s

## Gates

| metric | value | limit | |
|---|---|---|---|
| track/falls | 0.000 | <= 0.0 | pass |
| track/err_vx | 0.037 | <= 0.08 | pass |
| track/err_vy | 0.034 | <= 0.08 | pass |
| track/err_yaw | 0.020 | <= 0.08 | pass |
| stop/failure_rate | 0.000 | <= 0.05 | pass |
| stand/fall_rate_noise | 0.000 | <= 0.03 | pass |
| stand/fall_rate_random | 0.000 | <= 0.03 | pass |
| stand/restless_fraction | 0.000 | <= 0.05 | pass |
| push/walking_survival_rate | 1.000 | >= 0.75 | pass |
