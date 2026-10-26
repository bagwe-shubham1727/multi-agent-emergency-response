# Evaluation summary

Cells are mean ± std (ddof=1) across seeds and repeats; n = number of runs.
Paired rows compare conditions on identical worlds (same scenario, seed, repeat).
W/T/L counts runs where the first condition is better / equal / worse; higher is
better only for red_in_window, red_in_window_rate, served_rate, survival_weighted_score, victims_served; lower is better otherwise.
All metrics, including those that get worse, are in summary.csv and paired.csv.

## normal (offline)

| Condition | n | Served rate | RED in window | Survival score | Unserved RED | Response (min) | Door-to-doctor (min) | Conflicts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| b0 | 21 | 0.894 ± 0.125 | 0.471 ± 0.203 | 0.685 ± 0.130 | 1.33 ± 1.62 | 17.0 ± 7.0 | 0.5 ± 1.3 | 0.0 ± 0.0 |
| b0-cues | 21 | 0.865 ± 0.157 | 0.486 ± 0.206 | 0.671 ± 0.158 | 1.43 ± 1.75 | 16.8 ± 5.6 | 0.5 ± 1.3 | 0.0 ± 0.0 |
| b1 | 21 | 0.894 ± 0.128 | 0.505 ± 0.215 | 0.694 ± 0.158 | 1.24 ± 1.67 | 16.2 ± 4.6 | 0.4 ± 1.1 | 0.3 ± 0.6 |
| full | 21 | 0.905 ± 0.118 | 0.565 ± 0.185 | 0.727 ± 0.148 | 1.00 ± 1.45 | 15.1 ± 3.8 | 0.4 ± 1.1 | 2.1 ± 2.0 |
| full-no-float | 21 | 0.905 ± 0.118 | 0.565 ± 0.185 | 0.727 ± 0.148 | 1.00 ± 1.45 | 15.1 ± 3.8 | 0.4 ± 1.1 | 2.0 ± 2.0 |
| full-no-diversion | 21 | 0.877 ± 0.148 | 0.518 ± 0.215 | 0.692 ± 0.171 | 1.43 ± 2.11 | 16.0 ± 4.6 | 0.4 ± 1.1 | 1.4 ± 1.5 |
| full-no-escalation | 21 | 0.893 ± 0.129 | 0.505 ± 0.215 | 0.693 ± 0.159 | 1.24 ± 1.67 | 16.2 ± 4.6 | 0.4 ± 1.1 | 0.4 ± 0.7 |

| Comparison | Metric | n | Mean diff | W/T/L | Sign-test p |
|---|---|---:|---:|---:|---:|
| full vs b0 | served_rate | 21 | +0.011 | 7/11/3 | 0.3438 |
| full vs b0 | red_in_window_rate | 21 | +0.095 | 11/7/3 | 0.0574 |
| full vs b0 | survival_weighted_score | 21 | +0.042 | 11/5/5 | 0.2101 |
| full vs b0 | unserved_critical | 21 | -0.333 | 6/13/2 | 0.2891 |
| full vs b0 | response_time_mean_min | 21 | -1.841 | 12/0/9 | 0.6636 |
| full vs b0 | door_to_doctor_mean_min | 21 | -0.073 | 4/14/3 | 1.0000 |
| full vs b0 | conflicts | 21 | +2.095 | 0/5/16 | 0.0000 |
| full vs b0-cues | served_rate | 21 | +0.040 | 7/13/1 | 0.0703 |
| full vs b0-cues | red_in_window_rate | 21 | +0.079 | 8/11/2 | 0.1094 |
| full vs b0-cues | survival_weighted_score | 21 | +0.056 | 9/9/3 | 0.1460 |
| full vs b0-cues | unserved_critical | 21 | -0.429 | 5/16/0 | 0.0625 |
| full vs b0-cues | response_time_mean_min | 21 | -1.660 | 12/2/7 | 0.3593 |
| full vs b0-cues | door_to_doctor_mean_min | 21 | -0.074 | 4/14/3 | 1.0000 |
| full vs b0-cues | conflicts | 21 | +2.095 | 0/5/16 | 0.0000 |
| full vs b1 | served_rate | 21 | +0.011 | 3/16/2 | 1.0000 |
| full vs b1 | red_in_window_rate | 21 | +0.060 | 5/16/0 | 0.0625 |
| full vs b1 | survival_weighted_score | 21 | +0.033 | 6/13/2 | 0.2891 |
| full vs b1 | unserved_critical | 21 | -0.238 | 2/18/1 | 1.0000 |
| full vs b1 | response_time_mean_min | 21 | -1.086 | 7/11/3 | 0.3438 |
| full vs b1 | door_to_doctor_mean_min | 21 | +0.045 | 2/15/4 | 0.6875 |
| full vs b1 | conflicts | 21 | +1.762 | 0/6/15 | 0.0001 |
| b1 vs b0-cues | served_rate | 21 | +0.029 | 6/14/1 | 0.1250 |
| b1 vs b0-cues | red_in_window_rate | 21 | +0.019 | 5/12/4 | 1.0000 |
| b1 vs b0-cues | survival_weighted_score | 21 | +0.023 | 6/11/4 | 0.7539 |
| b1 vs b0-cues | unserved_critical | 21 | -0.190 | 5/15/1 | 0.2188 |
| b1 vs b0-cues | response_time_mean_min | 21 | -0.574 | 10/3/8 | 0.8145 |
| b1 vs b0-cues | door_to_doctor_mean_min | 21 | -0.119 | 4/15/2 | 0.6875 |
| b1 vs b0-cues | conflicts | 21 | +0.333 | 0/15/6 | 0.0312 |
| b0-cues vs b0 | served_rate | 21 | -0.030 | 1/15/5 | 0.2188 |
| b0-cues vs b0 | red_in_window_rate | 21 | +0.015 | 4/12/5 | 1.0000 |
| b0-cues vs b0 | survival_weighted_score | 21 | -0.015 | 4/11/6 | 0.7539 |
| b0-cues vs b0 | unserved_critical | 21 | +0.095 | 1/17/3 | 0.6250 |
| b0-cues vs b0 | response_time_mean_min | 21 | -0.181 | 7/4/10 | 0.6291 |
| b0-cues vs b0 | door_to_doctor_mean_min | 21 | +0.001 | 1/19/1 | 1.0000 |
| b0-cues vs b0 | conflicts | 21 | +0.000 | 0/21/0 | 1.0000 |
| b1 vs b0 | served_rate | 21 | -0.000 | 7/11/3 | 0.3438 |
| b1 vs b0 | red_in_window_rate | 21 | +0.034 | 8/8/5 | 0.5811 |
| b1 vs b0 | survival_weighted_score | 21 | +0.008 | 9/7/5 | 0.4240 |
| b1 vs b0 | unserved_critical | 21 | -0.095 | 5/14/2 | 0.4531 |
| b1 vs b0 | response_time_mean_min | 21 | -0.755 | 10/1/10 | 1.0000 |
| b1 vs b0 | door_to_doctor_mean_min | 21 | -0.118 | 4/14/3 | 1.0000 |
| b1 vs b0 | conflicts | 21 | +0.333 | 0/15/6 | 0.0312 |

## surge (offline)

| Condition | n | Served rate | RED in window | Survival score | Unserved RED | Response (min) | Door-to-doctor (min) | Conflicts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| b0 | 21 | 0.852 ± 0.124 | 0.319 ± 0.161 | 0.585 ± 0.143 | 2.76 ± 2.57 | 32.3 ± 12.5 | 0.9 ± 1.3 | 0.0 ± 0.0 |
| b0-cues | 21 | 0.783 ± 0.171 | 0.312 ± 0.157 | 0.535 ± 0.175 | 4.00 ± 4.18 | 30.5 ± 11.9 | 1.2 ± 1.6 | 0.0 ± 0.0 |
| b1 | 21 | 0.807 ± 0.151 | 0.350 ± 0.205 | 0.571 ± 0.179 | 3.52 ± 3.54 | 27.8 ± 9.1 | 0.4 ± 0.6 | 3.3 ± 2.4 |
| full | 21 | 0.796 ± 0.168 | 0.357 ± 0.189 | 0.571 ± 0.177 | 3.81 ± 4.12 | 25.6 ± 7.4 | 0.4 ± 0.5 | 9.6 ± 3.1 |
| full-no-float | 21 | 0.786 ± 0.165 | 0.357 ± 0.185 | 0.563 ± 0.177 | 4.05 ± 4.13 | 25.2 ± 7.4 | 0.5 ± 0.7 | 9.1 ± 3.0 |
| full-no-diversion | 21 | 0.793 ± 0.165 | 0.365 ± 0.190 | 0.573 ± 0.175 | 3.86 ± 4.11 | 25.9 ± 7.4 | 0.4 ± 0.5 | 8.1 ± 3.1 |
| full-no-escalation | 21 | 0.809 ± 0.145 | 0.346 ± 0.214 | 0.573 ± 0.179 | 3.38 ± 3.43 | 27.6 ± 9.2 | 0.3 ± 0.5 | 3.0 ± 2.2 |

| Comparison | Metric | n | Mean diff | W/T/L | Sign-test p |
|---|---|---:|---:|---:|---:|
| full vs b0 | served_rate | 21 | -0.056 | 5/6/10 | 0.3018 |
| full vs b0 | red_in_window_rate | 21 | +0.039 | 10/6/5 | 0.3018 |
| full vs b0 | survival_weighted_score | 21 | -0.013 | 9/2/10 | 1.0000 |
| full vs b0 | unserved_critical | 21 | +1.048 | 4/9/8 | 0.3877 |
| full vs b0 | response_time_mean_min | 21 | -6.727 | 15/0/6 | 0.0784 |
| full vs b0 | door_to_doctor_mean_min | 21 | -0.548 | 12/4/5 | 0.1435 |
| full vs b0 | conflicts | 21 | +9.619 | 0/0/21 | 0.0000 |
| full vs b0-cues | served_rate | 21 | +0.012 | 10/5/6 | 0.4545 |
| full vs b0-cues | red_in_window_rate | 21 | +0.045 | 13/3/5 | 0.0963 |
| full vs b0-cues | survival_weighted_score | 21 | +0.036 | 15/2/4 | 0.0192 |
| full vs b0-cues | unserved_critical | 21 | -0.190 | 9/8/4 | 0.2668 |
| full vs b0-cues | response_time_mean_min | 21 | -4.903 | 12/0/9 | 0.6636 |
| full vs b0-cues | door_to_doctor_mean_min | 21 | -0.867 | 15/4/2 | 0.0023 |
| full vs b0-cues | conflicts | 21 | +9.619 | 0/0/21 | 0.0000 |
| full vs b1 | served_rate | 21 | -0.011 | 5/8/8 | 0.5811 |
| full vs b1 | red_in_window_rate | 21 | +0.008 | 9/7/5 | 0.4240 |
| full vs b1 | survival_weighted_score | 21 | +0.000 | 10/4/7 | 0.6291 |
| full vs b1 | unserved_critical | 21 | +0.286 | 3/12/6 | 0.5078 |
| full vs b1 | response_time_mean_min | 21 | -2.164 | 13/0/8 | 0.3833 |
| full vs b1 | door_to_doctor_mean_min | 21 | -0.086 | 6/10/5 | 1.0000 |
| full vs b1 | conflicts | 21 | +6.333 | 0/0/21 | 0.0000 |
| b1 vs b0-cues | served_rate | 21 | +0.024 | 11/5/5 | 0.2101 |
| b1 vs b0-cues | red_in_window_rate | 21 | +0.038 | 10/5/6 | 0.4545 |
| b1 vs b0-cues | survival_weighted_score | 21 | +0.035 | 11/4/6 | 0.3323 |
| b1 vs b0-cues | unserved_critical | 21 | -0.476 | 7/9/5 | 0.7744 |
| b1 vs b0-cues | response_time_mean_min | 21 | -2.740 | 15/0/6 | 0.0784 |
| b1 vs b0-cues | door_to_doctor_mean_min | 21 | -0.781 | 15/5/1 | 0.0005 |
| b1 vs b0-cues | conflicts | 21 | +3.286 | 0/3/18 | 0.0000 |
| b0-cues vs b0 | served_rate | 21 | -0.069 | 1/9/11 | 0.0063 |
| b0-cues vs b0 | red_in_window_rate | 21 | -0.007 | 7/6/8 | 1.0000 |
| b0-cues vs b0 | survival_weighted_score | 21 | -0.049 | 3/3/15 | 0.0075 |
| b0-cues vs b0 | unserved_critical | 21 | +1.238 | 3/11/7 | 0.3438 |
| b0-cues vs b0 | response_time_mean_min | 21 | -1.823 | 14/1/6 | 0.1153 |
| b0-cues vs b0 | door_to_doctor_mean_min | 21 | +0.320 | 5/5/11 | 0.2101 |
| b0-cues vs b0 | conflicts | 21 | +0.000 | 0/21/0 | 1.0000 |
| b1 vs b0 | served_rate | 21 | -0.045 | 5/6/10 | 0.3018 |
| b1 vs b0 | red_in_window_rate | 21 | +0.031 | 8/5/8 | 1.0000 |
| b1 vs b0 | survival_weighted_score | 21 | -0.014 | 6/4/11 | 0.3323 |
| b1 vs b0 | unserved_critical | 21 | +0.762 | 4/9/8 | 0.3877 |
| b1 vs b0 | response_time_mean_min | 21 | -4.563 | 14/0/7 | 0.1892 |
| b1 vs b0 | door_to_doctor_mean_min | 21 | -0.461 | 13/5/3 | 0.0213 |
| b1 vs b0 | conflicts | 21 | +3.286 | 0/3/18 | 0.0000 |

## constrained (offline)

| Condition | n | Served rate | RED in window | Survival score | Unserved RED | Response (min) | Door-to-doctor (min) | Conflicts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| b0 | 21 | 0.772 ± 0.132 | 0.271 ± 0.114 | 0.506 ± 0.134 | 4.19 ± 3.03 | 38.9 ± 11.6 | 2.7 ± 2.5 | 0.0 ± 0.0 |
| b0-cues | 21 | 0.714 ± 0.155 | 0.260 ± 0.141 | 0.459 ± 0.155 | 5.14 ± 3.75 | 34.0 ± 10.0 | 3.2 ± 2.6 | 0.0 ± 0.0 |
| b1 | 21 | 0.734 ± 0.160 | 0.306 ± 0.166 | 0.502 ± 0.164 | 4.90 ± 3.83 | 33.5 ± 8.7 | 1.4 ± 1.3 | 5.1 ± 2.4 |
| full | 21 | 0.739 ± 0.175 | 0.344 ± 0.170 | 0.524 ± 0.173 | 5.00 ± 3.97 | 30.2 ± 7.6 | 1.0 ± 1.1 | 11.9 ± 2.4 |
| full-no-float | 21 | 0.719 ± 0.168 | 0.313 ± 0.163 | 0.498 ± 0.168 | 5.24 ± 3.83 | 30.5 ± 7.8 | 1.6 ± 1.3 | 12.3 ± 2.6 |
| full-no-diversion | 21 | 0.754 ± 0.179 | 0.347 ± 0.171 | 0.533 ± 0.178 | 4.71 ± 4.05 | 30.5 ± 7.3 | 1.1 ± 1.0 | 10.6 ± 2.6 |
| full-no-escalation | 21 | 0.744 ± 0.166 | 0.315 ± 0.149 | 0.514 ± 0.163 | 4.81 ± 3.67 | 32.6 ± 9.6 | 0.9 ± 0.9 | 5.0 ± 2.3 |

| Comparison | Metric | n | Mean diff | W/T/L | Sign-test p |
|---|---|---:|---:|---:|---:|
| full vs b0 | served_rate | 21 | -0.034 | 7/5/9 | 0.8036 |
| full vs b0 | red_in_window_rate | 21 | +0.073 | 13/1/7 | 0.2632 |
| full vs b0 | survival_weighted_score | 21 | +0.018 | 14/0/7 | 0.1892 |
| full vs b0 | unserved_critical | 21 | +0.810 | 4/8/9 | 0.2668 |
| full vs b0 | response_time_mean_min | 21 | -8.703 | 18/0/3 | 0.0015 |
| full vs b0 | door_to_doctor_mean_min | 21 | -1.658 | 17/0/4 | 0.0072 |
| full vs b0 | conflicts | 21 | +11.857 | 0/0/21 | 0.0000 |
| full vs b0-cues | served_rate | 21 | +0.025 | 12/5/4 | 0.0768 |
| full vs b0-cues | red_in_window_rate | 21 | +0.084 | 15/3/3 | 0.0075 |
| full vs b0-cues | survival_weighted_score | 21 | +0.065 | 17/1/3 | 0.0026 |
| full vs b0-cues | unserved_critical | 21 | -0.143 | 11/4/6 | 0.3323 |
| full vs b0-cues | response_time_mean_min | 21 | -3.854 | 14/0/7 | 0.1892 |
| full vs b0-cues | door_to_doctor_mean_min | 21 | -2.228 | 19/1/1 | 0.0000 |
| full vs b0-cues | conflicts | 21 | +11.857 | 0/0/21 | 0.0000 |
| full vs b1 | served_rate | 21 | +0.005 | 9/5/7 | 0.8036 |
| full vs b1 | red_in_window_rate | 21 | +0.038 | 11/6/4 | 0.1185 |
| full vs b1 | survival_weighted_score | 21 | +0.022 | 14/2/5 | 0.0636 |
| full vs b1 | unserved_critical | 21 | +0.095 | 6/7/8 | 0.7905 |
| full vs b1 | response_time_mean_min | 21 | -3.363 | 18/0/3 | 0.0015 |
| full vs b1 | door_to_doctor_mean_min | 21 | -0.382 | 11/1/9 | 0.8238 |
| full vs b1 | conflicts | 21 | +6.714 | 0/0/21 | 0.0000 |
| b1 vs b0-cues | served_rate | 21 | +0.020 | 11/3/7 | 0.4807 |
| b1 vs b0-cues | red_in_window_rate | 21 | +0.046 | 13/2/6 | 0.1671 |
| b1 vs b0-cues | survival_weighted_score | 21 | +0.042 | 15/0/6 | 0.0784 |
| b1 vs b0-cues | unserved_critical | 21 | -0.238 | 11/3/7 | 0.4807 |
| b1 vs b0-cues | response_time_mean_min | 21 | -0.491 | 12/0/9 | 0.6636 |
| b1 vs b0-cues | door_to_doctor_mean_min | 21 | -1.846 | 16/1/4 | 0.0118 |
| b1 vs b0-cues | conflicts | 21 | +5.143 | 0/0/21 | 0.0000 |
| b0-cues vs b0 | served_rate | 21 | -0.058 | 3/5/13 | 0.0213 |
| b0-cues vs b0 | red_in_window_rate | 21 | -0.011 | 6/4/11 | 0.3323 |
| b0-cues vs b0 | survival_weighted_score | 21 | -0.046 | 5/4/12 | 0.1435 |
| b0-cues vs b0 | unserved_critical | 21 | +0.952 | 3/7/11 | 0.0574 |
| b0-cues vs b0 | response_time_mean_min | 21 | -4.848 | 15/1/5 | 0.0414 |
| b0-cues vs b0 | door_to_doctor_mean_min | 21 | +0.570 | 5/2/14 | 0.0636 |
| b0-cues vs b0 | conflicts | 21 | +0.000 | 0/21/0 | 1.0000 |
| b1 vs b0 | served_rate | 21 | -0.038 | 5/3/13 | 0.0963 |
| b1 vs b0 | red_in_window_rate | 21 | +0.035 | 10/3/8 | 0.8145 |
| b1 vs b0 | survival_weighted_score | 21 | -0.004 | 10/0/11 | 1.0000 |
| b1 vs b0 | unserved_critical | 21 | +0.714 | 7/4/10 | 0.6291 |
| b1 vs b0 | response_time_mean_min | 21 | -5.339 | 19/0/2 | 0.0002 |
| b1 vs b0 | door_to_doctor_mean_min | 21 | -1.276 | 15/0/6 | 0.0784 |
| b1 vs b0 | conflicts | 21 | +5.143 | 0/0/21 | 0.0000 |
