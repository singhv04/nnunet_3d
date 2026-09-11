# metrics 
```

| Class                     | Dice   | IoU    | TP     | FP     | FN      | n_ref  | n_pred |
|---------------------------|--------|--------|--------|--------|---------|--------|--------|
| Epidural (EDH)            | 0.1760 | 0.1362 | 522.5  | 38.8   | 1489.4  | 2011.9 | 561.4  |
| Intraparenchymal (IPH)    | 0.6137 | 0.5149 | 6598.5 | 658.8  | 1864.7  | 8463.1 | 7257.3 |
| Intraventricular (IVH)    | 0.4649 | 0.3657 | 1138.9 | 176.1  | 699.1   | 1838.0 | 1315.1 |
| Subarachnoid (SAH)        | 0.2188 | 0.1451 | 1943.2 | 672.7  | 3198.4  | 5141.6 | 2615.9 |
| Subdural (SDH)            | 0.1913 | 0.1387 | 1776.3 | 724.7  | 3699.5  | 5475.8 | 2501.0 |


| Metric                 | Value     |
|------------------------|-----------|
| Foreground Mean Dice   | **0.3330**|
| Foreground Mean IoU    | 0.2601    |
| Total TP               | 2395.9    |
| Total FP               | 454.2     |
| Total FN               | 2190.2    |


- Best performance is observed for **Intraparenchymal Hemorrhage (IPH)**, likely due to larger lesion volume and higher prevalence.
- **Epidural, Subarachnoid, and Subdural hemorrhages** show lower Dice scores, consistent with their thin, irregular morphology and class imbalance.
- Metrics are reported **per-class Dice and IoU**, following standard practice in BHSD-related literature.
- Foreground mean Dice is computed across all hemorrhage classes excluding background.


| Hemorrhage Type          | nnUNet3D (BHSD Paper) | Our nnU-Net v2 | Difference |
|--------------------------|----------------------|---------------|------------|
| Epidural (EDH)           | 4.81                 | **17.60**     | ↑ +12.79   |
| Intraparenchymal (IPH)   | 54.12                | **61.37**     | ↑ +7.25    |
| Intraventricular (IVH)   | **51.48**            | 46.49         | ↓ −4.99    |
| Subarachnoid (SAH)       | 21.57                | **21.88**     | ≈ +0.31    |
| Subdural (SDH)           | 15.23                | **19.13**     | ↑ +3.90    |
| **Mean (Foreground)**    | 29.44                | **33.30**     | ↑ +3.86    |


- Observation:
    - Overall, the proposed nnU-Net v2 model demonstrates performance in the same range as the nnUNet3D benchmark reported in the BHSD study, with several hemorrhage subtypes showing improved Dice scores.

    - Notably, the model achieves substantially higher Dice scores for Epidural (EDH) and Intraparenchymal Hemorrhage (IPH), which may be attributed to improved architectural components and optimization strategies in nnU-Net v2. Performance for Subarachnoid (SAH) and Subdural Hemorrhage (SDH) is comparable to or slightly better than the benchmark, indicating stable segmentation capability for thin and irregular hemorrhage patterns.

    - However, performance for Intraventricular Hemorrhage (IVH) is lower compared to the nnUNet3D results reported in the literature. This discrepancy may stem from differences in data splits, training protocol, or class imbalance handling, and suggests that IVH segmentation remains a challenging task requiring further investigation.

    - Importantly, the reported results are derived from a limited evaluation protocol (single-fold or partial validation), whereas the BHSD benchmark reports results obtained via full cross-validation. As such, the current comparison should be considered preliminary, and a complete multi-fold evaluation is required for a rigorous and fair comparison.

- Key Takeaways:
    - Our nnU-Net v2 results are within the performance range reported for nnUNet3D on the BHSD dataset.
    - Improved Dice scores are observed for EDH, IPH, SAH, and SDH.
    - Lower IVH performance highlights an area for targeted improvement.
    - A full multi-fold cross-validation study can be done later for a definitive benchmark comparison.

```