# Zero-shot presence maps (train, in-sample diagnostics)

## pixelseal (170 images, 101.6 s)

- all clean mIoU 0.253; fill with true polarity 0.452
- whole-image bit accuracy: 0.983 (marked background) vs 0.545 (clean background)
- polarity from bit_acc: AUC 0.998, best accuracy 0.976
- polarity from agreement: AUC 0.996, best accuracy 0.971

| map | AUC | AUC clean bg | AUC marked bg | mIoU best z-thr | mIoU oracle fraction |
|---|---:|---:|---:|---:|---:|
| native_work | 0.629 | 0.613 | 0.645 | 0.343 | 0.513 |
| occlusion | 0.627 | 0.571 | 0.683 | 0.308 | 0.504 |
| native_512 | 0.623 | 0.570 | 0.675 | 0.319 | 0.503 |
| hp_energy | 0.548 | 0.559 | 0.537 | 0.292 | 0.466 |
| gradhp_s1 | 0.499 | 0.502 | 0.497 | 0.281 | 0.455 |
| gradhp_s8 | 0.498 | 0.516 | 0.479 | 0.286 | 0.457 |
| gradhp_s2 | 0.497 | 0.505 | 0.489 | 0.281 | 0.455 |
| gradhp_s4 | 0.492 | 0.505 | 0.478 | 0.283 | 0.455 |

## trustmark (170 images, 50.8 s)

- all clean mIoU 0.248; fill with true polarity 0.452
- whole-image bit accuracy: 0.978 (marked background) vs 0.530 (clean background)
- polarity from bit_acc: AUC 0.999, best accuracy 0.982
- polarity from agreement: AUC 0.999, best accuracy 0.982

| map | AUC | AUC clean bg | AUC marked bg | mIoU best z-thr | mIoU oracle fraction |
|---|---:|---:|---:|---:|---:|
| native_work | 0.594 | 0.558 | 0.630 | 0.325 | 0.482 |
| hp_energy | 0.565 | 0.587 | 0.543 | 0.286 | 0.462 |
| occlusion | 0.538 | 0.506 | 0.569 | 0.278 | 0.468 |
| native_512 | 0.513 | 0.500 | 0.527 | 0.280 | 0.455 |
| gradhp_s1 | 0.498 | 0.501 | 0.496 | 0.282 | 0.454 |
| gradhp_s2 | 0.494 | 0.498 | 0.490 | 0.279 | 0.454 |
| gradhp_s8 | 0.488 | 0.490 | 0.486 | 0.279 | 0.456 |
| gradhp_s4 | 0.485 | 0.488 | 0.482 | 0.277 | 0.454 |

## stegastamp (170 images, 62.6 s)

- all clean mIoU 0.254; fill with true polarity 0.460
- whole-image bit accuracy: 0.970 (marked background) vs 0.523 (clean background)
- polarity from bit_acc: AUC 0.999, best accuracy 0.988
- polarity from agreement: AUC 0.999, best accuracy 0.988

| map | AUC | AUC clean bg | AUC marked bg | mIoU best z-thr | mIoU oracle fraction |
|---|---:|---:|---:|---:|---:|
| hp_energy | 0.607 | 0.668 | 0.546 | 0.300 | 0.487 |
| occlusion | 0.533 | 0.535 | 0.531 | 0.278 | 0.463 |
| gradhp_s1 | 0.497 | 0.504 | 0.491 | 0.279 | 0.458 |
| gradhp_s2 | 0.496 | 0.507 | 0.484 | 0.275 | 0.457 |
| gradhp_s4 | 0.492 | 0.508 | 0.476 | 0.271 | 0.456 |
| gradhp_s8 | 0.479 | 0.494 | 0.464 | 0.269 | 0.454 |
