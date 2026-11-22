# Team Chapeau — European AI Championship, Paris 2026

Code, methods and write-ups of team **Chapeau** for the hackathon tasks. Each task lives in its own folder with a
README explaining the method, the results and how to run it.

**Slides** (6-minute presentation of the three tasks): [`slides/Team_Chapeau_EAIC_Paris_2026.pdf`](slides/Team_Chapeau_EAIC_Paris_2026.pdf)

| Folder | Task | Approach in one line |
|---|---|---|
| [`task_1/`](task_1/) | Watermark Localization | Segment the watermarked region (PixelSeal / TrustMark / StegaStamp) of edited AI-generated images with decoder signals, shape priors and metric-aware post-processing (public score 0.7444). |
| [`task_2/`](task_2/) | Model Equality Testing | Identify base model, deployment disguises and weight variants of 33 black-box LLM endpoints with likelihood statistics and paired permutation tests under local candidate models. |
| [`task_3/`](task_3/) | Machine Unlearning (ViT-B/16) | Re-initialise the top half of the network, re-distil it on synthetic in-style images that never contain the forget set, then scale the logits (public score ≈ 0.41). |

Task data, API keys, the organisers' server address and model checkpoints are not included
(except the small final submission files).
