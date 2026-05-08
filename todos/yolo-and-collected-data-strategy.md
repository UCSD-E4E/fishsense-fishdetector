# YOLO development + collected unlabeled data — strategy snapshot

Created 2026-05-06 after running the four-way DeepFish evaluation
([deepfish_yolo_segmentation_eval.ipynb](../deepfish_yolo_segmentation_eval.ipynb)).
Captures the current "should we keep developing our YOLO and how do we
fold in unlabeled collected data" recommendation before pivoting to a
new request.

## Where the YOLO model stands (DeepFish test, 96 valid + 90 empty)

Our YOLO (DeepFish-trained):
- AP@0.5 = **0.92** (best of any tested model, in-domain advantage)
- mAP@[.5:.95] = 0.65 (tied with FSC, slightly behind FPN-pipeline)
- Empty-frame FPR = **1%** (best of any tested model)
- AP@0.95 = **0.002** (architectural ceiling — YOLO seg's mask head is too
  low-resolution for tight contours; MaskRCNN gets 0.16, FPN gets 0.001)

In short: best detector + best empty-frame discipline + worst high-IoU
mask precision. The mask precision ceiling is architectural, not a
training-data problem — more data won't lift AP@0.95. A larger seg
variant (yolo26s-seg, imgsz=1024) would.

## Recommendation: keep developing, pivot strategy around the unlabeled data

The AP@0.5 = 0.92 number is real in-domain advantage that compounds with
more data. Don't switch to zero-shot upstream Fishial just because their
pipeline tracks ours on aggregate mAP — they hit those numbers zero-shot,
ours hits them in-domain, and the gap grows with every batch of new data
we incorporate.

## How to fold in collected unlabeled data — three options

### Option 1: Pseudo-label with YOLO26→FPN, train on combined
- Run YOLO26→FPN on collected unlabeled data
- Filter: YOLO26 conf > 0.5 AND FPN mask covers ≥40% of YOLO26 box
- Convert masks to YOLO-seg label format, append to training set
- Retrain
- **Cost**: cheap. **Risk**: bakes Fishial's failure modes into our model.

### Option 2: Pseudo-labels as a labeling-tool starting point
- Same auto-labeling, but route through a human review tool
- 5–10× faster than from-scratch labeling
- Highest quality, moderate cost

### Option 3 (RECOMMENDED): Active learning on the disagreement set
- Run BOTH our YOLO and YOLO26→FPN on unlabeled data
- Auto-accept where they agree (IoU > 0.7, same fish count) — ~70% typically
- Send the 30% disagreement set through human review with pseudo-label as starting point
- Retrain on (DeepFish + auto-accepted + human-verified)
- Iterate; each round your model agrees with the auto-labeler more, dropping the human-labeling load

This concentrates human effort on the hardest cases, where labeling
teaches the model the most.

## Architecture decision: don't switch to YOLO26→FPN pipeline

Tempting because it tracks upstream Fishial. Don't, because:
- Cropping with YOLO26 hurt every full-frame segmenter except FPN itself
  (-0.27 to -0.69 IoU delta for YOLO ours / FSC / MaskRCNN)
- Empty-frame FPR is 12% for YOLO26→FPN vs 1% for our YOLO. Big deal for
  field footage where most frames are empty.
- Your in-domain AP@0.5 advantage gets thrown away.

The case for switching opens up only if downstream length estimation
ends up bottlenecked by AP@0.95-style mask boundary precision. Worth
quantifying that downstream impact before any architecture change.

## Do this BEFORE pseudo-labeling

**Sanity-check the FPN preprocessing.** Standalone FPN got 0.155 mean IoU
and 100% empty-frame FPR — too bad to be solely "wrong use case". There's
a chance our `ff_predict_mask` preprocessing (mean/std, RGB ordering,
resize policy) is slightly off vs upstream's reference.

Concrete check:
1. Pick 5 DeepFish valid-split images
2. Run upstream's reference Colab notebook on them
3. Run our `ff_predict_mask` on the same images
4. Diff the binary masks pixel-wise

If outputs match: FPN really is unusable on full underwater frames.
Pseudo-labels from YOLO26→FPN are still viable (the cropping fixes it).
If outputs differ: fix our preprocessing first, then pseudo-label, then
re-evaluate FPN standalone.

This is load-bearing for whether the pseudo-labeling pipeline produces
high-quality labels.

## Open questions for future-me

- **Mask precision ceiling**: is AP@0.95 = 0.002 actually a problem
  downstream? Run length-estimation on the existing YOLO masks vs ground
  truth and quantify the error. If error is dominated by length-prediction
  algorithm noise rather than mask boundary quality, fine — keep YOLO.
- **Larger YOLO seg variant**: how much does going to yolo26s-seg or
  imgsz=1024 close the AP@0.95 gap? One small experiment.
- **Domain shift on collected data**: DeepFish is a specific underwater
  surveillance distribution. Field-collected data may look very different
  (different lighting, different species, different camera). Worth
  sanity-checking by running our DeepFish-trained YOLO on a sample of
  collected data and eyeballing whether the predictions look reasonable
  before pseudo-labeling at scale.

## Pointers

- Notebook with the full eval: [../deepfish_yolo_segmentation_eval.ipynb](../deepfish_yolo_segmentation_eval.ipynb)
- Current trained YOLO weights: [../runs/segment/train/weights/best.pt](../runs/segment/train/weights/best.pt)
- DeepFish data: [../data/DeepFish/](../data/) (gitignored)
- YOLO-format labels (external): `../../datasets/DeepFish/Segmentation/`
