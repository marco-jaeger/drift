# The DRIFT pipeline

This walkthrough follows one dataset through the full pipeline, stage by
stage. Every stage is one command of the form

```bash
uv run src/run.py <command> dataset=<name> [key=value ...]
```

Defaults live in each stage's config dataclass; `key=value` arguments
override them, and only `dataset` is mandatory. Every stage writes the fully
resolved configuration (plus timestamp and git commit) next to its outputs,
so any result folder is self-describing.

## Pipeline overview

<img src="https://github.com/marco-jaeger/drift/releases/download/v1.0.0/method_pipeline.png" width="750" alt="Detected icebergs with bounding boxes">

The pipeline consists of eight stages, each reading the previous stage's
file and writing its own, so it can be entered, repeated, or stopped at any
point:

1. **Detection**: Detect icebergs in every frame with a fine-tuned Faster
   R-CNN, and embed them inline
2. **Appearance embeddings**: Map every detection to a DINOv2 appearance
   vector for re-identification
3. **Tracking**: Associate detections across frames into trajectories using
   appearance, motion, and size
4. **Evaluation**: Score the tracking against ground truth with the CLEAR and
   Identity metrics
5. **Outline extraction**: Segment each tracked detection into an outline
   polygon (SAM or Otsu)
6. **Georeferencing**: Project the tracks onto the sea surface in UTM and
   measure iceberg sizes
7. **Fjord circulation**: Aggregate the trajectories into a gridded velocity
   field with vector and streamline panels

Stages 1 and 2 have a training counterpart (`train-detection`,
`train-embedding`) that can be run on the annotated training dataset
(`ekas-hill-train`) if necessary; the two resulting checkpoints in `models/` are shared by
all datasets. Visualization renders annotated frames, videos, and GIFs of any stage.

## Dataset configuration

The dataset is a directory under `data/`, holding either a single sequence or
one subdirectory per sequence (e.g. one per condition or time span).
`dataset=` names it, and everything else is derived from that name.

#### Basic directory structure

```
data/
└── ekas-hill/
    ├── camera.json                 # Calibrated glimpse camera (for stage 6+7 only)
    ├── mask.jpg                    # Optional land mask (black = ignored)
    ├── clear/
    │   ├── images/                 # Time-lapse frames, e.g. 000001.jpg
    │   ├── ground_truth/
    │   │   └── gt.txt              # MOT annotations (needed for training/eval only)
    └── melange/
        └── ...
```

`camera.json` and `mask.jpg` live at the dataset root and are shared by all
of its sequences. Outputs mirror the input tree e.g. `data/ekas-hill/clear/`
produces `results/ekas-hill/clear/`, one subdirectory per stage.

Naming a dataset processes all of its sequences, appending a sequence name
processes just that one:

```bash
uv run src/run.py detect dataset=ekas-hill          # every sequence
uv run src/run.py detect dataset=ekas-hill/clear    # one sequence
```

#### Dataset format

Annotations (`gt.txt`, `det.txt`, `track.txt`) follow the [MOT Challenge](https://motchallenge.net/instructions/)
format where each line is one iceberg in one frame. Ground truth is only needed to
train the models or to run `eval`:

`<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>`

- `<frame>`: Frame index, or the image filename without its extension
- `<id>`: Iceberg ID, consistent across frames
- `<bb_left>, <bb_top>, <bb_width>, <bb_height>`: Bounding box in pixels
- `<conf>`: Confidence
- `<x>, <y>, <z>`: Unused

## 1. Detection

`detect` finds icebergs in every frame with a Faster R-CNN fine-tuned on the
training dataset (multi-scale, sliding windows, followed by
confidence/size/mask filtering and non-maximum suppression). By default it
also computes the appearance embeddings for its detections inline (stage 2).

Input: `images/*` + `models/detection.pt` ->  Output: `detections/det.txt` + `detections/embeddings.pt`.


```bash
uv run src/run.py train-detection dataset=ekas-hill-train   # for training (optional)
uv run src/run.py detect dataset=ekas-hill
```

<img src="https://github.com/marco-jaeger/drift/releases/download/v1.0.0/detections.jpg" width="1000" alt="Detected icebergs with bounding boxes">

*Detected icebergs. Generate with:*
`uv run src/run.py visualize dataset=ekas-hill annotation_source=detections`

## 2. Appearance embeddings

`embed` maps every detection crop to a 256-dimensional appearance vector
using a DINOv2 backbone fine-tuned with metric learning, so that crops of the
same iceberg are close in embedding space and crops of different icebergs are
far apart.

Input: `images/*` + `detections/det.txt` + `models/embedding.pt` ->  Output: `detections/embeddings.pt`.

```bash
uv run src/run.py train-embedding dataset=ekas-hill-train   # for training (optional)
uv run src/run.py embed dataset=ekas-hill                   # if not done by detect
```

## 3. Tracking

`track` links detections across frames into iceberg trajectories. Each track
is propagated with a Kalman filter; candidate matches are scored by a
weighted combination of appearance similarity, spatial distance (measured and
Kalman-predicted), and size consistency, then assigned by mutual-best
matching. Tracks unmatched for more than `max_age` frames are closed.

Input: `detections/det.txt` + `detections/embeddings.pt` ->  Output: `tracking/track.txt`.


```bash
uv run src/run.py track dataset=ekas-hill
```

<img src="https://github.com/marco-jaeger/drift/releases/download/v1.0.0/tracking_bb.gif" width="1000" alt="Detected icebergs with bounding boxes">

Tracked icebergs with persistent IDs. Generate with:
`uv run src/run.py visualize dataset=ekas-hill/clear seq_length_limit=10 make_gif=True`

## 4. Evaluation

`eval` scores the tracking against the ground truth with the standard
MOT metrics: CLEAR (recall, localisation, ID switches, fragmentations,
track coverage) and Identity (IDF1/IDR/IDP). Results are printed as tables
and written as `metrics.json`.

```bash
uv run src/run.py eval dataset=ekas-hill
```

Input: `ground_truth/gt.txt` + `tracking/track.txt` +  ->  Output: `eval/eval.txt`.

## 5. Outline extraction

`extract-outlines` segments each tracked detection into an outline polygon
(box-prompted SAM by default; a CPU-only Otsu backend is available with
`method=otsu`). Every detection receives an outline and a bounding-box
rectangle as last resort.

Input: `images/*` + `tracking/track.txt` ->  Output: `tracking/outlines.npz`.

```bash
uv run src/run.py extract-outlines dataset=ekas-hill
```

<img src="https://github.com/marco-jaeger/drift/releases/download/v1.0.0/outlines.jpg" width="1000" alt="Detected icebergs with bounding boxes">

*Extracted outlines. Generate with:
`uv run src/run.py visualize dataset=ekas-hill/clear draw_outlines=true draw_masks=true draw_boxes=False draw_ids=False`*

## 6. Georeferencing

`georeference` projects each tracked iceberg from image space onto the sea
surface in UTM coordinates, using the calibrated camera and a flat-sea model.
The tracer point is the midpoint of the outline's waterline (the two
horizontal extrema projected onto the sea plane), which also yields an
apparent width in metres per detection and summarised per iceberg by a
track median. If no outlines available, bounding boxes are used as backup.

```bash
uv run src/run.py georeference dataset=ekas-hill
```
   
Input: `camera.json` + `tracking/track.txt` + `tracking/outlines.npz` ->  Output: `georeference/*`.

<table>
  <tr>
    <td width="49%" align="center"><img src="https://github.com/marco-jaeger/drift/releases/download/v1.0.0/tracks_map.png" width="500" alt="Detected icebergs with bounding boxes"></td>
    <td width="51%" align="center"><img src="https://github.com/marco-jaeger/drift/releases/download/v1.0.0/tracks_pixel.png" width="500" alt="Trajectories in image space coloured by drift speed"></td>
  </tr>
</table>

*UTM trajectories coloured by drift speed, written automatically to
`georeference/tracks_map_velocity.png`.*

## 7. Fjord circulation

`circulation` aggregates the georeferenced trajectories into a gridded
velocity field: per-segment velocities (displacement over elapsed time) are
interpolated onto a regular UTM grid, smoothed, and rendered as vector and
streamline panels. With `size_split` (e.g. `p50` or `12m`) the field can be
computed separately for small and large icebergs, using the per-iceberg
median width.

Input: `camera.json` + `tracking/track.txt` + `tracking/outlines.npz` (+ `iceberg_sizes.csv`) ->  Output: `circulation/*`.


```bash
uv run src/run.py circulation dataset=ekas-hill size_split=p50
```

<table>
  <tr>
    <td width="50%" align="center"><img src="https://github.com/marco-jaeger/drift/releases/download/v1.0.0/circulation_vectors.png" width="500" alt="Detected icebergs with bounding boxes"></td>
    <td width="50%" align="center"><img src="https://github.com/marco-jaeger/drift/releases/download/v1.0.0/circulation_streamlines.png" width="500" alt="Trajectories in image space coloured by drift speed"></td>
  </tr>
</table>

*Streamlines of the fjord circulation, written automatically to
`circulation/*`.*


## Final structure

After a full run of all eight stages:

```
drift/
├── data/                                   # Inputs, never written to
│   └── ekas-hill/
│       ├── camera.json                     # Calibrated glimpse camera
│       ├── mask.jpg                        # Optional land mask
│       └── clear/                          # One directory per sequence
│           ├── images/                     # Time-lapse frames
│           └── ground_truth         
│               └── gt.txt                  # MOT annotations (optional)
│ 
├── models/                                 # Shared by all datasets
│   ├── detection.pt                        # train-detection
│   ├── embedding.pt                        # train-embedding
│   └── mobile_sam.pt                       # Downloaded on first use
│
└── results/                                
    └── ekas-hill/
        └── clear/
            ├── detections/
            │   ├── det.txt                 # 1. detect
            │   └── embeddings.pt           # 2. embed
            ├── ground_truth/               # embed on the ground truth
            │   └── embeddings.pt
            ├── tracking/
            │   ├── track.txt               # 3. track
            │   ├── eval.txt                # 4. eval
            │   ├── outlines.npz            # 5. extract-outlines
            │   ├── images/                 # visualize output images
            │   ├── videos/tracking.mp4     #    make_video=true
            │   └── gifs/tracking.gif       #    make_gif=true
            ├── georeference/               # 6. georeference
            │   ├── tracks_utm.csv
            │   ├── iceberg_sizes.csv       # Outline mode only
            │   ├── tracks_map[_velocity].png
            │   ├── tracks_pixel[_velocity].png
            │   └── speed_legend.png
            └── circulation/                # 7. circulation
                ├── circulation_vectors_<label>.png
                ├── circulation_streamlines_<label>.png
                ├── circulation_legend.png
                └── circulation_metadata.json                    
```
