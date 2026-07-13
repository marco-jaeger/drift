# DRIFT

Detection and Re-Identification of Icebergs as Flow Tracers: multi-object
tracking of icebergs in glacier time-lapse imagery, with georeferencing and
fjord circulation analysis. Detection (Faster R-CNN), appearance embeddings
(DINOv2 + metric learning), Kalman-assisted tracking, MOT evaluation,
outline extraction (SAM/Otsu), pixel-to-UTM projection (glimpse), and gridded
velocity/streamline fields.

## Installation

Make sure [uv](https://docs.astral.sh/uv/) is installed on your system.

```bash
git clone https://github.com/marco-jaeger/drift.git
cd iceberg-tracking

uv sync
```

## Pipeline

```bash
uv run src/run.py <command> [key=value ...]
```

| Command           | Module           | Purpose                                            |
|-------------------|------------------|----------------------------------------------------|
| `train-detection` | detection.py     | Train Faster R-CNN (k-fold) on the train dataset   |
| `detect`          | detection.py     | Multi-scale sliding-window detection (+embeddings) |
| `train-embedding` | embedding.py     | Train the DINOv2 appearance model                  |
| `embed`           | embedding.py     | Embeddings for detections or ground truth          |
| `track`           | tracking.py      | Multi-object tracking                              |
| `eval`            | evaluation.py    | CLEAR/Identity metrics vs ground truth             |
| `visualize`       | visualize.py     | Annotated frames (boxes/IDs/outlines) and video    |
| `extract-outlines`| outlines.py      | Segment tracked detections (SAM or Otsu)           |
| `georeference`    | georeference.py  | Project tracks to UTM, sizes, trajectory figures   |
| `circulation`     | circulation.py   | Gridded velocity fields and streamline panels      |

Defaults live in each module's config dataclass; CLI `key=value` overrides
them. Every command writes the fully resolved config next to its outputs. 

Typical chain on a new dataset:

```bash
uv run src/run.py detect dataset=ekas-hill
uv run src/run.py track dataset=ekas-hill
uv run src/run.py extract-outlines dataset=ekas-hill
uv run src/run.py georeference dataset=ekas-hill
uv run src/run.py circulation dataset=ekas-hill size_split=p50
uv run src/run.py visualize dataset=ekas-hill draw_outlines=true
```
