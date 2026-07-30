# DRIFT: **D**etection and **R**e-**I**dentification of Icebergs as **F**low **T**racers

[![Python](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-red.svg)](https://pytorch.org/)

This repository provides an automated multi-object tracking framework that
detects and tracks icebergs from time-lapse imagery and turns their
trajectories into velocity fields to reveal circulation patterns within
glacier fjords.

|                                                                                                                                                      |                                                                                                                                                             |                                                                                                                                                                   |
|:----------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|  <img src="https://github.com/marco-jaeger/drift/releases/download/v1.0.0/detections.jpg" width="290" alt="Detected icebergs with bounding boxes">   | <img src="https://github.com/marco-jaeger/drift/releases/download/v1.0.0/tracking.gif" width="290" alt="Tracked icebergs with outlines and persistent IDs"> | <img src="https://github.com/marco-jaeger/drift/releases/download/v1.0.0/tracks_pixel.png" width="290" alt="Trajectories in image space coloured by drift speed"> |
|                                                                    **Detection**                                                                     |                                                                  **Tracking and outlines**                                                                  |                                                                 **Trajectories in image space**                                                                   |
| <img src="https://github.com/marco-jaeger/drift/releases/download/v1.0.0/tracks_map.png" width="290" alt="UTM trajectories coloured by drift speed"> |        <img src="https://github.com/marco-jaeger/drift/releases/download/v1.0.0/circulation_vectors.png" width="290" alt="Gridded velocity vectors">        |         <img src="https://github.com/marco-jaeger/drift/releases/download/v1.0.0/circulation_streamlines.png" width="290" alt="Circulation streamlines">          |
|                                                            **Georeferenced Trajectories**                                                            |                                                                     **Velocity field**                                                                      |                                                                    **Circulation Streamlines**                                                                    |



## Key Features
- Hybrid association of appearance and motion to track hundreds to thousands icebergs per frame, across varying conditions
- Iceberg outlines from SAM, giving per-iceberg sizes and waterline-based
  tracer points
- Camera projection turning pixel tracks into georeferenced trajectories in UTM coordinates, drift velocities and streamline
  fields, optionally split by iceberg size or time window
- Start right away with the pretrained detection and embedding models, or
  train from scratch on your own annotated data
- Built-in visualization at every stage: annotated frames, videos, and GIFs



## Installation

Make sure [uv](https://docs.astral.sh/uv/) is installed on your system.

```bash
git clone https://github.com/marco-jaeger/drift.git
cd drift

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
| `visualize`       | visualize.py     | Annotated frames (boxes/IDs/outlines), video, GIF  |
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

For a step-by-step walkthrough with more detailed explanations see [docs](docs/).


**Happy tracking!**
