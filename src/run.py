"""Command-line entry point for the iceberg tracking pipeline.

Usage:
    python src/run.py <command> [cfg=path/to/config.yaml] [key=value ...]

Commands:
    train-detection   Train the Faster R-CNN detection model
    detect            Detect icebergs (optionally generating embeddings inline)
    train-embedding   Train the DINOv2 appearance embedding model
    embed             Generate embeddings for detections or ground truth
    track             Run multi-object tracking
    eval              Evaluate tracking against ground truth
    visualize         Render annotated images, a video and/or a GIF
    extract-outlines  Segment tracked detections into outline polygons
    georeference      Project tracks to UTM and assemble trajectories
    circulation       Aggregate trajectories into circulation fields

Parameters use key=value syntax and override both the dataclass defaults and
the (optional) YAML config, e.g.:

    python src/run.py train-detection dataset=ekas-hill-train
    python src/run.py detect dataset=ekas-hill confidence_threshold=0.2
    python src/run.py embed dataset=ekas-hill-train embed_source=ground_truth
    python src/run.py track dataset=ekas-hill max_age=7 run_name=maxage7
    python src/run.py eval dataset=ekas-hill run_name=maxage7
    python src/run.py visualize dataset=ekas-hill annotation_source=tracking
    python src/run.py extract-outlines dataset=ekas-hill method=sam
    python src/run.py georeference dataset=ekas-hill
    python src/run.py circulation dataset=ekas-hill size_split=p50
"""

from helpers import load_config, parse_cli_args, setup_logging


def main():
    command, cfg_file, overrides = parse_cli_args()
    setup_logging()
    config = load_config(command, cfg_file, overrides)

    # Imports are local so each command only loads what it needs.
    if command == "train-detection":
        from detection import IcebergDetector
        IcebergDetector(config).train()
    elif command == "detect":
        from detection import IcebergDetector
        IcebergDetector(config).predict()
    elif command == "train-embedding":
        from embedding import IcebergEmbeddingsTrainer
        IcebergEmbeddingsTrainer(config).run_complete_pipeline()
    elif command == "embed":
        from embedding import generate_dataset_embeddings
        generate_dataset_embeddings(config)
    elif command == "track":
        from tracking import IcebergTracker
        IcebergTracker(config).track()
    elif command == "eval":
        from evaluation import eval_tracking
        eval_tracking(config)
    elif command == "visualize":
        from visualize import Visualizer
        Visualizer(config).run()
    elif command == "extract-outlines":
        from outlines import extract_dataset_outlines
        extract_dataset_outlines(config)
    elif command == "georeference":
        from georeference import run_georeference
        run_georeference(config)
    elif command == "circulation":
        from circulation import run_circulation
        run_circulation(config)


if __name__ == "__main__":
    main()
