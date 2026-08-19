from pathlib import Path


def train_model(model, train_data, output_dir: str | Path):
    """Train a model and prepare output artifacts folder."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return {"model": model, "output_dir": str(output_path)}
