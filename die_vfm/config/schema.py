"""Typed mirrors of the current Hydra config surface for die_vfm.

These dataclasses are intended to reflect the current config naming and shape
used by the repository's M1 / Round1 scope. They are a typed helper layer for
humans and tools only.

This module follows repository option C:

- `schema.py` is a current-config mirror
- `schema.py` is not the canonical config contract
- current support guarantees are still defined by runtime code, `configs/`, and
  `docs/current-spec.md`
- when this module drifts from runtime behavior, runtime and `configs/` win

Important:
    Presence in this module does not automatically imply current formal support
    or runtime validation. Current formal support boundaries are defined by
    runtime code, `configs/`, and `docs/current-spec.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Literal


ResumeMode = Literal["full_resume", "warm_start"]
TrainMode = Literal["bootstrap", "round1_frozen"]
SingleImageSource = Literal["img1", "img2"]


@dataclass(frozen=True)
class ProjectConfig:
    """Top-level project metadata."""

    name: str = "die_vfm"


@dataclass(frozen=True)
class RunConfig:
    """Run directory configuration."""

    output_root: str = "runs"
    run_name: str | None = None
    save_config_snapshot: bool = True


@dataclass(frozen=True)
class SystemConfig:
    """System-level runtime configuration."""

    seed: int = 42
    device: str = "cpu"
    num_workers: int = 4


@dataclass(frozen=True)
class ResumeConfig:
    """Configuration for checkpoint resume behavior.

    This remains part of the current root config surface for bootstrap and
    future training-centric rounds. It is not part of the formal
    `round1_frozen` single-shot contract.
    """

    enabled: bool = False
    mode: ResumeMode = "full_resume"
    checkpoint_path: str | None = None
    auto_resume_latest: bool = False


@dataclass(frozen=True)
class TrainConfig:
    """Current root train/orchestration config surface.

    Important:
        `num_epochs`, `selection_metric`, and `resume` remain in the root
        config surface for bootstrap compatibility and transitional runtime
        support, but they are not part of the formal `round1_frozen`
        single-shot contract defined by `docs/current-spec.md`.
    """

    mode: TrainMode = "bootstrap"
    num_epochs: int = 1
    freeze_backbone: bool = False
    freeze_pooler: bool = False
    selection_metric: str = "knn.top1_accuracy"
    log_every_n_steps: int = 10
    run_dataloader_smoke_test: bool = True
    run_model_forward_smoke_test: bool = True
    resume: ResumeConfig = field(default_factory=ResumeConfig)


@dataclass(frozen=True)
class DataloaderConfig:
    """Dataloader configuration."""

    batch_size: int = 4
    drop_last: bool = False
    pin_memory: bool = False
    persistent_workers: bool = False


@dataclass(frozen=True)
class DummyDatasetSplitSeedConfig:
    """Deterministic split seeds for the dummy dataset."""

    train: int = 101
    val: int = 202


@dataclass(frozen=True)
class DummyDatasetConfig:
    """Current formal dummy dataset config."""

    name: Literal["dummy"] = "dummy"
    image_size: list[int] = field(default_factory=lambda: [224, 224])
    num_channels: int = 3
    num_classes: int = 5
    train_size: int = 16
    val_size: int = 8
    label_offset: int = 0
    split_seed: DummyDatasetSplitSeedConfig = field(
        default_factory=DummyDatasetSplitSeedConfig
    )


@dataclass(frozen=True)
class NormalizeConfig:
    """Per-channel normalization configuration."""

    mean: list[float] = field(default_factory=list)
    std: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class Cifar10DatasetConfig:
    """CIFAR-10 dataset config present in current config composition."""

    name: Literal["cifar10"] = "cifar10"
    root: str = "${oc.env:DIE_VFM_DATA_ROOT,./data/cifar10}"
    image_size: list[int] = field(default_factory=lambda: [224, 224])
    download: bool = False
    normalize: NormalizeConfig = field(
        default_factory=lambda: NormalizeConfig(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    )


@dataclass(frozen=True)
class DomainDatasetConfig:
    """Domain dataset config mirrored from the current ingestion contract."""

    name: Literal["domain"] = "domain"
    manifest_path: str = (
        "${oc.env:DIE_VFM_DOMAIN_MANIFEST,./data/domain/manifest.csv}"
    )
    image_size: list[int] = field(default_factory=lambda: [224, 224])
    merge_images: bool = False
    single_image_source: SingleImageSource = "img1"
    require_non_empty_val: bool = False
    did_field: str = "DID"
    img1_field: str = "IMG_1"
    img2_field: str = "IMG_2"
    source_field: str = "Source"
    label_field: str = "Label"
    path_field: str = "PATH"
    normalize: NormalizeConfig = field(
        default_factory=lambda: NormalizeConfig(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    )
    label_map: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class DummyBackboneConfig:
    """Dummy backbone config."""

    name: Literal["dummy"] = "dummy"
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    embed_dim: int = 192


@dataclass(frozen=True)
class DINOv2BackboneConfig:
    """DINOv2 backbone config present in the config surface.

    This config is part of the current formal support surface for
    `bootstrap` and `round1_frozen`. Round2+ training semantics remain out of
    current scope.
    """

    name: Literal["dinov2"] = "dinov2"
    variant: str = "vit_base"
    pretrained: bool = True
    freeze: bool = False
    return_cls_token: bool = True
    allow_network: bool = True
    local_repo_path: str | None = None
    local_checkpoint_path: str | None = None


@dataclass(frozen=True)
class MeanPoolerConfig:
    """Mean pooler config."""

    name: Literal["mean"] = "mean"
    l2_norm: bool = False


@dataclass(frozen=True)
class IdentityPoolerConfig:
    """Identity pooler config."""

    name: Literal["identity"] = "identity"
    l2_norm: bool = False


@dataclass(frozen=True)
class AttnPoolerV1Config:
    """Attention pooler v1 config."""

    name: Literal["attn_pooler_v1"] = "attn_pooler_v1"
    hidden_dim: int = 256
    output_dim: int | None = None
    dropout: float = 0.0
    l2_norm: bool = False
    use_cls_token_as_query: bool = False
    return_token_weights: bool = True


@dataclass(frozen=True)
class ModelConfig:
    """Top-level model config."""

    name: str = "die_vfm"
    return_debug_outputs: bool = True
    backbone: DummyBackboneConfig | DINOv2BackboneConfig = field(
        default_factory=DummyBackboneConfig
    )
    pooler: MeanPoolerConfig | IdentityPoolerConfig | AttnPoolerV1Config = field(
        default_factory=MeanPoolerConfig
    )


@dataclass(frozen=True)
class EmbeddingArtifactConfig:
    """Embedding artifact export config.

    Current M1 runtime-effective fields:
        - enabled
        - output_subdir
        - export_splits
        - include_test_split

    Current M1 placeholder fields:
        - save_labels
        - save_metadata
        - artifact_version
        - shard_size

    Placeholder fields are represented here because they exist in the config
    surface, but they do not currently change exporter behavior.
    """

    enabled: bool = True
    output_subdir: str = "embeddings"
    export_splits: list[str] = field(default_factory=lambda: ["train", "val"])
    include_test_split: bool = False
    save_labels: bool = True
    save_metadata: bool = True
    artifact_version: str = "v1"
    shard_size: int | None = None


@dataclass(frozen=True)
class ArtifactConfig:
    """Artifact config group."""

    embedding: EmbeddingArtifactConfig = field(
        default_factory=EmbeddingArtifactConfig
    )


@dataclass(frozen=True)
class LinearProbeInputConfig:
    """Linear probe artifact input config."""

    train_split_dir: str | None = None
    val_split_dir: str | None = None
    normalize_embeddings: bool = False
    map_location: str = "cpu"


@dataclass(frozen=True)
class LinearProbeOutputConfig:
    """Linear probe output config."""

    output_dir: str | None = None
    save_predictions: bool = True
    save_history: bool = True


@dataclass(frozen=True)
class LinearProbeModelConfig:
    """Linear probe model config."""

    bias: bool = True


@dataclass(frozen=True)
class LinearProbeTrainerConfig:
    """Linear probe trainer config."""

    batch_size: int = 256
    num_epochs: int = 50
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    optimizer_name: str = "sgd"
    momentum: float = 0.9
    device: str = "cpu"
    seed: int = 0
    selection_metric: str = "val_accuracy"


@dataclass(frozen=True)
class LinearProbeConfig:
    """Current linear probe evaluator config surface."""

    enabled: bool = True
    input: LinearProbeInputConfig = field(default_factory=LinearProbeInputConfig)
    output: LinearProbeOutputConfig = field(default_factory=LinearProbeOutputConfig)
    model: LinearProbeModelConfig = field(default_factory=LinearProbeModelConfig)
    trainer: LinearProbeTrainerConfig = field(
        default_factory=LinearProbeTrainerConfig
    )


@dataclass(frozen=True)
class KNNInputConfig:
    """kNN artifact input config."""

    train_split_dir: str | None = None
    val_split_dir: str | None = None
    normalize_embeddings: bool = False
    map_location: str = "cpu"


@dataclass(frozen=True)
class KNNOutputConfig:
    """kNN output config."""

    output_dir: str | None = None
    save_predictions: bool = True


@dataclass(frozen=True)
class KNNEvaluatorConfig:
    """kNN evaluator config."""

    k: int = 20
    metric: str = "cosine"
    weighting: str = "uniform"
    temperature: float = 0.07
    batch_size: int = 1024
    device: str = "cpu"
    topk: list[int] = field(default_factory=lambda: [1, 5])


@dataclass(frozen=True)
class KNNConfig:
    """kNN evaluator config surface."""

    enabled: bool = True
    input: KNNInputConfig = field(default_factory=KNNInputConfig)
    output: KNNOutputConfig = field(default_factory=KNNOutputConfig)
    evaluator: KNNEvaluatorConfig = field(default_factory=KNNEvaluatorConfig)


@dataclass(frozen=True)
class CentroidInputConfig:
    """Centroid artifact input config."""

    train_split_dir: str | None = None
    val_split_dir: str | None = None
    normalize_embeddings: bool = False
    map_location: str = "cpu"


@dataclass(frozen=True)
class CentroidOutputConfig:
    """Centroid output config."""

    output_dir: str | None = None
    save_predictions: bool = True


@dataclass(frozen=True)
class CentroidEvaluatorConfig:
    """Centroid evaluator config."""

    metric: str = "cosine"
    batch_size: int = 1024
    device: str = "cpu"
    topk: list[int] = field(default_factory=lambda: [1, 5])


@dataclass(frozen=True)
class CentroidConfig:
    """Centroid evaluator config surface."""

    enabled: bool = True
    input: CentroidInputConfig = field(default_factory=CentroidInputConfig)
    output: CentroidOutputConfig = field(default_factory=CentroidOutputConfig)
    evaluator: CentroidEvaluatorConfig = field(
        default_factory=CentroidEvaluatorConfig
    )


@dataclass(frozen=True)
class RetrievalInputConfig:
    """Retrieval artifact input config."""

    train_split_dir: str | None = None
    val_split_dir: str | None = None
    normalize_embeddings: bool = False
    map_location: str = "cpu"


@dataclass(frozen=True)
class RetrievalOutputConfig:
    """Retrieval output config."""

    output_dir: str | None = None
    save_predictions: bool = True


@dataclass(frozen=True)
class RetrievalEvaluatorConfig:
    """Retrieval evaluator config."""

    metric: str = "cosine"
    batch_size: int = 1024
    device: str = "cpu"
    topk: list[int] = field(default_factory=lambda: [1, 5, 10])
    save_predictions_topk: int = 10
    exclude_same_image_id: bool = False


@dataclass(frozen=True)
class RetrievalConfig:
    """Retrieval evaluator config surface."""

    enabled: bool = True
    input: RetrievalInputConfig = field(default_factory=RetrievalInputConfig)
    output: RetrievalOutputConfig = field(default_factory=RetrievalOutputConfig)
    evaluator: RetrievalEvaluatorConfig = field(
        default_factory=RetrievalEvaluatorConfig
    )


@dataclass(frozen=True)
class EvaluationConfig:
    """Top-level evaluation orchestration config."""

    run_linear_probe: bool = False
    run_knn: bool = False
    run_centroid: bool = False
    run_retrieval: bool = False
    linear_probe: LinearProbeConfig = field(default_factory=LinearProbeConfig)
    knn: KNNConfig = field(default_factory=KNNConfig)
    centroid: CentroidConfig = field(default_factory=CentroidConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)


@dataclass(frozen=True)
class CurrentRootConfig:
    """Typed mirror of the current root Hydra config surface."""

    project: ProjectConfig = field(default_factory=ProjectConfig)
    run: RunConfig = field(default_factory=RunConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    dataset: DummyDatasetConfig | Cifar10DatasetConfig | DomainDatasetConfig = field(
        default_factory=DummyDatasetConfig
    )
    model: ModelConfig = field(default_factory=ModelConfig)
    artifact: ArtifactConfig = field(default_factory=ArtifactConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
