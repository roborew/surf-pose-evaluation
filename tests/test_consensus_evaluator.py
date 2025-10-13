import sys
from types import SimpleNamespace, ModuleType


if "torch" not in sys.modules:
    torch_stub = ModuleType("torch")
    torch_stub.cuda = SimpleNamespace(
        is_available=lambda: False, get_device_name=lambda _=0: "cpu"
    )
    torch_stub.backends = SimpleNamespace(
        mps=SimpleNamespace(is_available=lambda: False)
    )
    sys.modules["torch"] = torch_stub

if "mlflow" not in sys.modules:
    mlflow_stub = ModuleType("mlflow")
    mlflow_stub.set_tracking_uri = lambda *_args, **_kwargs: None
    mlflow_stub.start_run = lambda *_args, **_kwargs: SimpleNamespace(
        __enter__=lambda self: self, __exit__=lambda *a: None
    )
    sys.modules["mlflow"] = mlflow_stub

if "cv2" not in sys.modules:
    sys.modules["cv2"] = ModuleType("cv2")

if "scipy" not in sys.modules:
    scipy_module = ModuleType("scipy")
    scipy_spatial = ModuleType("scipy.spatial")
    scipy_distance = ModuleType("scipy.spatial.distance")

    def _cdist_stub(*_args, **_kwargs):
        raise NotImplementedError("cdist is not implemented in the test stub")

    scipy_distance.cdist = _cdist_stub
    sys.modules["scipy"] = scipy_module
    sys.modules["scipy.spatial"] = scipy_spatial
    sys.modules["scipy.spatial.distance"] = scipy_distance
    scipy_module.spatial = scipy_spatial
    scipy_spatial.distance = scipy_distance

if "sklearn" not in sys.modules:
    sklearn_module = ModuleType("sklearn")
    sklearn_cluster = ModuleType("sklearn.cluster")

    class _DBSCANStub:
        def __init__(self, *args, **kwargs):
            self.labels_ = []

        def fit(self, X):
            self.labels_ = [0 for _ in range(len(X))]
            return self

    sklearn_cluster.DBSCAN = _DBSCANStub
    sys.modules["sklearn"] = sklearn_module
    sys.modules["sklearn.cluster"] = sklearn_cluster
    sklearn_module.cluster = sklearn_cluster

if "utils.pose_evaluator" not in sys.modules:
    pose_evaluator_module = ModuleType("utils.pose_evaluator")

    class _PoseEvaluatorStub:
        def __init__(self, config):
            self.config = config
            self.run_manager = None
            self.prediction_handler = None

        def get_available_models(self):
            return []

    pose_evaluator_module.PoseEvaluator = _PoseEvaluatorStub
    sys.modules["utils.pose_evaluator"] = pose_evaluator_module

from utils.consensus_evaluator import ConsensusEvaluator


def _make_maneuver(mid: str) -> SimpleNamespace:
    return SimpleNamespace(maneuver_id=mid)


def _make_prediction(confidence: float, frames: int = 3):
    persons = [
        {
            "scores": [confidence],
            "keypoints": [{"confidence": confidence} for _ in range(3)],
        }
    ]
    return {"frames": [{"persons": persons} for _ in range(frames)]}


def test_aggregate_consensus_metrics_caps_high_confidence_to_perfect_score():
    maneuvers = [_make_maneuver("m1"), _make_maneuver("m2")]
    pytorch_predictions = [_make_prediction(2.5) for _ in maneuvers]
    reference_predictions = [_make_prediction(0.7) for _ in maneuvers]

    evaluator = ConsensusEvaluator.__new__(ConsensusEvaluator)
    evaluator.reference_models = ["pytorch_pose", "yolov8_pose", "mmpose"]
    evaluator.all_model_predictions = {
        "pytorch_pose": pytorch_predictions,
        "yolov8_pose": reference_predictions,
        "mmpose": reference_predictions,
    }

    result = evaluator._calculate_aggregate_consensus_metrics(
        "pytorch_pose", pytorch_predictions, maneuvers
    )

    assert (
        result["consensus_pck_0.2"] < 1.0
    ), "raw confidence >1 should not yield perfect consensus"
