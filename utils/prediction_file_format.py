"""
Standardized Prediction File Format for Pose Estimation

This module defines the standardized format for storing pose predictions
across all models (MediaPipe, YOLOv8, MMPose, PyTorch, BlazePose).

The format ensures consistency for visualization generation and analysis.
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class KeypointData:
    """Standardized keypoint data structure"""

    x: float
    y: float
    z: Optional[float] = None  # For 3D models (MediaPipe, BlazePose)
    confidence: float = 0.0
    visibility: Optional[float] = None  # For models that provide visibility


@dataclass
class PersonPrediction:
    """Prediction data for a single person"""

    person_id: int
    keypoints: List[KeypointData]
    bbox: List[float]  # [x_min, y_min, x_max, y_max]
    detection_confidence: float
    num_visible_keypoints: int

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "person_id": self.person_id,
            "keypoints": [asdict(kp) for kp in self.keypoints],
            "bbox": self.bbox,
            "detection_confidence": self.detection_confidence,
            "num_visible_keypoints": self.num_visible_keypoints,
        }


@dataclass
class FramePrediction:
    """Prediction data for a single frame"""

    frame_id: int  # Frame index within maneuver
    absolute_frame_id: int  # Frame index within original video
    timestamp: float  # Timestamp in seconds
    persons: List[PersonPrediction]
    inference_time: float

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "frame_id": self.frame_id,
            "absolute_frame_id": self.absolute_frame_id,
            "timestamp": self.timestamp,
            "persons": [person.to_dict() for person in self.persons],
            "inference_time": self.inference_time,
        }


@dataclass
class ManeuverPrediction:
    """Complete prediction data for a maneuver"""

    maneuver_id: str
    maneuver_type: str
    execution_score: float
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    total_frames: int
    fps: float

    # Video metadata
    video_path: str
    video_width: int
    video_height: int

    # Model metadata
    model_name: str
    model_config: Dict[str, Any]
    keypoint_format: str  # "coco_17", "mediapipe_33", "blazepose_33"
    keypoint_names: List[str]

    # Frame predictions
    frames: List[FramePrediction]

    # Summary statistics
    total_persons_detected: int
    avg_persons_per_frame: float
    avg_inference_time: float

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "maneuver_id": self.maneuver_id,
            "maneuver_type": self.maneuver_type,
            "execution_score": self.execution_score,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "video_path": self.video_path,
            "video_width": self.video_width,
            "video_height": self.video_height,
            "model_name": self.model_name,
            "model_config": self.model_config,
            "keypoint_format": self.keypoint_format,
            "keypoint_names": self.keypoint_names,
            "frames": [frame.to_dict() for frame in self.frames],
            "total_persons_detected": self.total_persons_detected,
            "avg_persons_per_frame": self.avg_persons_per_frame,
            "avg_inference_time": self.avg_inference_time,
        }


class PredictionFileHandler:
    """Handler for reading and writing standardized prediction files"""

    def __init__(self, base_path: Optional[str] = None):
        """Initialize prediction file handler

        Args:
            base_path: Base path for storing prediction files
        """
        self.base_path = Path(base_path) if base_path else Path("predictions")
        self.base_path.mkdir(parents=True, exist_ok=True)

    def convert_model_prediction_to_standard(
        self,
        model_result: Dict[str, Any],
        frame_id: int,
        absolute_frame_id: int,
        timestamp: float,
        keypoint_format: str,
        keypoint_names: List[str],
    ) -> FramePrediction:
        """Convert model-specific prediction to standardized format

        Args:
            model_result: Raw model prediction result
            frame_id: Frame index within maneuver
            absolute_frame_id: Frame index within original video
            timestamp: Timestamp in seconds
            keypoint_format: Format of keypoints (e.g., "coco_17")
            keypoint_names: List of keypoint names

        Returns:
            Standardized FramePrediction
        """
        persons = []

        keypoints = model_result.get("keypoints", np.array([]))
        scores = model_result.get("scores", np.array([]))
        bboxes = model_result.get("bbox", np.array([]))
        inference_time = model_result.get("metadata", {}).get("inference_time", 0.0)

        # Handle empty predictions
        if len(keypoints) == 0:
            return FramePrediction(
                frame_id=frame_id,
                absolute_frame_id=absolute_frame_id,
                timestamp=timestamp,
                persons=[],
                inference_time=inference_time,
            )

        # Convert each detected person
        for person_idx in range(len(keypoints)):
            person_keypoints = keypoints[person_idx]
            person_scores = (
                scores[person_idx]
                if len(scores) > person_idx
                else np.ones(len(person_keypoints))
            )
            person_bbox = (
                bboxes[person_idx] if len(bboxes) > person_idx else [0, 0, 0, 0]
            )

            # Convert keypoints to standard format
            keypoint_data = []
            visible_count = 0

            for kp_idx, (kp_name, kp_coords) in enumerate(
                zip(keypoint_names, person_keypoints)
            ):
                if len(kp_coords) == 2:  # 2D keypoints
                    x, y = kp_coords
                    z = None
                elif len(kp_coords) == 3:  # 3D keypoints
                    x, y, z = kp_coords
                else:
                    continue

                confidence = (
                    person_scores[kp_idx] if kp_idx < len(person_scores) else 0.0
                )

                # Determine visibility based on confidence and coordinates
                is_visible = confidence > 0.3 and x > 0 and y > 0
                if is_visible:
                    visible_count += 1

                keypoint_data.append(
                    KeypointData(
                        x=float(x),
                        y=float(y),
                        z=float(z) if z is not None else None,
                        confidence=float(confidence),
                        visibility=1.0 if is_visible else 0.0,
                    )
                )

            # Calculate detection confidence (average of keypoint confidences)
            detection_confidence = (
                float(np.mean(person_scores)) if len(person_scores) > 0 else 0.0
            )

            # Handle different bbox formats
            if isinstance(person_bbox, np.ndarray):
                if person_bbox.ndim == 3:  # Shape like (1, 1, 4) from MMPose
                    bbox_flat = person_bbox.flatten()[:4]
                elif person_bbox.ndim == 2:  # Shape like (1, 4)
                    bbox_flat = person_bbox.flatten()[:4]
                else:  # Shape like (4,)
                    bbox_flat = person_bbox[:4]
                bbox_list = [float(x) for x in bbox_flat]
            elif isinstance(person_bbox, (list, tuple)):
                bbox_list = [float(x) for x in person_bbox[:4]]
            else:
                bbox_list = [0.0, 0.0, 0.0, 0.0]  # Default empty bbox

            person_prediction = PersonPrediction(
                person_id=person_idx,
                keypoints=keypoint_data,
                bbox=bbox_list,
                detection_confidence=detection_confidence,
                num_visible_keypoints=visible_count,
            )

            persons.append(person_prediction)

        return FramePrediction(
            frame_id=frame_id,
            absolute_frame_id=absolute_frame_id,
            timestamp=timestamp,
            persons=persons,
            inference_time=inference_time,
        )

    def create_maneuver_prediction(
        self,
        maneuver,  # Maneuver object
        model_name: str,
        model_config: Dict[str, Any],
        keypoint_format: str,
        keypoint_names: List[str],
        frame_predictions: List[FramePrediction],
    ) -> ManeuverPrediction:
        """Create complete maneuver prediction

        Args:
            maneuver: Maneuver object with metadata
            model_name: Name of the pose estimation model
            model_config: Model configuration parameters
            keypoint_format: Format of keypoints
            keypoint_names: List of keypoint names
            frame_predictions: List of frame predictions

        Returns:
            Complete ManeuverPrediction
        """
        # Calculate summary statistics
        total_persons = sum(len(frame.persons) for frame in frame_predictions)
        avg_persons = total_persons / len(frame_predictions) if frame_predictions else 0
        avg_inference_time = (
            np.mean([frame.inference_time for frame in frame_predictions])
            if frame_predictions
            else 0
        )

        return ManeuverPrediction(
            maneuver_id=maneuver.maneuver_id,
            maneuver_type=maneuver.maneuver_type,
            execution_score=maneuver.execution_score,
            start_time=maneuver.start_time,
            end_time=maneuver.end_time,
            start_frame=maneuver.start_frame,
            end_frame=maneuver.end_frame,
            total_frames=maneuver.total_frames,
            fps=maneuver.fps,
            video_path=maneuver.file_path,
            video_width=maneuver.clip.width,
            video_height=maneuver.clip.height,
            model_name=model_name,
            model_config=model_config,
            keypoint_format=keypoint_format,
            keypoint_names=keypoint_names,
            frames=frame_predictions,
            total_persons_detected=total_persons,
            avg_persons_per_frame=avg_persons,
            avg_inference_time=avg_inference_time,
        )

    def save_prediction_file(
        self, prediction: ManeuverPrediction, output_path: Optional[str] = None
    ) -> str:
        """Save prediction to JSON file

        Args:
            prediction: ManeuverPrediction to save
            output_path: Optional custom output path

        Returns:
            Path to saved file
        """
        if output_path is None:
            # Create model-specific subdirectory
            model_dir = self.base_path / prediction.model_name
            model_dir.mkdir(parents=True, exist_ok=True)

            # Format execution score as 2-digit integer (e.g., 05, 10)
            execution_score_str = f"{int(prediction.execution_score):02d}"

            # Extract video stem from video path for consistency with visualizations
            video_stem = Path(prediction.video_path).stem

            # Create filename that matches visualization pattern:
            # maneuver_{type}_{score}_{video_stem}_predictions.json
            filename = f"maneuver_{prediction.maneuver_type}_{execution_score_str}_{video_stem}_predictions.json"
            output_path = model_dir / filename
        else:
            output_path = Path(output_path)

        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to JSON
        with open(output_path, "w") as f:
            json.dump(prediction.to_dict(), f, indent=2, default=str)

        logger.info(f"Saved prediction file: {output_path}")
        return str(output_path)

    def load_prediction_file(self, file_path: str) -> ManeuverPrediction:
        """Load prediction from JSON file

        Args:
            file_path: Path to prediction file

        Returns:
            Loaded ManeuverPrediction
        """
        with open(file_path, "r") as f:
            data = json.load(f)

        # Reconstruct frame predictions
        frames = []
        for frame_data in data["frames"]:
            persons = []
            for person_data in frame_data["persons"]:
                keypoints = [
                    KeypointData(**kp_data) for kp_data in person_data["keypoints"]
                ]
                person = PersonPrediction(
                    person_id=person_data["person_id"],
                    keypoints=keypoints,
                    bbox=person_data["bbox"],
                    detection_confidence=person_data["detection_confidence"],
                    num_visible_keypoints=person_data["num_visible_keypoints"],
                )
                persons.append(person)

            frame = FramePrediction(
                frame_id=frame_data["frame_id"],
                absolute_frame_id=frame_data["absolute_frame_id"],
                timestamp=frame_data["timestamp"],
                persons=persons,
                inference_time=frame_data["inference_time"],
            )
            frames.append(frame)

        # Reconstruct maneuver prediction
        prediction = ManeuverPrediction(
            maneuver_id=data["maneuver_id"],
            maneuver_type=data["maneuver_type"],
            execution_score=data["execution_score"],
            start_time=data["start_time"],
            end_time=data["end_time"],
            start_frame=data["start_frame"],
            end_frame=data["end_frame"],
            total_frames=data["total_frames"],
            fps=data["fps"],
            video_path=data["video_path"],
            video_width=data["video_width"],
            video_height=data["video_height"],
            model_name=data["model_name"],
            model_config=data["model_config"],
            keypoint_format=data["keypoint_format"],
            keypoint_names=data["keypoint_names"],
            frames=frames,
            total_persons_detected=data["total_persons_detected"],
            avg_persons_per_frame=data["avg_persons_per_frame"],
            avg_inference_time=data["avg_inference_time"],
        )

        return prediction

    def get_prediction_file_path(
        self, model_name: str, maneuver_id: str, base_path: Optional[str] = None
    ) -> str:
        """Get standardized prediction file path

        Args:
            model_name: Name of the pose estimation model
            maneuver_id: Unique maneuver identifier
            base_path: Optional base path override

        Returns:
            Standardized file path
        """
        if base_path:
            path = Path(base_path)
        else:
            path = self.base_path

        # Look in model-specific subdirectory first (new format)
        model_dir = path / model_name

        # Try new format first (with maneuver type and score)
        # We'll need to search for files that match the pattern since we don't have
        # the maneuver type and score from just the maneuver_id
        if model_dir.exists():
            # Search for files matching the pattern with the maneuver_id
            pattern = f"*{maneuver_id.split('_')[-1]}_predictions.json"
            matching_files = list(model_dir.glob(pattern))
            if matching_files:
                return str(matching_files[0])  # Return first match

        # Fallback to old format
        old_filename = f"{maneuver_id}_predictions.json"
        new_path = model_dir / old_filename

        # Also check very old format for backward compatibility
        old_filename_with_model = f"{model_name}_{maneuver_id}_predictions.json"
        old_path = path / old_filename_with_model

        # Return new path (even if it doesn't exist yet, for new files)
        # The visualization system will check existence
        return str(new_path)

    def get_prediction_file_path_with_details(
        self,
        model_name: str,
        maneuver_type: str,
        execution_score: float,
        video_path: str,
        base_path: Optional[str] = None,
    ) -> str:
        """Get prediction file path with full details for new naming format

        Args:
            model_name: Name of the pose estimation model
            maneuver_type: Type of maneuver (e.g., "Pumping")
            execution_score: Execution score (will be formatted as 2-digit integer)
            video_path: Path to the video file
            base_path: Optional base path override

        Returns:
            Standardized file path with new naming format
        """
        if base_path:
            path = Path(base_path)
        else:
            path = self.base_path

        # Create model-specific subdirectory path
        model_dir = path / model_name

        # Format execution score as 2-digit integer
        execution_score_str = f"{int(execution_score):02d}"

        # Extract video stem from video path
        video_stem = Path(video_path).stem

        # Create filename that matches visualization pattern
        filename = f"maneuver_{maneuver_type}_{execution_score_str}_{video_stem}_predictions.json"

        return str(model_dir / filename)

    def list_prediction_files(
        self, model_name: Optional[str] = None, maneuver_type: Optional[str] = None
    ) -> List[str]:
        """List available prediction files

        Args:
            model_name: Filter by model name
            maneuver_type: Filter by maneuver type

        Returns:
            List of prediction file paths
        """
        files = []

        if model_name:
            # Look in model-specific directory (new format)
            model_dir = self.base_path / model_name
            if model_dir.exists():
                # New format: maneuver_{type}_{score}_{video_stem}_predictions.json
                files.extend(model_dir.glob("maneuver_*_predictions.json"))
                # Old format: {maneuver_id}_predictions.json
                files.extend(model_dir.glob("*_predictions.json"))

            # Also check very old format for backward compatibility
            old_pattern = f"{model_name}_*_predictions.json"
            files.extend(self.base_path.glob(old_pattern))
        else:
            # Search all model directories (new format)
            for model_dir in self.base_path.iterdir():
                if model_dir.is_dir():
                    # New format: maneuver_{type}_{score}_{video_stem}_predictions.json
                    files.extend(model_dir.glob("maneuver_*_predictions.json"))
                    # Old format: {maneuver_id}_predictions.json
                    files.extend(model_dir.glob("*_predictions.json"))

            # Also check very old format files in root
            files.extend(self.base_path.glob("*_*_predictions.json"))

        # Remove duplicates
        files = list(set(files))

        if maneuver_type:
            # Filter by maneuver type
            filtered_files = []
            for file_path in files:
                try:
                    # Check if it's the new format first (faster)
                    filename = Path(file_path).name
                    if (
                        filename.startswith("maneuver_")
                        and f"_{maneuver_type}_" in filename
                    ):
                        filtered_files.append(str(file_path))
                    else:
                        # Fallback to loading the file to check maneuver type
                        prediction = self.load_prediction_file(str(file_path))
                        if prediction.maneuver_type == maneuver_type:
                            filtered_files.append(str(file_path))
                except Exception as e:
                    logger.warning(f"Could not load {file_path}: {e}")
            return filtered_files

        return [str(f) for f in files]


# Keypoint format mappings
KEYPOINT_FORMATS = {
    "coco_17": [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ],
    "mediapipe_33": [
        "nose",
        "left_eye_inner",
        "left_eye",
        "left_eye_outer",
        "right_eye_inner",
        "right_eye",
        "right_eye_outer",
        "left_ear",
        "right_ear",
        "mouth_left",
        "mouth_right",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_pinky",
        "right_pinky",
        "left_index",
        "right_index",
        "left_thumb",
        "right_thumb",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_heel",
        "right_heel",
        "left_foot_index",
        "right_foot_index",
    ],
    "blazepose_33": [
        "nose",
        "left_eye_inner",
        "left_eye",
        "left_eye_outer",
        "right_eye_inner",
        "right_eye",
        "right_eye_outer",
        "left_ear",
        "right_ear",
        "mouth_left",
        "mouth_right",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_pinky",
        "right_pinky",
        "left_index",
        "right_index",
        "left_thumb",
        "right_thumb",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_heel",
        "right_heel",
        "left_foot_index",
        "right_foot_index",
    ],
}


def get_keypoint_format_for_model(model_name: str) -> str:
    """Get keypoint format for a specific model

    Args:
        model_name: Name of the pose estimation model

    Returns:
        Keypoint format string
    """
    format_mapping = {
        "yolov8_pose": "coco_17",
        "mmpose": "coco_17",
        "pytorch_pose": "coco_17",
        "mediapipe": "mediapipe_33",
        "blazepose": "blazepose_33",
    }

    return format_mapping.get(model_name, "coco_17")


def get_keypoint_names_for_model(model_name: str) -> List[str]:
    """Get keypoint names for a specific model

    Args:
        model_name: Name of the pose estimation model

    Returns:
        List of keypoint names
    """
    format_name = get_keypoint_format_for_model(model_name)
    return KEYPOINT_FORMATS[format_name]
