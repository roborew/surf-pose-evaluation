# PyTorch Optuna Constraint Patch

## Add Detection Rate Constraint to Prevent Over-Conservative Tuning

Add this code to `utils/optuna_optimizer.py` in the `objective` function, right before line 294 where `trial_score` is calculated.

### Location: After line 291, before line 293

```python
                # Calculate trial score
                trial_score = np.mean(trial_metrics) if trial_metrics else 0
```

### Add this code block BEFORE the trial_score calculation:

```python
                # FOR PYTORCH POSE: Add detection rate constraint to prevent over-conservative tuning
                # This ensures we don't converge on extreme confidence thresholds that miss poses
                if model_name == "pytorch_pose":
                    # Calculate detection rate from processed maneuvers
                    detection_rates = []
                    for i, maneuver in enumerate(maneuvers[:len(trial_metrics)]):  # Only processed maneuvers
                        try:
                            # Quick detection check - run inference and count detections
                            cap = __import__("cv2").VideoCapture(str(maneuver.file_path))
                            if maneuver.start_frame > 0:
                                cap.set(__import__("cv2").CAP_PROP_POS_FRAMES, maneuver.start_frame)

                            total_frames = 0
                            detected_frames = 0
                            frame_idx = maneuver.start_frame

                            while cap.isOpened() and frame_idx < maneuver.end_frame:
                                ret, frame = cap.read()
                                if not ret:
                                    break

                                pred = model.predict(frame)
                                total_frames += 1

                                # Check if any person detected
                                if pred and len(pred.get('keypoints', [])) > 0:
                                    detected_frames += 1

                                frame_idx += 1

                            cap.release()

                            if total_frames > 0:
                                detection_rate = detected_frames / total_frames
                                detection_rates.append(detection_rate)

                        except Exception as e:
                            logging.warning(f"Failed to check detection rate for maneuver: {e}")
                            continue

                    # Calculate average detection rate
                    avg_detection_rate = np.mean(detection_rates) if detection_rates else 0.0
                    mlflow.log_metric("detection_rate_check", avg_detection_rate)

                    # Apply penalty if detection rate is too low
                    MINIMUM_DETECTION_RATE = 0.75  # Require at least 75% detection

                    if avg_detection_rate < MINIMUM_DETECTION_RATE:
                        # Heavy penalty for low detection rate
                        penalty = (MINIMUM_DETECTION_RATE - avg_detection_rate) * 2.0
                        print(f"   ⚠️  Low detection rate: {avg_detection_rate:.2%} (min: {MINIMUM_DETECTION_RATE:.0%})")
                        print(f"       Applying penalty: -{penalty:.4f}")

                        # Reduce score significantly but don't zero it completely
                        # This guides Optuna away from over-conservative parameters
                        trial_metrics = [max(0, m - penalty) for m in trial_metrics]
                        mlflow.log_metric("detection_rate_penalty", penalty)
                    else:
                        print(f"   ✅ Detection rate OK: {avg_detection_rate:.2%}")
```

### Then continue with existing code:

```python
                # Calculate trial score (existing line 294)
                trial_score = np.mean(trial_metrics) if trial_metrics else 0
```

## Alternative: Simpler Approach (Recommended)

If the above is too complex, just add this simpler constraint:

### After line 294 (where trial_score is calculated), add:

```python
                # FOR PYTORCH POSE: Penalize over-conservative confidence thresholds
                if model_name == "pytorch_pose":
                    confidence = config.get('confidence_threshold', 0.5)

                    # Heavy penalty if confidence > 0.7 (too restrictive)
                    if confidence > 0.7:
                        penalty = (confidence - 0.7) * 2.0  # Scale penalty
                        trial_score = max(0, trial_score - penalty)
                        print(f"   ⚠️  High confidence threshold: {confidence:.3f}")
                        print(f"       Applying penalty: -{penalty:.4f}")
                        mlflow.log_metric("confidence_penalty", penalty)
```

This simpler approach prevents Optuna from selecting confidence > 0.7, which was the core problem (Oct 25 selected 0.988!).
