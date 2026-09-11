class AIAgent:
    def __init__(self, threshold):
        self.threshold = threshold

    def analyze(
        self,
        detections,
        tracks,
        reid_count,
        motion_count,
        anomaly_score
    ):
        reasons = []

        # Anomaly
        if anomaly_score > self.threshold:
            reasons.append("Anomaly detected")

        # Motion
        if motion_count > 30:
            reasons.append("High motion")

        # Object tracking
        if tracks > 0:
            reasons.append("Object tracked")

        # Decision
        if anomaly_score > self.threshold * 1.5:
            status = "CRITICAL"
            action = "IMMEDIATE ALERT"
        elif anomaly_score > self.threshold:
            status = "ABNORMAL"
            action = "ALERT"
        elif motion_count > 30:
            status = "SUSPICIOUS"
            action = "MONITOR"
        else:
            status = "NORMAL"
            action = "NO ACTION"

        return {
            "status": status,
            "action": action,
            "reasons": reasons
        }