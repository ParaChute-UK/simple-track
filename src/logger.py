import pandas as pd


class EventLogger:
    def __init__(self):
        self.df = pd.DataFrame(
            columns=["event_id", "time", "event_type", "feature_id1", "feature_id2"]
        )
