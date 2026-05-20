"""Runtime environment detection for ScoutSync ML."""

import os


def is_streamlit_cloud() -> bool:
    """
    True when running on Streamlit Community Cloud or lite mode is forced.

    Set SCOUTSYNC_CLOUD_LITE=true in secrets to force offline seeding locally.
    """
    if os.getenv("SCOUTSYNC_CLOUD_LITE", "false").lower() == "true":
        return True
    if os.getenv("HOME") == "/home/adminuser":
        return True
    if "STREAMLIT_SERVER_PORT" in os.environ:
        return True
    return False
