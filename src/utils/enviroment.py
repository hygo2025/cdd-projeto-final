from datetime import datetime, date
from typing import Optional

from config import settings
import os


def env():
    return settings.get("ENV", os.environ.get("ENV"))


def is_local():
    return env() == "dev"


def is_env_prod():
    return env() == "prod"


def runner_type():
    return settings.get("RUNNER_TYPE", "").lower()
