"""Shared logging helpers for the memory package."""

from datetime import datetime


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{get_timestamp()}] {message}")
