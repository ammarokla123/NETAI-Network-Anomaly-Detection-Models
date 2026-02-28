#!/usr/bin/env python3
"""Launch the FastAPI inference service."""

from __future__ import annotations

import argparse
import logging

import uvicorn

from netai_anomaly.inference.service import app, load_model
from netai_anomaly.utils.config import get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Start anomaly detection inference server")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--threshold", type=float, default=None, help="Anomaly threshold")
    parser.add_argument("--device", default="auto", help="Device (cpu/cuda/auto)")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    args = parser.parse_args()

    device = get_device(args.device)
    load_model(args.checkpoint, device=device, threshold=args.threshold)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
