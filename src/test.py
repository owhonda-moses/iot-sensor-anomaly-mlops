#!/usr/bin/env python

#!/usr/bin/env python

import argparse
import pandas as pd

from predict import predict_iot

# numeric shortcuts for your three MACs
DEVICES = [
    "00:0f:00:70:91:0a",
    "1c:bf:ce:15:ec:4d",
    "b8:27:eb:bf:9d:51"
]

def parse_args():
    p = argparse.ArgumentParser(
        description="Classify IoT readings: batch CSV or single reading"
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv", help="Path to CSV of raw readings")
    group.add_argument("--single", action="store_true",
                       help="Classify one reading via CLI flags")
    p.add_argument("--device",  help="Device index (1/2/3) or full MAC")
    p.add_argument("--co",       type=float, default=0.0)
    p.add_argument("--humidity", type=float, default=0.0)
    p.add_argument("--lpg",      type=float, default=0.0)
    p.add_argument("--smoke",    type=float, default=0.0)
    p.add_argument("--temp",     type=float, default=0.0)
    p.add_argument("--ts_diff",  type=float, default=0.0)
    return p.parse_args()

def resolve_device(dev_arg: str):
    if dev_arg.isdigit():
        idx = int(dev_arg)
        return DEVICES[idx-1]
    if dev_arg in DEVICES:
        return dev_arg
    raise ValueError(f"Unknown device '{dev_arg}'")

def main():
    args = parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
    else:
        if not args.device:
            raise ValueError("`--device` is required with `--single`")
        mac = resolve_device(args.device)
        df = pd.DataFrame([{
            "device":   mac,
            "ts":       pd.Timestamp.now().timestamp(),
            "co":       args.co,
            "humidity": args.humidity,
            "light":    0,
            "motion":   0,
            "lpg":      args.lpg,
            "smoke":    args.smoke,
            "temp":     args.temp,
            "ts_diff":  args.ts_diff
        }])

    y_pred, scores = predict_iot(df)

    for i, (row, y, s) in enumerate(zip(df.to_dict("records"), y_pred, scores), start=1):
        label = "ANOMALY" if y == 1 else "normal"
        print(f"{i}. Device {row['device']} → {label} (p={s:.3f})")

if __name__ == "__main__":
    main()
