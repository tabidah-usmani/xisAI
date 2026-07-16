"""
Computes MAE (Mean Absolute Error) and MPE (Mean Percentage Error) for
width and height measurements, from the CSV log produced by measurement.py.

Usage:
    python measurement/accuracy_report.py
"""

import csv
import os
import statistics

LOG_PATH = "measurement/outputs/measurement_log.csv"
REPORT_PATH = "measurement/outputs/accuracy_summary.md"


def main():
    if not os.path.isfile(LOG_PATH):
        print(f"⚠️ No log file found at {LOG_PATH}. Run measurement.py first "
              f"with --gt-width-mm and --gt-height-mm.")
        return

    rows = []
    with open(LOG_PATH, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["width_abs_error_mm"] == "" or row["height_abs_error_mm"] == "":
                continue  # skip rows logged without ground truth
            rows.append(row)

    if not rows:
        print("⚠️ No rows with ground-truth values found. "
              "Re-run measurement.py with --gt-width-mm and --gt-height-mm.")
        return

    width_abs_errors = [float(r["width_abs_error_mm"]) for r in rows]
    height_abs_errors = [float(r["height_abs_error_mm"]) for r in rows]
    width_pct_errors = [float(r["width_pct_error"]) for r in rows]
    height_pct_errors = [float(r["height_pct_error"]) for r in rows]

    width_mae = statistics.mean(width_abs_errors)
    height_mae = statistics.mean(height_abs_errors)
    overall_mae = statistics.mean(width_abs_errors + height_abs_errors)

    width_mpe = statistics.mean(width_pct_errors)
    height_mpe = statistics.mean(height_pct_errors)
    overall_mpe = statistics.mean(width_pct_errors + height_pct_errors)

    print("=" * 60)
    print("ACCURACY SUMMARY")
    print("=" * 60)
    print(f"Instances measured: {len(rows)}\n")

    print(f"{'Image':<20}{'GT W':>8}{'Pred W':>10}{'W Err':>10}"
          f"{'GT H':>8}{'Pred H':>10}{'H Err':>10}")
    for r in rows:
        print(f"{r['image']:<20}{float(r['gt_width_mm']):>8.1f}"
              f"{float(r['pred_width_mm']):>10.2f}{float(r['width_abs_error_mm']):>10.2f}"
              f"{float(r['gt_height_mm']):>8.1f}{float(r['pred_height_mm']):>10.2f}"
              f"{float(r['height_abs_error_mm']):>10.2f}")

    print("\n" + "-" * 60)
    print(f"Width  MAE: {width_mae:.2f} mm   |   Width  MPE: {width_mpe:.2f}%")
    print(f"Height MAE: {height_mae:.2f} mm   |   Height MPE: {height_mpe:.2f}%")
    print(f"Overall MAE: {overall_mae:.2f} mm  |   Overall MPE: {overall_mpe:.2f}%")
    print("=" * 60)

    # Write a markdown summary usable directly in MEASUREMENT_REPORT.md
    with open(REPORT_PATH, "w") as f:
        f.write("# Accuracy Summary (auto-generated)\n\n")
        f.write(f"Instances measured: {len(rows)}\n\n")
        f.write("| Image | GT Width (mm) | Pred Width (mm) | Width Error (mm) | "
                "GT Height (mm) | Pred Height (mm) | Height Error (mm) |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['image']} | {float(r['gt_width_mm']):.1f} | "
                    f"{float(r['pred_width_mm']):.2f} | {float(r['width_abs_error_mm']):.2f} | "
                    f"{float(r['gt_height_mm']):.1f} | {float(r['pred_height_mm']):.2f} | "
                    f"{float(r['height_abs_error_mm']):.2f} |\n")
        f.write(f"\n**Width MAE:** {width_mae:.2f} mm  \n")
        f.write(f"**Width MPE:** {width_mpe:.2f}%  \n")
        f.write(f"**Height MAE:** {height_mae:.2f} mm  \n")
        f.write(f"**Height MPE:** {height_mpe:.2f}%  \n")
        f.write(f"**Overall MAE:** {overall_mae:.2f} mm  \n")
        f.write(f"**Overall MPE:** {overall_mpe:.2f}%  \n")

    print(f"\n✅ Markdown summary saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()