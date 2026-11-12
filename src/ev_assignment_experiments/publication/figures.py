"""Paper tables and vector figures drawn only from independently verified records."""

import csv
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import rc_context
from ev_assignment_experiments.tracking.store import Ref


def render_science(store, destination, manifest):
    from ev_assignment_experiments.workflows.evaluation import audit

    root = Path(destination)
    folder = root / "figures"
    folder.mkdir(exist_ok=True)
    tables = root / "tables"
    tables.mkdir(exist_ok=True)
    created = []
    with rc_context({"svg.hashsalt": "ev-assignment-v2", "font.size": 10, "pdf.fonttype": 42}):
        for a in manifest["selected_attempts"]:
            attempt = store.read(Ref(**a))["payload"]
            if attempt.get("schema_version") != 2:
                continue
            result = json.loads(store.object(Ref(**attempt["result"])))
            entries = result["plan"]["entries"]
            key = a["id"]
            with (tables / f"procurement-{key}.csv").open("w", newline="") as h:
                writer = csv.writer(h)
                writer.writerow(
                    ["vehicle", "station", "route_id", "start_h", "end_h", "contract_ac_kwh"]
                )
                writer.writerows(
                    [
                        [e["e"], e["c"], e["route_id"], e["k"] / 4, (e["r"] + 1) / 4, e["contract"]]
                        for e in entries
                    ]
                )
            fig, ax = plt.subplots(figsize=(6, 3.5), layout="constrained")
            ax.bar([str(e["e"]) for e in entries], [e["contract"] for e in entries])
            ax.set(
                xlabel="Selected vehicle",
                ylabel="Procured reserve (kWh AC)",
                title=result["method"] + " fixed plan",
            )
            if not entries:
                ax.text(0.5, 0.5, "No selected contracts", transform=ax.transAxes, ha="center")
            save(fig, folder / f"procurement-{key}")
            created.append(f"procurement-{key}")
        if not manifest.get("publication"):
            return created
        pub = store.read(Ref(**manifest["publication"]))["payload"]
        for item in pub.get("evaluations", []):
            ref = Ref(**item["run"])
            report = audit(store, ref)
            if not report["passed"]:
                raise ValueError("Cannot render incomplete common comparison")
            with (tables / f"evaluation-{ref.id}.csv").open("w", newline="") as h:
                writer = csv.writer(h)
                writer.writerow(
                    [
                        "metric",
                        "method",
                        "mean",
                        "paired_difference",
                        "paired_standard_error",
                        "samples",
                        "scope",
                    ]
                )
                for metric, summary in report["metrics"].items():
                    for method, value in summary["methods"].items():
                        writer.writerow(
                            [
                                metric,
                                method,
                                value["mean"],
                                value["paired_difference"],
                                value["paired_standard_error"],
                                summary["sample_count"],
                                report["scope"],
                            ]
                        )
            cost = report["metrics"]["cost"]
            methods = list(cost["methods"])
            values = [cost["methods"][m]["paired_difference"] for m in methods]
            errors = [cost["methods"][m]["paired_standard_error"] for m in methods]
            fig, ax = plt.subplots(figsize=(6, 3.5), layout="constrained")
            ax.bar(
                methods,
                values,
                yerr=errors if all(e is not None for e in errors) else None,
                capsize=4,
            )
            ax.axhline(0, color="black", linewidth=0.8)
            span = max(abs(v) + (e or 0) for v, e in zip(values, errors))
            # Preserve raw values; prevent floating-point noise from filling the plot.
            limit = max(1.0, 1.25 * span)
            ax.set_ylim(-limit, limit)
            if span < 1e-6:
                ax.text(
                    0.5,
                    0.92,
                    "Differences below $1e-6; raw values retained in CSV",
                    transform=ax.transAxes,
                    ha="center",
                    fontsize=8,
                )
            ax.set(
                ylabel="Paired total-cost difference (USD)",
                title=f"{item['label']} — {report['scope']}, n={cost['sample_count']}",
            )
            ax.set_xlabel("Baseline: " + cost["baseline"] + "; error bars: one paired SE")
            save(fig, folder / f"paired-cost-{ref.id}")
            created.append(f"paired-cost-{ref.id}")
    return created


def save(fig, path):
    fig.savefig(str(path) + ".svg", metadata={"Date": None}, bbox_inches="tight")
    fig.savefig(
        str(path) + ".pdf", metadata={"CreationDate": None, "ModDate": None}, bbox_inches="tight"
    )
    plt.close(fig)
