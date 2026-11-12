"""Explicit local development commands; no implicit remote dispatch."""

import argparse
import json

from ev_assignment_experiments.tracking.store import Ref, Store


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ev-assignment")
    parser.add_argument(
        "--store", required=True, help="Explicit archive path outside source checkout"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--experiment", required=True)
    p = sub.add_parser("run")
    p.add_argument("--id", required=True)
    p = sub.add_parser("audit")
    p.add_argument("--id", required=True)
    p = sub.add_parser("context")
    p.add_argument("--study", required=True)
    p.add_argument("--limit", type=int, default=10)
    p = sub.add_parser("sync")
    p.add_argument("--attempt", required=True)
    p.add_argument("--destination", required=True)
    sub.add_parser("index")
    p = sub.add_parser("bundle")
    p.add_argument("--attempt", action="append", required=True)
    p.add_argument("--destination", required=True)
    p = sub.add_parser("verify-bundle")
    p.add_argument("--destination", required=True)
    args = parser.parse_args(argv)
    store = Store(args.store)
    from ev_assignment_experiments.config.experiment import Experiment
    from ev_assignment_experiments.workflows.probability import plan, execute, reaudit
    from ev_assignment_experiments.tracking.context import context
    from ev_assignment_experiments.publication.bundle import build, verify

    try:
        if args.command == "plan":
            result = plan(store, Experiment.load(args.experiment)).as_dict()
        elif args.command == "run":
            ref = execute(store, Ref("runs", args.id))
            result = {"ref": ref.as_dict(), **store.read(ref)["payload"]}
        elif args.command == "audit":
            result = reaudit(store, Ref("attempts", args.id))
        elif args.command == "context":
            result = context(store, args.study, args.limit)
        elif args.command == "sync":
            result = store.sync_to(Store(args.destination), [Ref("attempts", args.attempt)])
        elif args.command == "index":
            store.rebuild_index()
            result = {"rebuilt": True}
        elif args.command == "bundle":
            result = build(store, [Ref("attempts", x) for x in args.attempt], args.destination)
        else:
            result = verify(args.destination)
        print(json.dumps(result, indent=2, allow_nan=False))
        return (
            1
            if result.get("execution") == "failed"
            or result.get("scientific") == "fail"
            or result.get("passed") is False
            else 0
        )
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
