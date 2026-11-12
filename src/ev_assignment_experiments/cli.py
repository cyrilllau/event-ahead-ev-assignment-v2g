"""Installed research CLI. All compute, data roots and publication choices are explicit."""

import argparse
import json
from pathlib import Path
import tomllib
from ev_assignment_experiments.tracking.store import Ref, Store


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ev-assignment")
    parser.add_argument(
        "--store", required=True, help="Absolute archive path outside source checkout"
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
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--markdown", action="store_true")
    p = sub.add_parser("job")
    jobs = p.add_subparsers(dest="action", required=True)
    q = jobs.add_parser("launch")
    q.add_argument("--id", required=True)
    q.add_argument("--resume")
    q = jobs.add_parser("status")
    q.add_argument("--attempt", required=True)
    q = jobs.add_parser("recover")
    q.add_argument("--budget-group", required=True)
    p = sub.add_parser("decide")
    p.add_argument("--attempt", required=True)
    p.add_argument(
        "--outcome", required=True, choices=["adopted", "approximate", "rejected", "superseded"]
    )
    p.add_argument("--reason", required=True)
    p.add_argument("--actor", required=True)
    p.add_argument("--supersedes")
    p = sub.add_parser("next")
    p.add_argument("--study", required=True)
    p.add_argument("--action", required=True)
    p.add_argument("--reason", required=True)
    p.add_argument("--actor", required=True)
    p.add_argument("--depends-on", action="append", default=[], help="kind/sha256")
    p = sub.add_parser("sync")
    p.add_argument("--attempt", action="append")
    p.add_argument("--study")
    p.add_argument("--destination", required=True)
    p.add_argument("--mode", choices=["metadata", "results", "full"], default="results")
    p = sub.add_parser("pull")
    p.add_argument("--host", required=True)
    p.add_argument("--remote-store", required=True)
    p.add_argument("--attempt", action="append", required=True)
    p.add_argument("--mode", choices=["metadata", "results", "full"], default="results")
    p = sub.add_parser("watch")
    p.add_argument("--study", required=True)
    p.add_argument("--destination", required=True)
    p.add_argument("--host")
    p.add_argument("--remote-store")
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--interval", type=float, default=5)
    p.add_argument("--mode", choices=["metadata", "results", "full"], default="results")
    sub.add_parser("index")
    p = sub.add_parser("select")
    p.add_argument("--definition", required=True, help="JSON study/title/items")
    p = sub.add_parser("bundle")
    p.add_argument("--attempt", action="append", required=True)
    p.add_argument("--destination", required=True)
    p.add_argument("--publication")
    p = sub.add_parser("verify-bundle")
    p.add_argument("--destination", required=True)
    p = sub.add_parser("release")
    p.add_argument("--destination", required=True)
    p.add_argument("--run", required=True)
    args = parser.parse_args(argv)
    store = Store(args.store)
    from ev_assignment_experiments.workflows import probability, l08
    from ev_assignment_experiments.workflows.audit import reaudit
    from ev_assignment_experiments.config.experiment import Experiment
    from ev_assignment_experiments.config.run import RunPlan
    from ev_assignment_experiments.publication.bundle import build, verify, select
    from ev_assignment_experiments.tracking.context import context
    from ev_assignment_experiments.tracking.sync import synchronize, study_roots, watch_local
    from ev_assignment_experiments.tracking.transfer import SSHSource, FilesystemSource
    from ev_assignment_experiments.runtime import supervisor, release

    try:
        if args.command == "plan":
            value = tomllib.loads(Path(args.experiment).read_text())
            result = (
                l08.plan(store, RunPlan.parse(value))
                if value.get("schema_version") == 2
                else probability.plan(store, Experiment.parse(value))
            ).as_dict()
        elif args.command == "run":
            run = Ref("runs", args.id)
            if store.read(run)["payload"].get("schema_version") == 2:
                result = supervisor.launch(store, run)
            else:
                ref = probability.execute(store, run)
                result = dict(ref=ref.as_dict(), **store.read(ref)["payload"])
        elif args.command == "audit":
            result = reaudit(store, Ref("attempts", args.id), register=True)
        elif args.command == "context":
            result = context(
                store, args.study, args.limit, offset=args.offset, refresh=args.refresh
            )
            if args.markdown:
                print(result["markdown"])
                return 0
        elif args.command == "job":
            if args.action == "launch":
                result = supervisor.launch(
                    store,
                    Ref("runs", args.id),
                    resume=Ref("checkpoints", args.resume) if args.resume else None,
                )
            elif args.action == "status":
                result = supervisor.status(store, args.attempt)
            else:
                from ev_assignment_experiments.runtime.ledger import BudgetLedger

                result = dict(
                    settlement=BudgetLedger(store.root, args.budget_group).recover_dead_claim()
                )
        elif args.command == "decide":
            from ev_assignment_experiments.tracking.decisions import decide

            result = decide(
                store,
                Ref("attempts", args.attempt),
                args.outcome,
                args.reason,
                actor=args.actor,
                supersedes=Ref("decisions", args.supersedes) if args.supersedes else None,
            ).as_dict()
        elif args.command == "next":
            from ev_assignment_experiments.tracking.decisions import next_action

            deps = [Ref(*x.split("/", 1)) for x in args.depends_on]
            result = next_action(
                store,
                args.study,
                args.action,
                dependencies=deps,
                reason=args.reason,
                actor=args.actor,
            ).as_dict()
        elif args.command == "sync":
            if bool(args.attempt) == bool(args.study):
                raise ValueError("Choose attempts or a study")
            roots = (
                [Ref("attempts", a) for a in args.attempt]
                if args.attempt
                else study_roots(store, args.study, args.mode)
            )
            result = synchronize(
                FilesystemSource(store), Store(args.destination), roots, mode=args.mode
            )
        elif args.command == "pull":
            result = synchronize(
                SSHSource(args.host, args.remote_store),
                store,
                [Ref("attempts", a) for a in args.attempt],
                mode=args.mode,
            )
        elif args.command == "watch":
            if args.host:
                from ev_assignment_experiments.tracking.sync import watch_ssh

                if not args.remote_store:
                    raise ValueError("Explicit remote store required")
                result = watch_ssh(
                    SSHSource(args.host, args.remote_store),
                    Store(args.destination),
                    args.study,
                    seconds=args.seconds,
                    interval=args.interval,
                    mode=args.mode,
                )
            else:
                result = watch_local(
                    store,
                    Store(args.destination),
                    args.study,
                    seconds=args.seconds,
                    interval=args.interval,
                    mode=args.mode,
                )
        elif args.command == "index":
            store.rebuild_index()
            result = {"rebuilt": True}
        elif args.command == "select":
            result = select(store, **json.loads(Path(args.definition).read_text())).as_dict()
        elif args.command == "bundle":
            result = build(
                store,
                [Ref("attempts", a) for a in args.attempt],
                args.destination,
                publication=Ref("publications", args.publication) if args.publication else None,
            )
        elif args.command == "release":
            run = store.read(Ref("runs", args.run))["payload"]
            result = release.build(store.object(Ref(**run["source"])), args.destination)
        else:
            result = verify(args.destination)
        print(json.dumps(result, indent=2, allow_nan=False))
        return int(
            result.get("execution") == "failed"
            or result.get("scientific") == "fail"
            or result.get("passed") is False
        )
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
