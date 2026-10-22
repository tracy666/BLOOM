"""Public commands for the biological figure workflow.

The scientific calculations remain in the retained analysis engine. This
module supplies descriptive command names and final-paper group selections.
"""

from __future__ import annotations

import argparse
import sys

from .groups import GROUPS, native_datasets, question_selector


def build_parser():
    """Expose the original options under the final manuscript's task groups."""
    from ._engine.cli import parser as engine_parser

    native = engine_parser()
    native_commands = next(
        action.choices
        for action in native._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    parser = argparse.ArgumentParser(
        prog="python -m bloom.figures",
        description="Generate BLOOM biological figure families in manuscript order.",
        epilog="Run groups to list Results sections. Rendering consumes saved evaluation outputs.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("groups", help="List the four manuscript figure groups")
    help_text = {
        "configure": "Create a portable figure input/output configuration",
        "check": "Check scientific plotting dependencies and available resources",
        "inventory": "Inspect saved prediction and diagnostic source contracts",
        "prepare": "Cache observed-reference geometry and expression summaries",
        "plan": "List eligible figure families before rendering",
        "render": "Render one or all manuscript figure groups",
        "verify": "Check rendered figures and result manifests",
    }
    for name, description in help_text.items():
        native_name = "run" if name == "render" else name
        child = commands.add_parser(
            name,
            parents=[native_commands[native_name]],
            add_help=False,
            help=description,
            description=description,
        )
        if name in ("render", "plan"):
            child.add_argument(
                "--group",
                choices=("all", *GROUPS),
                default="all",
                help="Paper group: measured (2.2/2.3), programmes (2.4/2.5), temporal (2.6/2.7), perturbation (2.8)",
            )
    from ._radar_engine.cli import parser as radar_parser

    radar_commands = next(
        action.choices
        for action in radar_parser()._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    for name, native_name in (("radar", "run"), ("radar-audit", "audit")):
        child = commands.add_parser(
            name,
            parents=[radar_commands[native_name]],
            add_help=False,
            help=(
                "Render benchmark radars"
                if name == "radar"
                else "Export radar values and selection audit"
            ),
        )
        for action in child._actions:
            if action.dest == "dataset":
                action.choices = ("all", "prenatal", "cortex", "sted", "big", "small")
    return parser


def _without_option(arguments, name):
    """Remove a public-only option before delegating to the retained parser."""
    output = []
    index = 0
    while index < len(arguments):
        item = arguments[index]
        if item == name:
            index += 2
        elif item.startswith(name + "="):
            index += 1
        else:
            output.append(item)
            index += 1
    return output


def _translate_datasets(arguments):
    translated = list(arguments)
    for index, item in enumerate(translated):
        if item == "--dataset":
            translated[index + 1] = native_datasets(translated[index + 1])
        elif item.startswith("--dataset="):
            translated[index] = "--dataset=" + native_datasets(item.split("=", 1)[1])
        elif item == "--dataset-config":
            key, separator, value = translated[index + 1].partition("=")
            if separator:
                translated[index + 1] = native_datasets(key) + "=" + value
        elif item.startswith("--dataset-config="):
            prefix, value = item.split("=", 1)
            key, separator, path = value.partition("=")
            if separator:
                translated[index] = prefix + "=" + native_datasets(key) + "=" + path
    return translated


def main(argv=None):
    """Run a paper-aligned command without changing scientific calculations."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(arguments)
    if args.command == "groups":
        for name, spec in GROUPS.items():
            print(f"{name:14s} Results {spec['sections']:9s} {spec['description']}")
        return 0
    delegated = _without_option(arguments, "--group")
    if args.command in ("render", "plan") and args.question == "all":
        delegated = _without_option(delegated, "--question")
        delegated += ["--question", question_selector(args.group)]
    if args.command == "render":
        delegated[0] = "run"
    try:
        delegated = _translate_datasets(delegated)
        if args.command in ("radar", "radar-audit"):
            from ._radar_engine.cli import main as radar_main

            delegated[0] = "run" if args.command == "radar" else "audit"
            return radar_main(delegated)
        from ._engine.cli import main as engine_main

        return engine_main(delegated)
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
