import argparse
import dataclasses
import json
from collections.abc import Callable, Mapping
from typing import TypeVar

from miles.utils.pydantic_utils import FrozenStrictBaseModel

CONFIG_JSON_FLAG = "--config-json"

_ConfigT = TypeVar("_ConfigT", bound=FrozenStrictBaseModel)
_ArgsT = TypeVar("_ArgsT")


def config_to_argv(config: FrozenStrictBaseModel) -> list[str]:
    argv = [CONFIG_JSON_FLAG, config.model_dump_json()]

    parsed = parse_config_argv(type(config), argv)
    assert parsed == config, f"config argv roundtrip mismatch: {parsed!r} != {config!r}"
    return argv


def parse_config_argv(config_cls: type[_ConfigT], argv: list[str] | None) -> _ConfigT:
    parser = argparse.ArgumentParser()
    parser.add_argument(CONFIG_JSON_FLAG, required=True)
    args = parser.parse_args(argv)
    return config_cls.model_validate_json(args.config_json)


def render_cli_argv(
    args_obj: _ArgsT,
    *,
    make_parser: Callable[[], argparse.ArgumentParser],
    from_parsed: Callable[[argparse.Namespace], _ArgsT],
    required_argv: list[str] | None = None,
    dest_prefix: str = "",
    field_to_dest: Mapping[str, str] | None = None,
) -> list[str]:
    actions_by_dest = _actions_by_dest(make_parser())

    def parse(argv: list[str]) -> _ArgsT:
        return from_parsed(make_parser().parse_args(argv))

    def render_fields(rendered_field_names: frozenset[str]) -> list[str]:
        return _render_fields(
            args_obj,
            field_names=rendered_field_names,
            actions_by_dest=actions_by_dest,
            dest_prefix=dest_prefix,
            field_to_dest=field_to_dest or {},
        )

    base_argv = list(required_argv or [])
    field_names: frozenset[str] = frozenset()
    max_passes = len(dataclasses.fields(args_obj)) + 2

    for skip_structured in (True, False):
        for _ in range(max_passes):
            next_field_names = _select_field_names(
                args_obj,
                parse=parse,
                render_fields=render_fields,
                base_argv=base_argv,
                field_names=field_names,
                skip_structured=skip_structured,
            )
            if next_field_names == field_names:
                break
            field_names = next_field_names
        else:
            raise AssertionError(
                f"cli argv rendering did not converge within {max_passes} passes: {sorted(field_names)}"
            )

    argv = base_argv + render_fields(field_names)

    parsed = parse(argv)
    assert parsed == args_obj, f"cli argv roundtrip mismatch: {parsed!r} != {args_obj!r}"
    return argv


def _select_field_names(
    args_obj: _ArgsT,
    *,
    parse: Callable[[list[str]], _ArgsT],
    render_fields: Callable[[frozenset[str]], list[str]],
    base_argv: list[str],
    field_names: frozenset[str],
    skip_structured: bool,
) -> frozenset[str]:
    def implied_by(rendered_field_names: frozenset[str]) -> _ArgsT:
        return parse(base_argv + render_fields(rendered_field_names))

    missing = _candidate_field_names(args_obj, cli_defaults=implied_by(field_names), skip_structured=skip_structured)
    still_needed = {
        name for name in field_names if getattr(args_obj, name) != getattr(implied_by(field_names - {name}), name)
    }

    return missing | frozenset(still_needed)


def _candidate_field_names(args_obj: _ArgsT, *, cli_defaults: _ArgsT, skip_structured: bool = False) -> frozenset[str]:
    return frozenset(
        field.name
        for field in dataclasses.fields(args_obj)
        if (value := getattr(args_obj, field.name)) is not None
        and not (skip_structured and dataclasses.is_dataclass(value))
        and value != getattr(cli_defaults, field.name)
    )


def _render_fields(
    args_obj: _ArgsT,
    *,
    field_names: frozenset[str],
    actions_by_dest: dict[str, argparse.Action],
    dest_prefix: str,
    field_to_dest: Mapping[str, str],
) -> list[str]:
    argv: list[str] = []
    for field in dataclasses.fields(args_obj):
        if field.name not in field_names:
            continue
        value = getattr(args_obj, field.name)

        action = _resolve_action(
            actions_by_dest, field_name=field.name, dest_prefix=dest_prefix, field_to_dest=field_to_dest
        )
        argv.extend(_render_action_argv(action, value))
    return argv


def _actions_by_dest(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    actions_by_dest: dict[str, argparse.Action] = {}
    for action in parser._actions:
        actions_by_dest.setdefault(action.dest, action)
    return actions_by_dest


def _resolve_action(
    actions_by_dest: dict[str, argparse.Action],
    *,
    field_name: str,
    dest_prefix: str,
    field_to_dest: Mapping[str, str],
) -> argparse.Action:
    if field_name in field_to_dest:
        candidates = [field_to_dest[field_name]]
    else:
        candidates = [field_name, dest_prefix + field_name] if dest_prefix else [field_name]

    for dest in candidates:
        action = actions_by_dest.get(dest)
        if action is not None and action.option_strings:
            return action

    raise AssertionError(
        f"{field_name!r} cannot be rendered: the parser registers no option for dest {candidates!r}. "
        f"Add an entry to field_to_dest, or pass the value through the native passthrough path."
    )


def _render_action_argv(action: argparse.Action, value: object) -> list[str]:
    if isinstance(action, argparse.BooleanOptionalAction):
        return [_boolean_option_string(action, value=bool(value))]

    if action.nargs == 0:
        flag = _long_option_string(action)
        assert (
            value == action.const
        ), f"{flag} cannot be rendered: the CLI only has a flag for {action.const!r}, not {value!r}"
        return [flag]

    flag = _long_option_string(action)

    if isinstance(action, argparse._AppendAction):
        argv: list[str] = []
        for item in value:
            argv.append(flag)
            argv.extend(_scalar_tokens(item))
        return argv

    if action.nargs in ("*", "+") or isinstance(action.nargs, int):
        if isinstance(value, dict):
            assert all(
                item is not None for item in value.values()
            ), f"{flag} cannot be rendered: a None value in {action.dest} cannot round-trip through the CLI"
            return [flag, *(f"{key}={item}" for key, item in value.items())]
        assert all(
            item is not None for item in value
        ), f"{flag} cannot be rendered: a None element of {action.dest} cannot round-trip through the CLI"
        return [flag, *(str(item) for item in value)]

    if dataclasses.is_dataclass(value):
        return [flag, json.dumps(dataclasses.asdict(value))]

    if isinstance(value, dict | list | tuple):
        return [flag, json.dumps(value)]

    return [flag, str(value)]


def _scalar_tokens(item: object) -> list[str]:
    if isinstance(item, list | tuple):
        return [str(element) for element in item]
    return [str(item)]


def _long_option_string(action: argparse.Action) -> str:
    long_options = [option for option in action.option_strings if option.startswith("--")]
    return long_options[0] if long_options else action.option_strings[0]


def _boolean_option_string(action: argparse.Action, *, value: bool) -> str:
    negative = [option for option in action.option_strings if option.startswith("--no-")]
    positive = [option for option in action.option_strings if not option.startswith("--no-")]
    if value:
        assert positive, f"{action.dest!r} cannot be rendered: no positive option string"
        return positive[0]
    assert negative, f"{action.dest!r} cannot be rendered: no negative option string"
    return negative[0]
