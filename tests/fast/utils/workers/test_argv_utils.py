import argparse
import dataclasses
import json
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from miles.rollout.session.config import SessionServerConfig
from miles.router.config import MilesRouterConfig
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers import argv_utils
from miles.utils.workers.argv_utils import (
    CONFIG_JSON_FLAG,
    _actions_by_dest,
    _candidate_field_names,
    _render_fields,
    config_to_argv,
    parse_config_argv,
    render_cli_argv,
)


class _DemoConfig(FrozenStrictBaseModel):
    text: str
    count: int
    ratio: float
    enabled: bool
    maybe_timeout: float | None
    tags: list[str] | None
    options: dict[str, Any] | None


def _make_demo_config(**overrides) -> _DemoConfig:
    kwargs = dict(
        text="hello world",
        count=3,
        ratio=0.5,
        enabled=True,
        maybe_timeout=None,
        tags=["a", "b"],
        options={"nested": {"k": [1, 2]}},
    )
    kwargs.update(overrides)
    return _DemoConfig(**kwargs)


class TestConfigToArgv:
    def test_roundtrip_preserves_every_field_type(self):
        """str, int, float, bool, None, list, and nested dict all survive."""
        config = _make_demo_config()
        assert parse_config_argv(_DemoConfig, config_to_argv(config)) == config

    @pytest.mark.parametrize(
        "text",
        ["with space", 'quo"te', "single'quote", "中文字符", "line\nbreak", "--looks-like-a-flag", ""],
    )
    def test_roundtrip_survives_hostile_strings(self, text):
        """Quoting-hostile string values survive the argv boundary."""
        config = _make_demo_config(text=text)
        assert parse_config_argv(_DemoConfig, config_to_argv(config)).text == text

    def test_roundtrip_preserves_none_versus_value(self):
        """None and a real value on a nullable field stay distinguishable."""
        assert parse_config_argv(_DemoConfig, config_to_argv(_make_demo_config())).maybe_timeout is None
        config = _make_demo_config(maybe_timeout=30.0)
        assert parse_config_argv(_DemoConfig, config_to_argv(config)).maybe_timeout == 30.0

    def test_argv_is_a_flag_value_pair(self):
        """The rendered argv is exactly the config-json flag plus its payload."""
        argv = config_to_argv(_make_demo_config())
        assert argv[0] == CONFIG_JSON_FLAG
        assert len(argv) == 2

    def test_production_roundtrip_check_cannot_be_skipped(self, monkeypatch):
        """A parse that fails to reproduce the config aborts the render."""
        monkeypatch.setattr(argv_utils, "parse_config_argv", lambda config_cls, argv: _make_demo_config(count=999))
        with pytest.raises(AssertionError, match="roundtrip mismatch"):
            config_to_argv(_make_demo_config())

    def test_real_worker_configs_roundtrip(self):
        """The miles router and session server configs survive the boundary."""
        router_config = MilesRouterConfig(
            host="127.0.0.1",
            port=30080,
            max_connections=256,
            timeout=None,
            health_check_interval=10.0,
            health_check_failure_threshold=3,
        )
        assert parse_config_argv(MilesRouterConfig, config_to_argv(router_config)) == router_config

        session_config = SessionServerConfig(
            host="127.0.0.1",
            port=30100,
            instance_id="abc",
            backend_url="http://127.0.0.1:30000",
            timeout=600.0,
            hf_checkpoint="/fake/model",
            chat_template_path=None,
            tito_model="qwen3",
            apply_chat_template_kwargs={"enable_thinking": False},
            use_rollout_routing_replay=True,
            use_rollout_indexer_replay=False,
            sglang_speculative_algorithm=None,
            num_layers=None,
            moe_router_topk=None,
            save_debug_trajectory_data=None,
            lora_rank=0,
            lora_adapter_path=None,
        )
        assert parse_config_argv(SessionServerConfig, config_to_argv(session_config)) == session_config


class TestParseConfigArgv:
    def test_missing_flag_is_rejected(self):
        """An argv without the config-json flag fails to parse."""
        with pytest.raises(SystemExit):
            parse_config_argv(_DemoConfig, [])

    def test_unknown_flag_is_rejected(self):
        """Stray extra flags fail to parse instead of being ignored."""
        argv = config_to_argv(_make_demo_config())
        with pytest.raises(SystemExit):
            parse_config_argv(_DemoConfig, [*argv, "--unknown", "1"])

    def test_invalid_json_is_rejected(self):
        """A payload that is not valid JSON fails validation loudly."""
        with pytest.raises(ValidationError):
            parse_config_argv(_DemoConfig, [CONFIG_JSON_FLAG, "not json"])

    def test_extra_json_fields_are_rejected(self):
        """A payload with unknown fields violates the strict schema."""
        payload = _make_demo_config().model_dump_json().replace("{", '{"unknown_field": 1, ', 1)
        with pytest.raises(ValidationError):
            parse_config_argv(_DemoConfig, [CONFIG_JSON_FLAG, payload])


@dataclasses.dataclass
class _DemoArgs:
    name: str = "default-name"
    count: int = 0
    ratio: float = 1.0
    verbose: bool = False
    enabled: bool = True
    items: list[str] = dataclasses.field(default_factory=list)
    mapping: dict[str, str] = dataclasses.field(default_factory=dict)
    cli_filled: str | None = None


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="default-name")
    parser.add_argument("--count", type=int, default=0)
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--enabled", action="store_true", default=True)
    parser.add_argument("--items", nargs="*", default=[])
    parser.add_argument("--mapping", nargs="*", default=[])
    parser.add_argument("--cli-filled", default="filled-by-cli")
    return parser


def _from_parsed(parsed: argparse.Namespace) -> _DemoArgs:
    return _DemoArgs(
        name=parsed.name,
        count=parsed.count,
        ratio=parsed.ratio,
        verbose=parsed.verbose,
        enabled=parsed.enabled,
        items=list(parsed.items),
        mapping=dict(item.split("=", 1) for item in parsed.mapping),
        cli_filled=parsed.cli_filled,
    )


def _render(args_obj: _DemoArgs) -> list[str]:
    return render_cli_argv(args_obj, make_parser=_make_parser, from_parsed=_from_parsed)


def _parse(argv: list[str]) -> _DemoArgs:
    return _from_parsed(_make_parser().parse_args(argv))


def _make_cli_default_args(**overrides) -> _DemoArgs:
    args_obj = _parse([])
    for name, value in overrides.items():
        setattr(args_obj, name, value)
    return args_obj


class TestRenderCliArgv:
    def test_cli_defaults_render_to_an_empty_argv(self):
        """An object matching the CLI defaults needs no flags at all."""
        assert _render(_parse([])) == []

    def test_scalar_bool_list_and_dict_fields_roundtrip(self):
        """Every rendered field kind survives parse back to an equal object."""
        args_obj = _make_cli_default_args(
            name="other",
            count=3,
            ratio=0.5,
            verbose=True,
            items=["a", "b"],
            mapping={"k1": "v1", "k2": "v2"},
        )
        argv = _render(args_obj)
        assert "--verbose" in argv
        assert _parse(argv) == args_obj

    def test_cli_only_defaults_are_not_rendered(self):
        """A field keeping its CLI default (even when it differs from the
        dataclass default) stays off the command line."""
        argv = _render(_make_cli_default_args(count=3))
        assert "--cli-filled" not in argv

    def test_unrenderable_false_on_a_true_default_flag_fails_loudly(self):
        """A store-true flag whose CLI default is True cannot express False."""
        with pytest.raises(AssertionError, match="cannot be rendered"):
            _render(_make_cli_default_args(enabled=False))

    def test_roundtrip_mismatch_aborts_the_render(self):
        """A from_parsed that fails to reproduce the object aborts the render."""
        with pytest.raises(AssertionError, match="roundtrip mismatch"):
            render_cli_argv(
                _make_cli_default_args(count=3),
                make_parser=_make_parser,
                from_parsed=lambda parsed: _make_cli_default_args(count=999),
            )


@dataclasses.dataclass
class _AliasArgs:
    server_cert_path: str | None = None
    prefill_urls: list[tuple] = dataclasses.field(default_factory=list)
    dllm_fdfo: bool = True
    mm_process_config: dict[str, Any] | None = None
    plain: str = "default"


def _make_alias_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-tls-cert-path", default=None)
    parser.add_argument("--router-prefill", action="append", nargs="+", default=[])
    parser.add_argument("--router-dllm-fdfo", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--router-mm-process-config", type=json.loads, default=None)
    parser.add_argument("--router-plain", default="default")
    return parser


def _alias_from_parsed(parsed: argparse.Namespace) -> _AliasArgs:
    return _AliasArgs(
        server_cert_path=parsed.router_tls_cert_path,
        prefill_urls=[(url, int(bootstrap_port)) for url, bootstrap_port in parsed.router_prefill],
        dllm_fdfo=parsed.router_dllm_fdfo,
        mm_process_config=parsed.router_mm_process_config,
        plain=parsed.router_plain,
    )


_ALIAS_FIELD_TO_DEST = {"server_cert_path": "router_tls_cert_path", "prefill_urls": "router_prefill"}


def _render_alias(args_obj: _AliasArgs) -> list[str]:
    return render_cli_argv(
        args_obj,
        make_parser=_make_alias_parser,
        from_parsed=_alias_from_parsed,
        dest_prefix="router_",
        field_to_dest=_ALIAS_FIELD_TO_DEST,
    )


class TestRenderCliArgvAgainstTheRealParserShape:
    """The renderer must take flag names and value shapes from the parser, not from field names."""

    def test_prefixed_dest_is_resolved_without_an_explicit_mapping(self):
        """A field whose dest is merely prefixed needs no entry in field_to_dest."""
        argv = _render_alias(_AliasArgs(plain="other"))
        assert argv == ["--router-plain", "other"]

    def test_aliased_field_renders_the_registered_flag(self):
        """A field name that differs from its flag renders the flag the parser actually accepts."""
        argv = _render_alias(_AliasArgs(server_cert_path="/certs/a.pem"))
        assert argv == ["--router-tls-cert-path", "/certs/a.pem"]

    def test_boolean_optional_action_can_express_false(self):
        """A BooleanOptionalAction defaulting to True renders its negative option."""
        argv = _render_alias(_AliasArgs(dllm_fdfo=False))
        assert argv == ["--no-router-dllm-fdfo"]

    def test_json_valued_option_renders_a_single_json_token(self):
        """A dict option parsed by json.loads renders one JSON document, not key=value pairs."""
        argv = _render_alias(_AliasArgs(mm_process_config={"image": {"max_pixels": 1}}))
        assert argv == ["--router-mm-process-config", '{"image": {"max_pixels": 1}}']

    def test_append_action_repeats_the_flag_per_entry(self):
        """An append option renders once per entry, spreading each entry's tokens."""
        argv = _render_alias(_AliasArgs(prefill_urls=[("http://a:1", 9000), ("http://b:2", 9001)]))
        assert argv == ["--router-prefill", "http://a:1", "9000", "--router-prefill", "http://b:2", "9001"]

    @pytest.mark.parametrize(
        "args_obj",
        [
            _AliasArgs(server_cert_path="/certs/a.pem"),
            _AliasArgs(dllm_fdfo=False),
            _AliasArgs(mm_process_config={"image": {"max_pixels": 1}}),
            _AliasArgs(prefill_urls=[("http://a:1", 9000)]),
        ],
        ids=["aliased", "boolean-optional-false", "json-dict", "append-list"],
    )
    def test_every_shape_survives_the_production_roundtrip(self, args_obj: _AliasArgs):
        """Each shape parses back to an equal object, which is what the production assert enforces."""
        assert _alias_from_parsed(_make_alias_parser().parse_args(_render_alias(args_obj))) == args_obj

    def test_a_field_with_no_registered_option_fails_loudly(self):
        """An unrenderable field is rejected instead of being rendered as a guessed flag."""

        @dataclasses.dataclass
        class _UnknownArgs:
            not_on_the_cli: str = "default"

        with pytest.raises(AssertionError, match="cannot be rendered"):
            render_cli_argv(
                _UnknownArgs(not_on_the_cli="other"),
                make_parser=_make_alias_parser,
                from_parsed=lambda parsed: _UnknownArgs(),
                dest_prefix="router_",
            )


def _render_selected(
    args_obj: Any, *, cli_defaults: Any, make_parser: Callable[[], argparse.ArgumentParser]
) -> list[str]:
    return _render_fields(
        args_obj,
        field_names=_candidate_field_names(args_obj, cli_defaults=cli_defaults),
        actions_by_dest=_actions_by_dest(make_parser()),
        dest_prefix="",
        field_to_dest={},
    )


@dataclasses.dataclass
class _NullableArgs:
    text: str | None = "text-default"
    number: int | None = 7
    items: list[str] | None = None
    mapping: dict[str, str] | None = None


def _make_nullable_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="text-default")
    parser.add_argument("--number", type=int, default=7)
    parser.add_argument("--items", nargs="*", default=None)
    parser.add_argument("--mapping", nargs="*", default=None)
    return parser


def _render_nullable(args_obj: _NullableArgs, *, cli_defaults: _NullableArgs) -> list[str]:
    return _render_selected(args_obj, cli_defaults=cli_defaults, make_parser=_make_nullable_parser)


@dataclasses.dataclass
class _DerivedArgs:
    mode: str = "off"
    label: str | None = None
    count: int = 0
    verbose: bool = False


def _make_derived_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="off")
    parser.add_argument("--label", default=None)
    parser.add_argument("--count", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    return parser


def _from_parsed_derived(parsed: argparse.Namespace) -> _DerivedArgs:
    return _DerivedArgs(
        mode=parsed.mode,
        label=parsed.label if parsed.mode == "on" else "off-label",
        count=parsed.count,
        verbose=parsed.verbose,
    )


class TestRenderCliArgvNoneHandling:
    def test_a_none_value_is_omitted_instead_of_stringified(self):
        """A None-valued option disappears rather than rendering the literal word None."""
        argv = _render_nullable(_NullableArgs(text=None), cli_defaults=_NullableArgs())
        assert argv == []
        assert "None" not in argv

    def test_a_none_value_is_omitted_even_when_the_cli_default_differs(self):
        """None never renders a value, whatever the CLI default for that option is."""
        argv = _render_nullable(_NullableArgs(text=None, number=None), cli_defaults=_NullableArgs(text="other"))
        assert argv == []

    def test_a_none_list_or_dict_option_is_omitted(self):
        """Nullable list and dict options vanish when they are None."""
        argv = _render_nullable(_NullableArgs(items=None, mapping=None), cli_defaults=_NullableArgs(items=["a"]))
        assert argv == []

    def test_an_empty_string_value_is_still_rendered(self):
        """The empty string is a real value and keeps its flag and argument."""
        argv = _render_nullable(_NullableArgs(text=""), cli_defaults=_NullableArgs())
        assert argv == ["--text", ""]

    def test_empty_string_survives_a_full_roundtrip(self):
        """An empty-string option parses back to an equal object."""
        args_obj = _make_cli_default_args(name="")
        argv = _render(args_obj)
        assert argv == ["--name", ""]
        assert _parse(argv) == args_obj

    def test_falsy_but_not_none_values_are_rendered(self):
        """Zero and empty containers render, unlike None."""
        argv = _render_nullable(_NullableArgs(number=0, items=[], mapping={}), cli_defaults=_NullableArgs(items=["a"]))
        assert argv == ["--number", "0", "--items", "--mapping"]

    def test_a_false_flag_matching_its_cli_default_is_omitted(self):
        """A store-true flag left False is absent because that is its parsed value."""
        args_obj = _make_cli_default_args(verbose=False, count=2)
        argv = _render(args_obj)
        assert "--verbose" not in argv
        assert _parse(argv) == args_obj

    def test_a_zero_value_differing_from_the_cli_default_is_rendered(self):
        """A numeric zero that differs from the CLI default reaches the command line."""
        args_obj = _make_cli_default_args(ratio=0.0)
        argv = _render(args_obj)
        assert argv == ["--ratio", "0.0"]
        assert _parse(argv) == args_obj

    def test_a_derived_none_field_roundtrips_once_the_flag_is_omitted(self):
        """Omitting a None option lets a derived field parse back to None."""
        args_obj = _DerivedArgs(mode="on", label=None)
        argv = render_cli_argv(args_obj, make_parser=_make_derived_parser, from_parsed=_from_parsed_derived)
        assert argv == ["--mode", "on"]
        assert _from_parsed_derived(_make_derived_parser().parse_args(argv)) == args_obj

    def test_a_none_element_of_a_list_fails_loudly_instead_of_rendering_the_word_none(self):
        """A None inside a list value aborts naming the field instead of emitting "None"."""
        with pytest.raises(AssertionError, match="--items cannot be rendered.*items"):
            _render_nullable(_NullableArgs(items=["a", None]), cli_defaults=_NullableArgs())

    def test_a_none_value_in_a_dict_fails_loudly_instead_of_rendering_the_word_none(self):
        """A None inside a dict value aborts naming the field instead of emitting "key=None"."""
        with pytest.raises(AssertionError, match="--mapping cannot be rendered.*mapping"):
            _render_nullable(_NullableArgs(mapping={"k": None}), cli_defaults=_NullableArgs())

    def test_a_none_inside_a_container_never_reaches_the_rendered_argv(self):
        """Regression: rendering a list holding None used to produce the literal string "None"."""
        with pytest.raises(AssertionError, match="cannot be rendered"):
            _render(_make_cli_default_args(items=["a", None]))

    def test_render_parse_render_is_stable(self):
        """Reparsing a rendered argv and rendering again yields the identical argv."""
        args_obj = _make_cli_default_args(
            name="other",
            count=3,
            ratio=0.0,
            verbose=True,
            items=["a", "b"],
            mapping={"k1": "v1"},
            cli_filled=None,
        )
        first = _render_selected(args_obj, cli_defaults=_parse([]), make_parser=_make_parser)
        second = _render_selected(_parse(first), cli_defaults=_parse([]), make_parser=_make_parser)
        assert first == second
        assert "None" not in first


@dataclasses.dataclass
class _GraphConfig:
    backend: str = "eager"
    max_bs: int = 4


@dataclasses.dataclass
class _StructuredArgs:
    max_bs: int = 4
    graph: _GraphConfig | None = None


def _make_structured_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-bs", type=int, default=4)
    parser.add_argument("--graph", type=lambda raw: _GraphConfig(**json.loads(raw)), default=None)
    return parser


def _from_parsed_structured(parsed: argparse.Namespace) -> _StructuredArgs:
    graph = parsed.graph if parsed.graph is not None else _GraphConfig(max_bs=parsed.max_bs)
    return _StructuredArgs(max_bs=parsed.max_bs, graph=graph)


def _render_structured(args_obj: _StructuredArgs) -> list[str]:
    return render_cli_argv(args_obj, make_parser=_make_structured_parser, from_parsed=_from_parsed_structured)


class TestRenderCliArgvDataclassFields:
    def test_a_derived_dataclass_field_that_the_other_flags_imply_is_not_rendered(self):
        """Regression: a post-parse derived dataclass used to be emitted as an unparseable Python repr."""
        args_obj = _StructuredArgs(max_bs=16, graph=_GraphConfig(max_bs=16))

        assert _render_structured(args_obj) == ["--max-bs", "16"]

    def test_a_derived_dataclass_field_never_reaches_the_argv_as_a_repr(self):
        """The rendered argv must not contain the dataclass constructor text under any flag."""
        argv = _render_structured(_StructuredArgs(max_bs=16, graph=_GraphConfig(max_bs=16)))

        assert not any("_GraphConfig(" in item for item in argv)

    def test_a_user_set_dataclass_field_roundtrips_as_json(self):
        """A dataclass value the other flags do not imply is serialized so the parser can read it back."""
        args_obj = _StructuredArgs(max_bs=16, graph=_GraphConfig(backend="cuda-graph", max_bs=99))
        argv = _render_structured(args_obj)

        assert argv == ["--max-bs", "16", "--graph", json.dumps({"backend": "cuda-graph", "max_bs": 99})]
        assert _from_parsed_structured(_make_structured_parser().parse_args(argv)) == args_obj

    def test_a_dataclass_field_equal_to_the_bare_cli_default_is_not_rendered(self):
        """The plain defaults check still short-circuits before the derived-default comparison."""
        assert _render_structured(_StructuredArgs(max_bs=4, graph=_GraphConfig())) == []


@dataclasses.dataclass
class _SentinelArgs:
    disaggregation_mode: str = "null"
    load_balance_method: str = "round_robin"
    count: int = 0


def _make_sentinel_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--disaggregation-mode", default="null")
    parser.add_argument("--load-balance-method", default="auto")
    parser.add_argument("--count", type=int, default=0)
    return parser


def _from_parsed_sentinel(parsed: argparse.Namespace) -> _SentinelArgs:
    load_balance_method = parsed.load_balance_method
    if load_balance_method == "auto":
        load_balance_method = "follow_bootstrap_room" if parsed.disaggregation_mode == "prefill" else "round_robin"
    return _SentinelArgs(
        disaggregation_mode=parsed.disaggregation_mode,
        load_balance_method=load_balance_method,
        count=parsed.count,
    )


def _render_sentinel(args_obj: _SentinelArgs) -> list[str]:
    return render_cli_argv(args_obj, make_parser=_make_sentinel_parser, from_parsed=_from_parsed_sentinel)


class TestRenderCliArgvSentinelResolvedDefaults:
    def test_a_field_another_flag_re_resolves_is_still_rendered(self):
        """Regression: a prefill engine's round_robin used to be elided because the flagless baseline agreed."""
        args_obj = _SentinelArgs(disaggregation_mode="prefill", load_balance_method="round_robin")
        argv = _render_sentinel(args_obj)

        assert argv == ["--disaggregation-mode", "prefill", "--load-balance-method", "round_robin"]
        assert _from_parsed_sentinel(_make_sentinel_parser().parse_args(argv)) == args_obj

    def test_a_field_the_other_flags_already_imply_stays_off_the_command_line(self):
        """The baseline follows the emitted flags, so a value they resolve to needs no flag of its own."""
        args_obj = _SentinelArgs(disaggregation_mode="prefill", load_balance_method="follow_bootstrap_room")
        argv = _render_sentinel(args_obj)

        assert argv == ["--disaggregation-mode", "prefill"]
        assert _from_parsed_sentinel(_make_sentinel_parser().parse_args(argv)) == args_obj

    def test_an_unrelated_flag_leaves_the_resolved_field_alone(self):
        """A flag that resolves nothing keeps the first baseline as the fixed point."""
        assert _render_sentinel(_SentinelArgs(count=2)) == ["--count", "2"]


@dataclasses.dataclass
class _RequiredDemoArgs:
    model: str
    count: int = 0


def _make_required_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--count", type=int, default=0)
    return parser


def _from_parsed_required(parsed: argparse.Namespace) -> _RequiredDemoArgs:
    return _RequiredDemoArgs(model=parsed.model, count=parsed.count)


class TestRenderCliArgvRequiredArgv:
    def test_required_flags_are_emitted_exactly_once(self):
        """required_argv seeds the defaults probe and stays in the final argv."""
        args_obj = _RequiredDemoArgs(model="m", count=3)
        argv = render_cli_argv(
            args_obj,
            make_parser=_make_required_parser,
            from_parsed=_from_parsed_required,
            required_argv=["--model", "m"],
        )
        assert argv.count("--model") == 1
        assert _from_parsed_required(_make_required_parser().parse_args(argv)) == args_obj

    def test_a_parser_with_required_flags_needs_required_argv(self):
        """Without required_argv the defaults probe hits the missing required flag."""
        with pytest.raises(SystemExit):
            render_cli_argv(
                _RequiredDemoArgs(model="m"),
                make_parser=_make_required_parser,
                from_parsed=_from_parsed_required,
            )
