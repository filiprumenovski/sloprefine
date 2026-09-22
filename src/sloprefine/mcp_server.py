"""MCP server: sloprefine as tools an agent can call on its own drafts.

Design notes, because the obvious wrapping is the wrong one
-----------------------------------------------------------
**Text in, not paths.** A model revising its own output has the draft in
context and no file on disk. Paths are accepted too, for a draft that is
already written out, but text is the primary input.

**Compact returns.** The agent format costs about a fifth of the JSON and
carries the same instructions. A tool that returns 8 kB of structured metrics
spends the caller's context on numbers it will not act on. `check` returns the
grouped agent report; `metrics` exists separately for when the numbers are
actually wanted.

**The stop condition travels with the result.** The failure mode of a linter
in a loop is the model revising until the count hits zero, which measurably
degrades the text (see punch.py, and the fiction corpus where human writing
scores worse than machine writing on half these rules). So `check` reports
rounds remaining, and `drift` returns a verdict whose whole job is to say
"stop, that edit did not help".

**Honest tool descriptions.** These are what the calling model reads, and they
are the only place it learns that half the rules failed their own audit. A
tool description that oversells makes the model trust the number more than the
prose, which is the opposite of what this package is for.

Run:

    pip install "sloprefine[mcp]"
    sloprefine-mcp

Claude Desktop / Claude Code config:

    {"mcpServers": {"sloprefine": {"command": "sloprefine-mcp"}}}
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from .agent import MAX_ROUNDS, style_contract
from .agent import drift as run_drift
from .agent import review as run_review
from .punch import PROFILES
from .punch import apply as apply_profile
from .punch import compute as compute_punch
from .report import Config, analyze
from .text import Document

Profile = Literal["talk", "essay", "docs"]

try:
    from mcp.server.mcpserver import MCPServer
except ImportError as exc:  # pragma: no cover - optional extra
    # ModuleNotFoundError, not ImportError: it is a subclass, so callers that
    # catch ImportError are unaffected, but pytest.importorskip only skips on
    # the narrower type and errors the whole collection on the wider one.
    raise ModuleNotFoundError(
        'the MCP server needs the optional extra: pip install "sloprefine[mcp]"'
    ) from exc


server = MCPServer(
    name="sloprefine",
    instructions=(
        "Lints prose for the published markers of machine-generated writing, "
        "for use on your own drafts before returning them.\n\n"
        "Call check after writing and act on the fixes it returns. Call drift "
        "before accepting a revision, because roughly two thirds of revision "
        "rounds in testing introduced a new hit while fixing an old one, and "
        "because revising until the count reaches zero measurably flattens "
        "prose.\n\n"
        "This tool counts patterns. It cannot tell whether the writing is "
        "good, and on the one labelled corpus available half its rules fire "
        "MORE on human writing than on machine writing. Treat a hit as "
        "something to look at, not an instruction to obey."
    ),
)


def _load(text: str | None, path: str | None) -> tuple[str, str]:
    if text:
        return "<text>", text
    if path:
        p = Path(path)
        if not p.is_file():
            raise ValueError(f"no such file: {path}")
        return str(p), p.read_text(encoding="utf-8")
    raise ValueError("pass either text or path")


def _config(profile: Profile | None, config_path: str = "") -> Config:
    """Defaults plus the profile, NOT whatever .sloprefine.toml happens to sit
    in the server's working directory.

    An MCP server is launched by the agent host from an arbitrary cwd, so
    picking up a config from there makes the same text score differently for
    reasons the caller cannot see. A config is used only when its path is
    passed explicitly.
    """
    config = (Config.from_toml(Path(config_path)) if config_path
              else Config())
    return apply_profile(profile, config) if profile else config


@server.tool(
    description=(
        "Lint a draft and return what to change. Returns PASS, or FAIL with "
        "the hit count, the worst paragraph, and one imperative fix per rule "
        "with the spans that triggered it.\n\n"
        "profile picks the register: 'talk' for spoken delivery (pins the "
        "cadence rules hard and bans generic second person), 'essay' for "
        "written prose, 'docs' for technical documentation, which enumerates "
        "things for a living and needs a loose parallelism budget.\n\n"
        "Fix the highest-severity group first; they are returned in that "
        "order. Do not revise past a PASS."
    ),
)
def check(text: str = "", path: str = "", profile: Profile | None = None,
          round_number: int = 1, config_path: str = "") -> str:
    name, body = _load(text or None, path or None)
    review = run_review(name, body, _config(profile, config_path))
    out = [review.render()]
    if not review.passed:
        left = MAX_ROUNDS - round_number
        out.append(
            f"[rounds left: {max(0, left)} of {MAX_ROUNDS}]"
            if left > 0 else
            f"[round limit reached: {MAX_ROUNDS}. Past this you are writing "
            "for the linter. Accept the draft or revise by judgement.]"
        )
    return "\n".join(out)


@server.tool(
    description=(
        "Compare two revisions and say whether the edit helped. Call this "
        "before accepting a rewrite.\n\n"
        "Verdicts: improved (take it), traded (fewer hits but bought with "
        "lexical density, meaning the revision added function words), churned "
        "(the edit did not reduce hits, so revert), choppy (the revision "
        "raised the staccato cadence, which outranks every other metric here "
        "and means revert).\n\n"
        "Only 'improved' is a reason to keep the new version."
    ),
)
def drift(before: str, after: str, profile: Profile | None = None,
          config_path: str = "") -> str:
    return run_drift(before, after, _config(profile, config_path)).render()


@server.tool(
    description=(
        "Return generation-time constraints to follow BEFORE writing, rather "
        "than fixing afterwards. Cheaper than a revision round, since a "
        "pattern never written costs nothing to remove.\n\n"
        "audience changes what is asked for: 'expert' readers weight sentiment "
        "dynamics, rhetorical variety and unpredictability; 'general' readers "
        "weight readability and lexical richness. The two populations "
        "measurably want different things, so there is no neutral setting."
    ),
)
def contract(profile: Profile | None = None,
             audience: Literal["expert", "general"] | None = None) -> str:
    disabled = ()
    if profile and profile in PROFILES:
        disabled = tuple(
            r for r, w in PROFILES[profile].pins.items() if w == 0.0
        )
    return style_contract(disabled=disabled, audience=audience)


@server.tool(
    description=(
        "Numbers rather than instructions, for when the caller wants to track "
        "a draft over time. punch is the staccato register as one score "
        "(choppiness, paragraph closers, balanced pairs, triads). cadence, "
        "shape and syntax are its components.\n\n"
        "Every threshold behind these numbers is calibrated on a single "
        "labelled document. Use them to compare two drafts of the same piece, "
        "not to judge one draft against an absolute."
    ),
)
def metrics(text: str = "", path: str = "", profile: Profile | None = None,
            config_path: str = "") -> str:
    name, body = _load(text or None, path or None)
    result = analyze(name, body, _config(profile, config_path))
    doc = Document(body, name)
    shape = (
        f"closers {result.structure.get('closer_ratio', 0):.0%}  "
        f"specifics {result.structure.get('specifics_per_1k', 0)}/1k  "
        f"vague paragraphs {result.structure.get('vague_paragraphs', 0)}"
    )
    totals = (
        f"total {result.total} hits ({result.per_1k}/1k, "
        f"weighted {result.score}/1k)"
    )
    lines = [
        compute_punch(doc).render(),
        result.cadence.render() if result.cadence else "",
        shape,
        result.templates.render() if result.templates else "",
        totals,
    ]
    return "\n".join(line for line in lines if line)


def main() -> None:
    """stdio by default, HTTP on request.

    stdio is a subprocess the host launches on the same machine, which is why
    a remote client cannot see this server no matter how its config is
    written. HTTP transport makes it reachable, and that is a different
    security posture: prose is then arriving over a socket rather than being
    read from a file the caller already had.

    Binds loopback unless told otherwise. `--host 0.0.0.0` is a deliberate act
    of exposing an unauthenticated endpoint that accepts other people's
    writing, so it should be typed rather than defaulted into.
    """
    import argparse
    import os

    p = argparse.ArgumentParser(
        prog="sloprefine-mcp",
        description="MCP server for sloprefine. Defaults to stdio, which is "
                    "what a local agent host launches.")
    p.add_argument("--transport", default="stdio",
                   choices=("stdio", "sse", "streamable-http"),
                   help="stdio (default) for a local host; streamable-http to "
                        "serve a remote client")
    p.add_argument("--host", default="127.0.0.1",
                   help="HTTP transports only. Loopback by default; "
                        "0.0.0.0 exposes it to your network (default: "
                        "%(default)s)")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")),
                   help="HTTP transports only (default: $PORT or 8000)")
    args = p.parse_args()

    if args.transport == "stdio":
        server.run()
        return
    server.run(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":  # pragma: no cover
    main()
