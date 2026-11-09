"""Alien First Contact — dashboard.

    streamlit run app.py

Browse every run: success curve, the invented language drawn as a map, a round-by-round
conversation replay, and each agent's private notes. Opens runs.db read-only, so it is
safe to use while experiments are running.
"""

from __future__ import annotations

import html
import json
import sqlite3
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from analyze import LEVEL_LABEL, lexicon_of, load_run, rolling_rate
from costs import estimate_cost
from metrics import topsim_test
from world import COLORS, GRID_RADIUS, SHAPES, Cell, Obj, make_objects

DB = Path(__file__).parent / "runs.db"

# Palette (dark surface). Text stays in ink tokens; color carries meaning only.
INK, MUTED, LINE = "#ece9f5", "#8b8fa3", "#232838"
ACCENT, TEAL, GOOD, BAD = "#9085e9", "#2dd4bf", "#34d399", "#f87171"
OBJECT_COLORS = {"red": "#ef4444", "blue": "#3b82f6", "green": "#22c55e",
                 "yellow": "#facc15", "purple": "#a855f7", "orange": "#f97316"}
SHAPE_SVG = {
    "circle": '<circle cx="12" cy="12" r="9"/>',
    "square": '<rect x="3.5" y="3.5" width="17" height="17" rx="2.5"/>',
    "triangle": '<polygon points="12,3 21.5,20 2.5,20"/>',
    "star": '<polygon points="12,2 14.9,8.6 22,9.2 16.6,13.9 18.2,21 12,17.3 5.8,21 7.4,13.9 2,9.2 9.1,8.6"/>',
    "diamond": '<polygon points="12,2 22,12 12,22 2,12"/>',
    None: '<rect x="3" y="3" width="18" height="18" rx="6"/>',
}

st.set_page_config(page_title="Alien First Contact", page_icon="👽", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono&family=Noto+Sans+Math&family=Noto+Sans+Symbols&family=Noto+Sans+Symbols+2&display=swap');
html, body, .stApp, [data-testid="stMarkdownContainer"] {{ font-family: 'Inter', sans-serif; }}
.stApp {{
  background:
    radial-gradient(1100px 520px at 8% -8%, rgba(144,133,233,.20), transparent 60%),
    radial-gradient(900px 480px at 105% 0%, rgba(45,212,191,.10), transparent 60%),
    #0a0c12;
}}
.stApp::before {{
  content: ""; position: fixed; inset: 0; pointer-events: none; opacity: .55;
  background-image:
    radial-gradient(1px 1px at 12% 22%, #fff 50%, transparent 51%),
    radial-gradient(1px 1px at 68% 14%, #fff 50%, transparent 51%),
    radial-gradient(1.5px 1.5px at 84% 64%, #cfc8ff 50%, transparent 51%),
    radial-gradient(1px 1px at 32% 78%, #fff 50%, transparent 51%),
    radial-gradient(1px 1px at 52% 46%, #fff 50%, transparent 51%),
    radial-gradient(1.5px 1.5px at 22% 56%, #9ff 50%, transparent 51%),
    radial-gradient(1px 1px at 92% 30%, #fff 50%, transparent 51%),
    radial-gradient(1px 1px at 42% 8%, #fff 50%, transparent 51%);
}}
#MainMenu, footer, [data-testid="stDecoration"] {{ visibility: hidden; }}
.block-container {{ padding-top: 2.2rem; max-width: 1240px; }}
h1, h2, h3, h4 {{ font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -.02em; }}
section[data-testid="stSidebar"] {{ background: rgba(10,12,18,.94); border-right: 1px solid #1b2030; }}
div[data-testid="stTabs"] button p {{ font-family: 'Space Grotesk', sans-serif; font-size: 1rem; }}

.glyph, .chip {{ font-family: 'Noto Sans Math', 'Noto Sans Symbols', 'Noto Sans Symbols 2', 'Segoe UI Symbol', sans-serif; }}
.hero {{ margin-bottom: 1.4rem; }}
.hero .eyebrow {{ color: {TEAL}; font-size: .78rem; letter-spacing: .18em; text-transform: uppercase; }}
.hero h1 {{ font-size: 3.1rem; line-height: 1.05; margin: .35rem 0 .6rem; padding: 0;
  background: linear-gradient(92deg, {INK} 10%, {ACCENT} 55%, {TEAL});
  -webkit-background-clip: text; background-clip: text; color: transparent; }}
.hero p {{ color: {MUTED}; max-width: 760px; font-size: 1.04rem; line-height: 1.6; margin: 0; }}
.hero .float {{ font-size: 1.5rem; letter-spacing: .5rem; color: rgba(144,133,233,.55); margin-top: .7rem; }}

.kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: .6rem 0 1.2rem; }}
.kpi {{ background: rgba(18,22,34,.78); border: 1px solid {LINE}; border-radius: 16px; padding: 14px 16px;
  backdrop-filter: blur(6px); }}
.kpi .label {{ color: {MUTED}; font-size: .72rem; letter-spacing: .1em; text-transform: uppercase; }}
.kpi .value {{ font-family: 'Space Grotesk', sans-serif; font-size: 1.75rem; color: {INK}; margin-top: 2px; }}
.kpi .sub {{ color: {MUTED}; font-size: .78rem; }}
.kpi.accent {{ border-color: rgba(144,133,233,.45); box-shadow: 0 0 0 1px rgba(144,133,233,.12), 0 8px 30px rgba(144,133,233,.10); }}

.runhead {{ display: flex; flex-wrap: wrap; align-items: center; gap: 10px; margin: .2rem 0 .3rem; }}
.runhead h2 {{ margin: 0 !important; padding: 0 !important; font-size: 1.7rem; }}
.tag {{ display: inline-block; padding: 3px 11px; border-radius: 999px; border: 1px solid #2a3042;
  color: {MUTED}; font-size: .78rem; background: rgba(18,22,34,.6); }}
.tag.live {{ color: {TEAL}; border-color: rgba(45,212,191,.45); }}
.alphabet {{ margin: .5rem 0 1rem; color: {MUTED}; font-size: .85rem; }}

.chip {{ display: inline-flex; align-items: center; justify-content: center; min-width: 2.15rem; height: 2.15rem;
  padding: 0 .35rem; margin: 2px; border-radius: 10px; font-size: 1.2rem; color: {INK};
  background: rgba(144,133,233,.13); border: 1px solid rgba(144,133,233,.38); }}
.chip.big {{ min-width: 3rem; height: 3rem; font-size: 1.75rem; border-radius: 14px;
  box-shadow: 0 0 24px rgba(144,133,233,.25); }}
.chip.ghost {{ background: transparent; border-style: dashed; color: {MUTED}; }}

.matrix, .matrix tr, .matrix th, .mini, .mini tr {{ border: none !important; background: transparent !important; }}
.matrix {{ border-collapse: separate !important; border-spacing: 7px; margin: 0 auto; }}
.matrix th {{ color: {MUTED}; font-weight: 500; font-size: .8rem; text-transform: capitalize; padding: 4px; }}
.matrix th svg {{ display: block; margin: 0 auto 3px; }}
.matrix th.row svg {{ display: inline-block; vertical-align: middle; margin: 0 6px 0 0; }}
.matrix td {{ background: rgba(18,22,34,.85) !important; border: 1px solid {LINE} !important; border-radius: 14px;
  text-align: center; padding: 9px 6px; min-width: 92px; }}
.matrix td.held {{ border: 1px dashed {TEAL} !important; background: rgba(45,212,191,.07) !important; }}
.matrix td .note {{ display: block; font-size: .62rem; color: {TEAL}; letter-spacing: .08em; margin-top: 3px; }}
.grid7 td {{ min-width: 64px; padding: 7px 3px; }}
.grid7 td.me {{ background: rgba(144,133,233,.16) !important; border-color: rgba(144,133,233,.5) !important; color: {INK};
  font-family: 'Space Grotesk', sans-serif; font-size: .8rem; }}

.panel {{ background: rgba(18,22,34,.82); border: 1px solid {LINE}; border-radius: 18px; padding: 18px 20px; }}
.panel .who {{ font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; color: {ACCENT}; margin-bottom: 10px; }}
.panel .reason {{ color: {MUTED}; font-size: .88rem; line-height: 1.55; margin-top: 14px; border-left: 2px solid #2a3042; padding-left: 12px; }}
.wire {{ display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 10px; padding-top: 40px; }}
.wire .lbl {{ color: {MUTED}; font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; }}
.wire .arrow {{ color: {ACCENT}; font-size: 1.6rem; }}
.obj {{ display: flex; align-items: center; gap: 10px; padding: 8px 12px; border-radius: 12px;
  border: 1px solid {LINE}; background: rgba(10,12,18,.6); margin: 5px 0; color: {INK}; }}
.obj .n {{ color: {MUTED}; font-family: 'JetBrains Mono', monospace; font-size: .8rem; width: 1.2rem; }}
.obj.target {{ border-color: rgba(144,133,233,.7); }}
.obj.picked.ok {{ border-color: {GOOD}; background: rgba(52,211,153,.08); }}
.obj.picked.bad {{ border-color: {BAD}; background: rgba(248,113,113,.08); }}
.obj .flag {{ margin-left: auto; font-size: .7rem; letter-spacing: .08em; text-transform: uppercase; color: {MUTED}; }}
.verdict {{ font-family: 'Space Grotesk', sans-serif; font-size: 1.05rem; margin-top: 12px; }}
.verdict.ok {{ color: {GOOD}; }} .verdict.bad {{ color: {BAD}; }}
.mini {{ border-collapse: separate !important; border-spacing: 3px; margin: 4px 0; }}
.mini td {{ width: 28px; height: 28px; border-radius: 6px; border: none !important; background: rgba(255,255,255,.04); text-align: center;
  font-family: 'JetBrains Mono', monospace; font-size: .78rem; color: {MUTED}; padding: 0; }}
.mini td.s {{ background: rgba(144,133,233,.25); color: {INK}; }}
.mini td.t {{ background: {ACCENT}; color: #0a0c12; font-weight: 700; }}
.mini td.c {{ background: rgba(45,212,191,.18); color: {INK}; }}
.mini td.c.tg {{ background: rgba(144,133,233,.55); }}
.mini td.c.pk {{ outline: 2px solid {BAD}; }}
.mini td.c.pk.ok {{ outline-color: {GOOD}; }}
.notecard {{ background: rgba(18,22,34,.82); border: 1px solid {LINE}; border-radius: 16px; padding: 18px 20px;
  white-space: pre-wrap; line-height: 1.6; color: {INK}; font-family: 'Inter', 'Noto Sans Math', 'Noto Sans Symbols 2', sans-serif; }}
.caption {{ color: {MUTED}; font-size: .84rem; margin-top: .5rem; line-height: 1.5; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- data


def connect() -> sqlite3.Connection:
    return sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)


@st.cache_data(ttl=15)
def runs_table() -> pd.DataFrame:
    with connect() as conn:
        df = pd.read_sql_query(
            "SELECT run_id, level, experiment, feedback, model, input_tokens, output_tokens,"
            " finished_at,"
            " (SELECT COUNT(*) FROM rounds r WHERE r.run_id = runs.run_id) AS rounds_played,"
            " (SELECT COALESCE(SUM(success), 0) FROM rounds r WHERE r.run_id = runs.run_id)"
            "   AS successes"
            " FROM runs WHERE mock = 0 ORDER BY run_id DESC", conn)
    df = df[df.rounds_played > 0].copy()
    df["success"] = 100 * df.successes / df.rounds_played
    df["cost"] = [estimate_cost(m, i, o) or 0.0
                  for m, i, o in zip(df.model, df.input_tokens, df.output_tokens, strict=True)]
    df["running"] = [bool(pd.isna(f)) for f in df.finished_at]
    return df


@st.cache_data(ttl=15)
def run_details(run_id: int):
    with connect() as conn:
        run, rows = load_run(conn, run_id)
        detail = pd.read_sql_query(
            "SELECT round_no, phase, speaker_id, listener_id, target, candidates, message,"
            " valid, error, picked, success, speaker_reasoning, listener_reasoning"
            " FROM rounds WHERE run_id=? ORDER BY round_no", conn, params=(run_id,))
        notes = pd.read_sql_query("SELECT round_no, agent, notes FROM notes WHERE run_id=?"
                                  " ORDER BY round_no", conn, params=(run_id,))
    return run, rows, detail, notes


@st.cache_data(ttl=60)
def structure(run_id: int, speaker: str, level: int, names_msgs: tuple):
    """Topographic similarity for one dialect (cached: the permutation test takes a moment)."""
    by_name = {t.name: t for t in make_objects(level)}
    return topsim_test({by_name[n]: list(m) for n, m in names_msgs}, shuffles=300)


# ---------------------------------------------------------------- small html builders

esc = html.escape


def chips(symbols, big: bool = False) -> str:
    if isinstance(symbols, str):
        symbols = json.loads(symbols) if symbols.startswith("[") else symbols.split()
    if not symbols:
        return '<span class="chip ghost">∅</span>'
    cls = "chip big" if big else "chip"
    return "".join(f'<span class="{cls}">{esc(s)}</span>' for s in symbols)


def shape_svg(shape: str | None, color: str | None, size: int = 22) -> str:
    fill = OBJECT_COLORS.get(color or "", "#5b6075")
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="{fill}" '
            f'style="flex:none">{SHAPE_SVG.get(shape, SHAPE_SVG[None])}</svg>')


def kpis(items: list[tuple[str, str, str, bool]]) -> str:
    cards = "".join(
        f'<div class="kpi{" accent" if hi else ""}"><div class="label">{esc(label)}</div>'
        f'<div class="value">{value}</div><div class="sub">{esc(sub)}</div></div>'
        for label, value, sub, hi in items)
    return f'<div class="kpis">{cards}</div>'


def mini_grid(marks: dict[tuple[int, int], tuple[str, str]]) -> str:
    """7x7 grid; marks map (dx, dy) -> (text, css classes). S marks the centre."""
    r = GRID_RADIUS
    rows = []
    for dy in range(r, -r - 1, -1):
        cells = []
        for dx in range(-r, r + 1):
            if (dx, dy) == (0, 0):
                cells.append('<td class="s">S</td>')
            elif (dx, dy) in marks:
                text, cls = marks[(dx, dy)]
                cells.append(f'<td class="{cls}">{esc(text)}</td>')
            else:
                cells.append("<td></td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table class="mini">{"".join(rows)}</table>'


def object_row(thing, n: int | None = None, classes: str = "", flag: str = "") -> str:
    num = f'<span class="n">{n}</span>' if n is not None else ""
    icon = shape_svg(thing.shape, thing.color) if isinstance(thing, Obj) else ""
    flag_html = f'<span class="flag">{esc(flag)}</span>' if flag else ""
    return f'<div class="obj {classes}">{num}{icon}<span>{esc(thing.name)}</span>{flag_html}</div>'


# ---------------------------------------------------------------- chart


def success_chart(rows, n_candidates: int, window: int = 10) -> alt.LayerChart:
    frames = []
    for phase in dict.fromkeys(r.phase for r in rows):
        rs = [r for r in rows if r.phase == phase]
        frames.append(pd.DataFrame({"round": [r.round_no for r in rs],
                                    "success": rolling_rate([r.success for r in rs], window),
                                    "phase": phase}))
    df = pd.concat(frames)
    x = alt.X("round:Q", title="Round", axis=alt.Axis(grid=False))
    y = alt.Y("success:Q", title=f"Success, last {window} rounds",
              scale=alt.Scale(domain=[0, 100]),
              axis=alt.Axis(values=[0, 25, 50, 75, 100], labelExpr="datum.value + '%'"))
    area = alt.Chart(df).mark_area(
        interpolate="monotone",
        color=alt.Gradient(gradient="linear", x1=1, x2=1, y1=1, y2=0,
                           stops=[alt.GradientStop(color="rgba(144,133,233,0)", offset=0),
                                  alt.GradientStop(color="rgba(144,133,233,0.32)", offset=1)]),
    ).encode(x=x, y=y, detail="phase:N")
    line = alt.Chart(df).mark_line(color=ACCENT, strokeWidth=2.4, interpolate="monotone").encode(
        x=x, y=y, detail="phase:N",
        tooltip=[alt.Tooltip("round:Q"), alt.Tooltip("success:Q", format=".0f", title="success %"),
                 alt.Tooltip("phase:N")])
    base_df = pd.DataFrame({"y": [100 / n_candidates], "label": ["random guessing"]})
    base = alt.Chart(base_df).mark_rule(color=MUTED, strokeDash=[4, 4]).encode(y="y:Q")
    base_txt = alt.Chart(base_df).mark_text(align="left", dx=6, dy=-8, color=MUTED, fontSize=11
                                            ).encode(y="y:Q", text="label:N", x=alt.value(0))
    layers = [area, line, base, base_txt]
    starts = df.groupby("phase", sort=False)["round"].min().reset_index().iloc[1:]
    if not starts.empty:
        layers.append(alt.Chart(starts).mark_rule(color="#3a4157", strokeDash=[2, 4]).encode(x="round:Q"))
        layers.append(alt.Chart(starts).mark_text(align="left", dx=6, color=TEAL, fontSize=11)
                      .encode(x="round:Q", text="phase:N", y=alt.value(10)))
    return (alt.layer(*layers).properties(height=320)
            .configure(background="transparent")
            .configure_view(stroke=None)
            .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor="#1c2130",
                            domainColor="#2a3042", tickColor="#2a3042", labelFont="Inter",
                            titleFont="Inter", titleFontWeight=500))


# ---------------------------------------------------------------- language map


def language_map(run: dict, lexicon: dict, rows) -> str:
    """The invented language drawn as a map: each meaning's cell holds its alien word."""
    level = run["level"]
    by_name = {t.name: list(m) for t, m in lexicon.items()}
    held = set(run.get("held_out") or [])
    first_test: dict[str, list[str]] = {}
    for r in rows:
        if r.phase == "test" and r.target not in first_test:
            first_test[r.target] = json.loads(r.message)

    def cell(name: str) -> str:
        if name in held:
            msg = first_test.get(name)
            inner = chips(msg) if msg else '<span class="chip ghost">?</span>'
            return f'<td class="held">{inner}<span class="note">NEVER SEEN</span></td>'
        msg = by_name.get(name)
        inner = chips(msg) if msg else '<span class="chip ghost">·</span>'
        return f"<td>{inner}</td>"

    if level == 1:
        head = "".join(f"<th>{shape_svg(None, c, 26)}{c}</th>" for c in COLORS)
        return f'<table class="matrix"><tr>{head}</tr><tr>{"".join(cell(c) for c in COLORS)}</tr></table>'
    if level == 2:
        head = "<th></th>" + "".join(f"<th>{shape_svg(s, None, 22)}{s}</th>" for s in SHAPES)
        body = "".join(
            f'<tr><th class="row">{shape_svg(None, c, 16)}{c}</th>'
            + "".join(cell(f"{c} {s}") for s in SHAPES) + "</tr>"
            for c in COLORS)
        return f'<table class="matrix"><tr>{head}</tr>{body}</table>'
    r = GRID_RADIUS
    body = ""
    for dy in range(r, -r - 1, -1):
        body += "<tr>" + "".join('<td class="me">YOU</td>' if (dx, dy) == (0, 0)
                                 else cell(Cell(dx, dy).name) for dx in range(-r, r + 1)) + "</tr>"
    return f'<table class="matrix grid7">{body}</table>'


# ---------------------------------------------------------------- page: header

runs = runs_table()

st.markdown("""
<div class="hero">
  <div class="eyebrow">Emergent communication · LLM agents</div>
  <h1>Alien First Contact</h1>
  <p>Two AI agents share no language. With nothing but random alien symbols and a guessing game,
  they invent one from scratch: vocabulary first, then grammar. Every round below is real model output.</p>
  <div class="float glyph">⟁ ⧖ ⍨ ⏇ ⌹ ⍟ ⏧ ⩛</div>
</div>
""", unsafe_allow_html=True)

if runs.empty:
    st.info("No real runs yet. Start one with `python main.py --level 2 --rounds 200`.")
    st.stop()

st.markdown(kpis([
    ("Experiments", f"{len(runs)}", f"{int(runs.running.sum())} running now", False),
    ("Rounds played", f"{int(runs.rounds_played.sum()):,}", "each logged with both agents' reasoning", False),
    ("Best run", f"{runs.success.max():.0f}%", "success · 25% = random guessing", True),
    ("API cost", f"${runs.cost.sum():.2f}", "all finished experiments", False),
]), unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🛸 Experiments")
    labels = {}
    for _, r in runs.iterrows():
        kind = "full feedback" if r.feedback == "full" else r.experiment
        live = " ●" if r.running else ""
        labels[int(r.run_id)] = f"#{r.run_id} · L{r.level} · {kind} · {r.success:.0f}%{live}"
    ids = list(labels)
    wanted = st.query_params.get("run")  # link to one experiment: ?run=5
    start = ids.index(int(wanted)) if wanted and wanted.isdigit() and int(wanted) in ids else 0
    run_id = st.selectbox("Experiment", ids, index=start, format_func=labels.get,
                          label_visibility="collapsed")
    st.query_params["run"] = str(run_id)
    if st.button("↻  Refresh data", width="stretch"):
        st.cache_data.clear()
    st.markdown("""<div class="caption"><b>Level 1</b> · 6 colors<br><b>Level 2</b> · 30 color + shape
    objects<br><b>Level 3</b> · 48 cells on a 7×7 grid<br><br>8 random symbols per run · at most 3 per
    message · 4 candidates per round</div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------- page: one experiment

run, rows, detail, notes = run_details(run_id)
alphabet = json.loads(run["alphabet"])
cost = estimate_cost(run["model"], run["input_tokens"], run["output_tokens"])
is_live = bool(runs.loc[runs.run_id == run_id, "running"].iloc[0])
train = [r for r in rows if r.phase == "train"]
speakers = sorted({r.speaker_id for r in train})
lexicons = {s: lexicon_of([r for r in train if r.speaker_id == s], run["level"]) for s in speakers}
scores = {s: structure(run_id, s, run["level"],
                       tuple(sorted((t.name, tuple(m)) for t, m in lex.items())))
          for s, lex in lexicons.items()}
score = next((v for v in scores.values() if v is not None), None)

tags = [f"Level {run['level']} · {LEVEL_LABEL.get(run['level'])}", run["experiment"],
        f"feedback: {run['feedback']}", run["model"]]
tag_html = "".join(f'<span class="tag">{esc(str(t))}</span>' for t in tags)
live_html = '<span class="tag live">● running</span>' if is_live else ""
st.markdown(f'<div class="runhead"><h2>Experiment #{run_id}</h2>{tag_html}{live_html}</div>'
            f'<div class="alphabet">This run\'s alphabet &nbsp; {chips(alphabet)}</div>',
            unsafe_allow_html=True)

last = rows[-20:]
st.markdown(kpis([
    ("Rounds", f"{len(rows)}", f"{len({r.phase for r in rows})} phase(s)", False),
    ("Success", f"{100 * sum(r.success for r in rows) / len(rows):.0f}%",
     f"guessing = {100 / run['n_candidates']:.0f}%", False),
    ("Last 20 rounds", f"{100 * sum(r.success for r in last) / len(last):.0f}%", "where they ended up", True),
    ("Structure", f"{score.score:.2f}" if score else "—",
     "topographic similarity" if score else "needs level 2 or 3", False),
    ("Cost", f"${cost:.2f}" if cost is not None and not is_live else "—",
     "estimated" if not is_live else "shown when finished", False),
]), unsafe_allow_html=True)

tab_lang, tab_curve, tab_replay, tab_notes, tab_all = st.tabs(
    ["🗺️  Language", "📈  Learning curve", "💬  Conversation", "📓  Private notes", "🧪  All experiments"])

with tab_lang:
    for s in speakers:
        if len(speakers) > 1:
            sc = scores[s]
            st.markdown(f"#### Tribe {s}" + (f" · structure {sc.score:.2f}" if sc else ""))
        st.markdown(language_map(run, lexicons[s], rows), unsafe_allow_html=True)
    msg = "Each cell shows the message the speaker used most for that meaning in the second half of training."
    if run["held_out"]:
        msg += (" Dashed cells were hidden during training; they show the message the speaker invented "
                "the first time it ever saw that object.")
    if score:
        msg += (f" Structure score {score.score:.2f}, against {abs(score.shuffled_mean):.2f} for the same "
                f"messages shuffled (p = {score.p_value:.3f}).")
    st.markdown(f'<div class="caption">{msg}</div>', unsafe_allow_html=True)

with tab_curve:
    st.altair_chart(success_chart(rows, run["n_candidates"]), width="stretch")
    phases = detail.groupby("phase", sort=False).success.agg(["mean", "count"])
    if len(phases) > 1:
        st.markdown(kpis([(f"Phase · {p}", f"{100 * m:.0f}%", f"{int(n)} rounds", i == len(phases) - 1)
                          for i, (p, (m, n)) in enumerate(phases.iterrows())]),
                    unsafe_allow_html=True)
    st.markdown('<div class="caption">Rolling success over the last 10 rounds. The dashed line is '
                'random guessing; the average restarts at each phase change.</div>',
                unsafe_allow_html=True)

with tab_replay:
    by_name = {t.name: t for t in make_objects(run["level"])}
    n = st.slider("Round", 1, len(detail), len(detail), key=f"round_{run_id}")
    r = detail.iloc[n - 1]
    target = by_name[r.target]
    candidates = [by_name[c] for c in json.loads(r.candidates)]
    message = json.loads(r.message)
    st.markdown(f'<div class="caption" style="margin-bottom:.8rem">Round <b>{r.round_no}</b> · '
                f'phase {esc(r.phase)} · speaker {esc(r.speaker_id)} → listener {esc(r.listener_id)}'
                f'</div>', unsafe_allow_html=True)
    left, mid, right = st.columns([5, 2, 5])
    with left:
        seen = (mini_grid({(target.dx, target.dy): ("T", "t")}) if isinstance(target, Cell)
                else object_row(target, classes="target"))
        st.markdown(f'<div class="panel"><div class="who">Speaker sees</div>{seen}'
                    f'<div class="reason">{esc(r.speaker_reasoning)}</div></div>',
                    unsafe_allow_html=True)
    with mid:
        st.markdown(f'<div class="wire"><div class="lbl">Message</div><div>{chips(message, big=True)}'
                    f'</div><div class="arrow">→</div></div>', unsafe_allow_html=True)
    with right:
        picked = r.picked
        if not r.valid:
            body = f'<div class="verdict bad">⛔ Rejected by the validator: {esc(r.error)}</div>'
        elif isinstance(target, Cell):
            marks = {}
            for i, c in enumerate(candidates, start=1):
                cls = "c" + (" tg" if c == target else "")
                if c.name == picked:
                    cls += " pk" + (" ok" if r.success else "")
                marks[(c.dx, c.dy)] = (str(i), cls)
            body = mini_grid(marks)
        else:
            body = ""
            for i, c in enumerate(candidates, start=1):
                if c.name == picked:
                    body += object_row(c, i, "picked " + ("ok" if r.success else "bad"), "picked")
                elif c == target:
                    body += object_row(c, i, "target", "target")
                else:
                    body += object_row(c, i)
        if r.valid:
            body += ('<div class="verdict ok">✓ Understood</div>' if r.success else
                     f'<div class="verdict bad">✗ Picked {esc(str(picked))}</div>')
        st.markdown(f'<div class="panel"><div class="who">Listener sees</div>{body}'
                    f'<div class="reason">{esc(r.listener_reasoning or "")}</div></div>',
                    unsafe_allow_html=True)

with tab_notes:
    if notes.empty:
        st.markdown('<div class="caption">This run predates the private-notes feature.</div>',
                    unsafe_allow_html=True)
    else:
        a, b = st.columns([1, 2])
        agent = a.selectbox("Agent", sorted(notes.agent.unique()))
        mine = notes[notes.agent == agent]
        at = b.select_slider("Written after round", options=list(mine.round_no))
        st.markdown(f'<div class="notecard">{esc(mine[mine.round_no == at].notes.iloc[0])}</div>'
                    f'<div class="caption">Written by the agent, for itself. The other agent never '
                    f'sees these.</div>', unsafe_allow_html=True)

with tab_all:
    table = runs.assign(
        level=lambda d: "L" + d.level.astype(str) + " " + d.level.map(LEVEL_LABEL),
        status=lambda d: d.running.map({True: "● running", False: "done"}),
    )[["run_id", "level", "experiment", "feedback", "rounds_played", "success", "cost", "status"]]
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={
            "run_id": st.column_config.NumberColumn("#", format="%d"),
            "rounds_played": st.column_config.NumberColumn("rounds"),
            "success": st.column_config.ProgressColumn("success", min_value=0, max_value=100,
                                                       format="%.0f%%"),
            "cost": st.column_config.NumberColumn("cost", format="$%.2f"),
        })
