"""Build deliverables/literature_matrix.xlsx from the verified paper records and this study's results.

Inputs:  deliverables/data/litmatrix.jsonl (one verified record per paper),
         results/t4/report_numbers.json (paired success rates, as in the report),
         results/t4/profile/graph_t4.json (latency).
Usage:   python scripts/build_literature_matrix.py   (then recalculate formulas, e.g. LibreOffice)
"""

import json
import re
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables" / "literature_matrix.xlsx"

TARGETS = {
    "SmolVLA": "Model size, memory, latency (async execution)",
    "pi0": "Latency (prefix KV reuse, chunking)",
    "OpenVLA": "Memory (quantisation), fine-tuning cost (LoRA)",
    "OpenVLA-OFT": "Latency / throughput",
    "FAST": "Training cost (fewer action tokens)",
    "TinyVLA": "Model size, latency, training data",
    "VLA-Cache": "Compute, latency",
    "DeeR-VLA": "Compute, memory",
    "EfficientVLA": "Compute, latency, memory",
    "EfficientVLA-Survey": "Model, training and data efficiency (taxonomy)",
    "DiffusionPolicy": "N/A (generative action-head reference)",
    "CogACT": "Compute (VLM runs once, small DiT denoises)",
    "A2C2": "Reactivity of chunked/asynchronous execution",
    "RTC": "Latency hiding (asynchronous execution)",
    "RealtimeVLA": "Latency (CPU launch overhead, kernel fusion)",
    "Islam-ONNX": "Latency (runtime/export), closed-loop side effects",
    "Coda": "Latency (fewer flow steps + learned correction)",
    "vla.cpp": "Latency, memory, edge deployment (runtime)",
}

F = "Arial"
HDR_FONT = Font(name=F, bold=True, color="FFFFFF", size=10)
HDR_FILL = PatternFill("solid", fgColor="1F4E79")
BODY = Font(name=F, size=9)
BOLD = Font(name=F, size=9, bold=True)
WRAP = Alignment(wrap_text=True, vertical="top")
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
ALT = PatternFill("solid", fgColor="F2F6FA")


DISPLAY = {"pi0": "π0", "DiffusionPolicy": "Diffusion Policy", "RealtimeVLA": "Realtime-VLA",
           "Islam-ONNX": "Islam (ONNX)"}


def pretty(v):
    """ASCII notation from the reading notes -> readable symbols (cell text only)."""
    if not isinstance(v, str):
        return v
    v = v.replace("->", "→").replace("+-", "±")
    for a, b in ((r"\beps_theta\b", "ε_θ"), (r"\bv_theta\b", "v_θ"), (r"\beps_k\b", "ε_k"), (r"\beps\b", "ε"),
                 (r"\btau\b", "τ"), (r"\btheta\b", "θ"), (r"A\^tau", "A^τ"), (r"\bpi0\.5\b", "π0.5"),
                 (r"\bpi0\b", "π0"), (r"\bdelta\b", "δ")):
        v = re.sub(a, b, v)
    return v


def header(ws, row, headers):
    for j, h in enumerate(headers, 1):
        c = ws.cell(row=row, column=j, value=h)
        c.font, c.fill, c.border = HDR_FONT, HDR_FILL, BORDER
        c.alignment = Alignment(wrap_text=True, vertical="center")


def table(ws, headers, rows, widths, start_row=1):
    header(ws, start_row, headers)
    for i, r in enumerate(rows, start_row + 1):
        for j, v in enumerate(r, 1):
            c = ws.cell(row=i, column=j, value=pretty(v))
            c.font = BOLD if j == 1 else BODY
            c.alignment, c.border = WRAP, BORDER
            if (i - start_row) % 2 == 0:
                c.fill = ALT
    for j, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(j)].width = w
    ws.freeze_panes = ws.cell(row=start_row + 1, column=2)


# The ten papers kept in the matrix: most cited and most directly tied to this study, ordered from the
# model studied, through the action heads, to the efficiency levers by the part of the loop they change.
CORE = ["SmolVLA", "pi0", "DiffusionPolicy", "OpenVLA-OFT", "VLA-Cache", "EfficientVLA", "RTC", "RealtimeVLA",
        "Coda", "Islam-ONNX"]
RELEVANCE = {
    "SmolVLA": "The model studied: its LIBERO checkpoint is reproduced and optimised here.",
    "pi0": "Same design as SmolVLA (VLM prefix + flow-matching action expert, 10 Euler steps on a cached prefix).",
    "DiffusionPolicy": "Reference diffusion action head for the diffusion vs flow-matching comparison.",
    "OpenVLA-OFT": "Reports speed and success together; chunked parallel decoding is the 'execute more per call' idea.",
    "VLA-Cache": "Reuses visual tokens across frames: the idea behind the stale-image variant (H1).",
    "EfficientVLA": "Training-free acceleration with a small success loss: the same setting as this study.",
    "RTC": "Asynchronous execution of chunked flow policies: the alternative to re-planning at every step (H1).",
    "RealtimeVLA": "CUDA graphs and kernel-level work for pi0: the graph lever of H2.",
    "Coda": "Fewer flow steps for pi0.5 (3 steps hold, 1 step drops): the step lever of H2 and its contrast.",
    "Islam-ONNX": "Faster deployment of this very checkpoint lowered success (70% to 41%): motivates paired tests.",
}


def sheet_matrix(wb, papers):
    ws = wb.active
    ws.title = "Matrix"
    headers = ["Paper", "Title", "Relevance to this study", "Efficiency lever (what is reduced and how)",
               "Reported efficiency gain (hardware)", "Benchmark & success rate", "Training-free",
               "Main target", "Action generation", "Model size & backbone", "Assumptions", "Trade-offs",
               "Open questions", "First author", "Year", "Venue (arXiv)", "Source (read)"]
    rows = []
    for p in papers:
        venue = p["venue"] if p["venue"] else "arXiv"
        tf = "yes" if p["training_free"].lower().startswith("yes") else "no"
        rows.append([DISPLAY.get(p["id"], p["id"]), p["title"], RELEVANCE[p["id"]], p["efficiency_lever"],
                     p["reported_gain"], p["benchmark_success"], tf, TARGETS.get(p["id"], ""),
                     p["action_generation"], p["params_and_backbone"], p["assumptions"], p["tradeoffs"],
                     p["open_questions"], p["first_author"], p["year"], f'{venue} ({p["arxiv_id"]})', p["url"]])
    table(ws, headers, rows, [14, 30, 40, 44, 44, 44, 10, 22, 34, 34, 36, 36, 40, 16, 6, 24, 30])
    ws.row_dimensions[1].height = 30
    ws.freeze_panes = "C2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{len(rows) + 1}"
    for c in ws["G"][1:]:
        c.alignment = Alignment(horizontal="center", vertical="top")


def sheet_diffusion_flow(wb):
    ws = wb.create_sheet("Diffusion vs Flow")
    h = ["Model", "Generative family", "Network predicts", "Training objective", "Time / noise-level sampling",
         "Inference sampler & steps", "What each step re-runs", "Chunk / executed actions", "Reported latency",
         "Source"]
    r = [
        ["Diffusion Policy", "Diffusion (DDPM)", "Noise eps", "MSE(eps, eps_theta(O, A^0 + eps_k, k))",
         "Uniform over 100 DDPM steps", "Sim: 100 DDPM steps; real: DDIM 10 steps (App. A.4 says 16)",
         "Full denoising network (~9-80M)", "Tp=16 predicted, Ta=8 executed", "0.1 s on RTX 3080 (DDIM 10)",
         "arxiv.org/abs/2303.04137"],
        ["CogACT", "Diffusion (DiT action module)", "Noise eps",
         "MSE on noise; 8 diffusion steps sampled per training sample",
         "Diffusion timestep added to cognition feature (sinusoidal)", "DDIM 10 steps, classifier-free guidance 1.5",
         "DiT only (13-308M); VLM runs once to produce the cognition feature",
         "16 (current + 15 future), adaptive ensemble",
         "Not reported (third party: 54 ms RTX 4090; 0.234 s A40)", "arxiv.org/abs/2411.19650"],
        ["pi0", "Flow matching (linear path)", "Velocity u = A - eps",
         "E||v_theta(A^tau, o) - (A - eps)||^2, A^tau = tau*A + (1-tau)*eps (tau=0 noise)",
         "Beta((s - tau)/s; 1.5, 1), s=0.999 (emphasises noisy times)", "Forward Euler, 10 steps (delta = 0.1)",
         "~300M action expert on cached VLM prefix KV", "H=50; re-plan every 16 (20 Hz) or 25 (50 Hz) actions",
         "73 ms on RTX 4090 (10 flow steps: 27 ms)", "arxiv.org/abs/2410.24164"],
        ["SmolVLA (released code)", "Flow matching (linear path)", "Velocity u = eps - A",
         "E||v_theta(A^tau, tau | o) - (eps - A)||^2, A^tau = tau*eps + (1-tau)*A (tau=1 noise)",
         "Beta(1.5, 1) rescaled to [0.001, 1]", "Forward Euler from tau=1 to 0, 10 steps",
         "Action expert (97M in this checkpoint) on cached prefix KV",
         "n=50; this checkpoint executes 1 action per re-plan",
         "This study, Tesla T4: 789 ms per re-plan (bf16, 10 steps)", "arxiv.org/abs/2506.01844; LeRobot v0.6.1"],
        ["RTC (on pi0.5)", "Flow matching + guided inpainting", "Velocity (base policy)",
         "None (inference-time only)", "n/a", "5 denoising steps with vector-Jacobian guidance",
         "Expert + guidance backward pass", "H=50, execution horizon 25",
         "76 -> 97 ms per chunk on RTX 4090 (guidance overhead)", "arxiv.org/abs/2506.07339"],
    ]
    table(ws, h, r, [20, 20, 16, 40, 30, 32, 34, 28, 30, 26])
    n2 = len(r) + 3
    ws.cell(row=n2, column=1, value="Key differences").font = Font(name=F, bold=True, size=10)
    diffs = [
        "1. Target: diffusion regresses the injected noise; flow matching regresses the velocity of a straight "
        "path between noise and data.",
        "2. Path: the DDPM forward process follows a variance-preserving schedule (curved trajectories under the "
        "probability-flow ODE); the linear flow path has constant velocity per sample, so coarse Euler steps stay "
        "close (this study: 3 steps deviate 0.32 vs a 1.37 sampling spread, LIBERO-Object).",
        "3. Sampling: DDPM needs many steps (DDIM ~10); flow models use ~10 Euler steps by default and degrade "
        "gracefully when fewer are used, down to a model-dependent limit (Coda: pi0.5 holds at 3 steps, drops "
        "at 1; SmolVLA drops at 1).",
        "4. Cost per step: in all four VLA-style models a step re-runs only a small head on a cached VLM output, "
        "so per-step cost is set by the head size and by kernel-launch overhead at batch 1.",
    ]
    for k, t in enumerate(diffs, n2 + 1):
        c = ws.cell(row=k, column=1, value=t)
        c.font, c.alignment = BODY, WRAP
        ws.merge_cells(start_row=k, start_column=1, end_row=k, end_column=10)
        ws.row_dimensions[k].height = 30


def sheet_this_study(wb):
    ws = wb.create_sheet("This study (T4)")
    ws["A1"] = ("SmolVLA (HuggingFaceVLA/smolvla_libero) on LIBERO-Spatial, Tesla T4 - latency of one re-plan "
                "(median, CUDA events, results/t4/profile/graph_t4.json)")
    ws["A1"].font = Font(name=F, bold=True, size=11)
    g = json.load(open(ROOT / "results/t4/profile/graph_t4.json"))["results"]
    cfgs = [("Released", "native", 10, False), ("Fewer steps", "native", 3, False),
            ("Released + graph", "native", 10, True), ("fp16", "fp16", 10, False),
            ("fp16 + graph", "fp16", 10, True), ("fp16, 3 steps", "fp16", 3, False),
            ("fp16, 3 steps + graph", "fp16", 3, True), ("fp16, 2 steps + graph", "fp16", 2, True),
            ("fp16, 1 step + graph", "fp16", 1, True)]
    start = 3
    header(ws, start, ["Configuration", "Precision", "Flow steps", "CUDA graph", "Latency per re-plan (ms)",
                       "Speed-up vs released"])
    for i, (lab, mode, s, gr) in enumerate(cfgs, start + 1):
        ms = g[mode]["per_steps"][str(s)]["graph_ms" if gr else "eager_ms"]["p50"]
        vals = [lab, "bf16" if mode == "native" else mode, s, "yes" if gr else "no", round(ms, 1)]
        for j, v in enumerate(vals, 1):
            c = ws.cell(row=i, column=j, value=v)
            c.font, c.border = (BOLD if j == 1 else BODY), BORDER
        ws.cell(row=i, column=5).font = Font(name=F, size=9, color="0000FF")
        ws.cell(row=i, column=5).number_format = "0.0"
        c = ws.cell(row=i, column=6, value=f"=$E${start + 1}/E{i}")
        c.number_format, c.font, c.border = '0.00"x"', BODY, BORDER
    note = start + len(cfgs) + 2
    ws.cell(row=note, column=1, value=(
        "Blue numbers are measured inputs (60 calls per configuration after warm-up, interleaved rounds); "
        "speed-ups are formulas against the released configuration.")).font = Font(name=F, size=9, italic=True)

    rn = json.load(open(ROOT / "results/t4/report_numbers.json"))
    t, pre = rn["table_states_0_19"], rn["pre_specified"]
    rows = [("Open loop, 5 actions per re-plan", "0-19", t["open_loop_k5"], True),
            ("Open loop, 10 actions per re-plan", "0-19", t["open_loop_k10"], True),
            ("Open loop 5 + gripper-event re-plan", "0-19", t["open_loop_k5_events"], True),
            ("Stale image (cached up to 5 steps), 3 steps", "0-19", t["stale_image"], True),
            ("bf16, 3 steps", "0-19", t["bf16_s3"], True),
            ("fp16, 3 steps", "0-19", t["fp16_s3"], True),
            ("fp16, 2 steps", "0-19", t["fp16_s2"], True),
            ("fp16, 1 step", "0-19", t["fp16_s1"], True),
            ("fp16, 3 steps, one fixed noise per episode (probe)", "0-19", t["fp16_s3_fixed_noise"], True),
            ("fp16, 3 steps (states 20-49 added after the interim look)", "20-49", pre["fp16_s3_states_20_49"], False),
            ("fp16, 3 steps, all 500 pairs", "0-49", pre["fp16_s3_states_0_49"], False),
            ("fp16, 1 step (confirmation fixed before the run)", "20-49", pre["fp16_s1_states_20_49"], False)]
    r0 = note + 2
    head = ["Configuration (paired with the released policy)", "Initial states", "Pairs", "Released SR (same pairs)",
            "Variant SR", "Difference (points)", "95% CI low (points)", "95% CI high (points)", "McNemar p",
            "Holm p (Table 4 family)"]
    header(ws, r0, head)
    fmts = [None, None, "0", "0.0%", "0.0%", "+0.0;-0.0;0.0", "+0.0;-0.0;0.0", "+0.0;-0.0;0.0", "0.000", "0.000"]
    for i, (lab, states, c, in_table) in enumerate(rows, r0 + 1):
        vals = [lab, states, c["n_pairs"], c["sr_a"], c["sr_b"], c["diff_b_minus_a"] * 100,
                c["ci95_newcombe"][0] * 100, c["ci95_newcombe"][1] * 100, c["mcnemar_p"],
                c.get("mcnemar_p_holm") if in_table else None]
        for j, (x, fm) in enumerate(zip(vals, fmts), 1):
            cell = ws.cell(row=i, column=j, value=x)
            cell.font, cell.border = (BOLD if j == 1 else BODY), BORDER
            if fm:
                cell.number_format = fm
    ws.cell(row=r0 + len(rows) + 2, column=1, value=(
        "Source: results/t4/report_numbers.json (scripts/report_numbers.py): Newcombe paired 95% CI, exact McNemar; "
        "Holm over the nine comparisons on states 0-19 (report Table 4); the tests on states 20-49 are unadjusted. "
        "Non-inferiority margin: -5 points, fixed before any run.")).font = Font(name=F, size=9, italic=True)
    for j, w in enumerate([50, 10, 7, 13, 11, 12, 12, 12, 11, 12], 1):
        ws.column_dimensions[get_column_letter(j)].width = w


def sheet_notes(wb):
    ws = wb.create_sheet("Notes")
    notes = [
        "How this matrix was built: every entry was read from the paper's own arXiv abstract, HTML or PDF "
        "(column 'Source (read)'). 'Not reported' marks values not found in the text that was read; "
        "'(interpretation)' marks open questions that are this study's reading, not the authors' statement.",
        "Venues and years follow the publication (e.g. NeurIPS 2025); the report's bibliography keeps the BibTeX "
        "exported from Google Scholar, which dates NeurIPS 2025 proceedings (volume 38) as 2026.",
        "Inconsistencies found in the sources: Diffusion Policy states 10 DDIM inference steps (Sec. 3.4) and 16 "
        "(App. A.4); VLA-Cache's 1.7x headline is larger than its per-model tables (OpenVLA 1.63x).",
        "Scope: the matrix keeps the ten papers that are most cited and most directly tied to this study (column "
        "'Relevance to this study'). Also read, and cited in the report where used, but left out to keep the matrix "
        "focused: OpenVLA, FAST, TinyVLA, DeeR-VLA, CogACT, A2C2, an efficient-VLA survey and vla.cpp. Of these, "
        "A2C2 matters most for H1: it trains its own SmolVLA from scratch on LIBERO, and its execution-horizon "
        "results (e=5: 0.83 > e=1: 0.77) do not transfer to the checkpoint used here, where executing 5 actions per "
        "re-plan lowers SR by 12.5 pp.",
        "Islam (2026) uses the same checkpoint and LeRobot version on a Turing GPU (RTX 2060): its PyTorch+AMP "
        "Spatial success (70.0%, 100 episodes) is in the range of this study's baseline (77.0%, 200 episodes); its "
        "ONNX speed-up came with a behaviour change (truncated instructions), which motivates checking closed-loop "
        "equivalence for every efficiency change.",
        "Coda (2026) and Realtime-VLA (2025) already apply fewer flow steps and CUDA graphs, respectively; this "
        "study does not claim these levers as new.",

    ]
    ws.column_dimensions["A"].width = 140
    for i, t in enumerate(notes, 1):
        c = ws.cell(row=i, column=1, value=t)
        c.font, c.alignment = BODY, WRAP
        ws.row_dimensions[i].height = 42


def main():
    by_id = {d["id"]: d for d in (json.loads(line) for line in open(ROOT / "deliverables/data/litmatrix.jsonl"))}
    papers = [by_id[i] for i in CORE]
    wb = Workbook()
    sheet_matrix(wb, papers)
    sheet_diffusion_flow(wb)
    sheet_this_study(wb)
    sheet_notes(wb)
    for ws in wb.worksheets:
        ws.page_setup.orientation = "landscape"
        ws.page_setup.fitToWidth, ws.page_setup.fitToHeight = 1, 0
        ws.sheet_properties.pageSetUpPr.fitToPage = True
    wb.save(OUT)
    print(OUT, len(papers), "papers")


if __name__ == "__main__":
    main()
