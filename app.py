from __future__ import annotations
import json
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import streamlit as st

# Qiskit robust imports
try:
    from qiskit_aer import AerSimulator
except Exception:
    from qiskit.providers.aer import AerSimulator  # type: ignore

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_vector

import matplotlib.pyplot as plt
import plotly.express as px

# ---------------- helpers & small improvements ----------------
def ascii_circuit(qc: QuantumCircuit) -> str:
    try:
        td = qc.draw(output="text")
        return td.single_string() if hasattr(td, "single_string") else str(td)
    except Exception:
        try:
            return str(qc.draw(output="text"))
        except Exception:
            return "(circuit render not available)"

# Custom exceptions for clearer errors
class CircuitError(Exception):
    """Base exception for circuit-related errors."""

class GateError(CircuitError):
    """Raised when an invalid gate or target configuration is used."""

@dataclass
class GateOp:
    gate: str
    targets: List[int]
    theta: Optional[float] = None

    def __post_init__(self) -> None:
        # basic validations to fail fast when adding invalid operations
        if not isinstance(self.targets, list):
            raise GateError("targets must be a list of integers")
        if any(not isinstance(t, (int, np.integer)) for t in self.targets):
            raise GateError("each target must be an integer")
        g = (self.gate or "").upper()
        if g in {"RX", "RY", "RZ"} and self.theta is None:
            raise GateError(f"{g} requires a theta parameter")
        if g in {"H", "X", "Y", "Z", "RX", "RY", "RZ"} and len(self.targets) != 1:
            raise GateError(f"{g} requires exactly one target qubit")
        if g in {"CX", "CZ"} and len(self.targets) != 2:
            raise GateError(f"{g} requires two qubit indices: control, target")

    def to_row(self) -> Dict[str, Any]:
        return {
            "gate": self.gate,
            "targets": ",".join(map(str, self.targets)),
            "theta(rad)": None if self.theta is None else round(float(self.theta), 5),
        }

SINGLE_QUBIT_GATES = ["H", "X", "Y", "Z", "RX", "RY", "RZ"]
TWO_QUBIT_GATES = ["CX", "CZ"]
MEASURE_GATE = "Measure"

def build_circuit(n_qubits: int, ops: List[GateOp]) -> QuantumCircuit:
    if not isinstance(n_qubits, int) or n_qubits <= 0:
        raise ValueError("n_qubits must be a positive integer")
    qc = QuantumCircuit(n_qubits, n_qubits)
    for op in ops:
        g = op.gate.upper()
        # validate target indices are in range
        if any(t < 0 or t >= n_qubits for t in op.targets):
            raise GateError(f"Target index out of range for gate {op.gate}: {op.targets}")
        if g in {"H", "X", "Y", "Z", "RX", "RY", "RZ"}:
            t = int(op.targets[0])
            if g == "H": qc.h(t)
            elif g == "X": qc.x(t)
            elif g == "Y": qc.y(t)
            elif g == "Z": qc.z(t)
            elif g == "RX": qc.rx(op.theta or 0.0, t)
            elif g == "RY": qc.ry(op.theta or 0.0, t)
            elif g == "RZ": qc.rz(op.theta or 0.0, t)
        elif g in {"CX", "CZ"}:
            c, t = map(int, op.targets)
            if c == t:
                raise GateError(f"Control and target must differ for {g}")
            if g == "CX": qc.cx(c, t)
            else: qc.cz(c, t)
        elif g == "MEASURE":
            qc.barrier()
            qc.measure(range(n_qubits), range(n_qubits))
        else:
            raise GateError(f"Unsupported gate: {op.gate}")
    return qc

# Cache the simulator instance (lighter-weight than caching results)
@st.cache_resource
def get_simulator() -> AerSimulator:
    return AerSimulator()

def _merge_counts_list(counts_list: List[Dict[str, int]]) -> Dict[str, int]:
    merged: Dict[str, int] = {}
    for c in counts_list:
        for k, v in c.items():
            merged[k] = merged.get(k, 0) + v
    return merged

def simulate_counts(qc: QuantumCircuit, shots: int = 1024) -> Dict[str, int]:
    sim = get_simulator()
    try:
        tqc = transpile(qc, sim)
    except Exception:
        tqc = qc
    try:
        result = sim.run(tqc, shots=shots).result()
    except Exception:
        # simulator failed: return empty counts instead of raising in UI
        return {}
    try:
        counts = result.get_counts()
    except Exception:
        counts = {}
    if isinstance(counts, list):
        counts = _merge_counts_list(counts)
    return counts or {}

def simulate_statevector(qc: QuantumCircuit) -> Optional[np.ndarray]:
    try:
        no_meas = QuantumCircuit(qc.num_qubits)
        for instr, qargs, cargs in qc.data:
            # skip classical measure instructions
            if instr.name.lower() == "measure":
                continue
            no_meas.append(instr, qargs, cargs)
        sv = Statevector.from_instruction(no_meas)
        return np.array(sv.data, dtype=complex)
    except Exception:
        return None

def bloch_vector_from_statevector(sv: np.ndarray) -> Optional[Tuple[float, float, float]]:
    if sv is None or sv.shape != (2,):
        return None
    a, b = sv
    x = 2 * np.real(np.conjugate(a) * b)
    y = 2 * np.imag(np.conjugate(b) * a)
    z = np.abs(a) ** 2 - np.abs(b) ** 2
    return float(x), float(y), float(z)

# -------------- load content --------------
@st.cache_data
def load_content(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

CONTENT = load_content("learn/content_en.json")

# -------------- page setup --------------
st.set_page_config(page_title="Quantum Learn & Practice", layout="wide")

if "ops" not in st.session_state:
    st.session_state.ops = []
if "scores" not in st.session_state:
    st.session_state.scores = {}

st.title("🧠 Quantum Learn & Practice")
st.caption("Learn quantum computing concepts and practice by building circuits.")

with st.sidebar:
    st.header("⚙️ Settings")
    n_qubits = st.number_input("Number of qubits", min_value=1, max_value=5, value=1, step=1)
    shots = st.slider("Shots", min_value=100, max_value=8192, value=1024, step=100)
    module_titles = [m["title"] for m in CONTENT["modules"]]
    module_idx = st.selectbox("Module", options=list(range(len(module_titles))),
                              format_func=lambda i: f"{i+1}. {module_titles[i]}",
                              index=0)
    st.divider()
    st.subheader("Quick actions")
    if st.button("Clear ops"):
        st.session_state.ops = []

learn_tab, practice_tab, challenge_tab, sandbox_tab, progress_tab = st.tabs([
    "📘 Learn", "🧪 Practice", "🎯 Challenge", "🧰 Sandbox", "🏅 Progress"
])

# ---------------- Learn ----------------
with learn_tab:
    mod = CONTENT["modules"][module_idx]
    st.subheader(f"Module {module_idx+1} — {mod['title']}")

    for block in mod["learn"]:
        btype = block.get("type")

        if btype == "md":
            st.markdown(block["text"])

        elif btype == "demo_statevector":
            st.info(block.get("text", "Demo"))
            demo_ops = [GateOp(gate="H", targets=[0])]
            demo_qc = build_circuit(1, demo_ops)
            sv = simulate_statevector(demo_qc)
            if sv is not None:
                pretty = [f"{amp.real:+.4f}{amp.imag:+.4f}i" for amp in sv]
                probs = [float(abs(a)**2) for a in sv]
                st.write("**Statevector (|0⟩→H→|+⟩):**")
                st.code("[" + ", ".join(pretty) + "]")
                st.write(f"Probabilities: p(0)={probs[0]:.3f}, p(1)={probs[1]:.3f}")
                st.caption("Global phase does not change these probabilities.")
            else:
                st.warning("Statevector not available.")

        elif btype == "task":
            st.info(block["text"])
            col1, col2 = st.columns([2, 3], vertical_alignment="top")
            with col1:
                if st.button("Run task", key=block.get("id", "learn_task")):
                    ops = [GateOp(gate=step["gate"], targets=step.get("targets", [])) for step in block["action"]["preset_circuit"]]
                    qc = build_circuit(block["action"]["n_qubits"], ops)
                    counts = simulate_counts(qc, shots=block["action"]["shots"])
                    st.session_state["learn_task_result_" + block.get("id", "t")] = {
                        "counts": counts,
                        "qc_text": ascii_circuit(qc),
                    }
            with col2:
                res = st.session_state.get("learn_task_result_" + block.get("id", "t"))
                if res:
                    st.code(res["qc_text"])
                    counts = res["counts"]
                    if counts:
                        fig = px.bar(x=list(counts.keys()), y=list(counts.values()), labels={"x": "bitstring", "y": "count"}, title="Measurement histogram")
                        st.plotly_chart(fig, use_container_width=True)
                    st.caption("After H on |0⟩, expect ~50/50 distribution.")

        elif btype == "quiz_mcq":
            st.markdown(f"**Question:** {block['question']}")
            key = f"quiz_{block['id']}"
            choice = st.radio("Options", block["options"], index=None, key=key, horizontal=False)
            if choice is not None:
                correct_idx = block["answer_index"]
                if block["options"].index(choice) == correct_idx:
                    st.success("Correct ✔ " + block.get("explain_correct", ""))
                else:
                    st.error("Not quite. " + block.get("explain_wrong", ""))

        elif btype == "bloch_demo":
            st.info(block.get("text", "Bloch sphere demo"))
            demo_ops = [GateOp(gate="H", targets=[0])] if block.get("id") == "m1_bloch" else []
            # for module 2 RY sweep, we'll compute at a slider
            if block.get("id") == "m2_bloch":
                theta = st.slider("θ (radians)", min_value=0.0, max_value=float(np.pi), value=float(np.pi/2), step=0.01, key="m2_bloch_theta")
                demo_ops = [GateOp(gate="RY", targets=[0], theta=theta)]
            demo_qc = build_circuit(1, demo_ops)
            sv = simulate_statevector(demo_qc)
            vec = bloch_vector_from_statevector(sv) if sv is not None else None
            if vec is None:
                st.warning("Could not compute Bloch vector.")
            else:
                x, y, z = vec
                st.write(f"Bloch vector (x, y, z) = ({x:.3f}, {y:.3f}, {z:.3f})")
                fig_bloch = plot_bloch_vector([x, y, z])
                st.pyplot(fig_bloch.figure)

        elif btype == "task_param":
            st.info(block["text"])
            act = block["action"]
            n_qub = act["n_qubits"]
            tq = act["target_qubit"]
            shots_local = act["shots"]
            target_p0 = act["target_p0"]
            tol = act["tolerance"]

            theta = st.slider("θ (radians)", min_value=0.0, max_value=float(np.pi), value=float(np.pi/3), step=0.01, key=f"theta_{block['id']}")
            ops = [GateOp(gate="RY", targets=[tq], theta=theta), GateOp(gate="Measure", targets=[])]
            qc = build_circuit(n_qub, ops)

            col_l, col_r = st.columns([2, 3])
            with col_l:
                st.code(ascii_circuit(qc))
                if st.button("Run", key=f"run_{block['id']}"):
                    counts = simulate_counts(qc, shots=shots_local)
                    st.session_state[f"task_param_res_{block['id']}"] = counts
            with col_r:
                counts = st.session_state.get(f"task_param_res_{block['id']}")
                if counts:
                    total = sum(counts.values())
                    p0 = counts.get("0", 0) / total if total else 0.0
                    fig = px.bar(x=list(counts.keys()), y=list(counts.values()), labels={"x": "bitstring", "y": "count"}, title="Measurement histogram")
                    st.plotly_chart(fig, use_container_width=True)
                    st.write(f"p(0) = {p0:.3f}  |  target: {target_p0:.2f} ± {tol:.2f}")
                    if abs(p0 - target_p0) <= tol:
                        st.success("Great! You hit the target range. 🎯")
                    else:
                        st.info("Hint: For a single qubit, RY(θ) on |0⟩ gives p(0)=cos²(θ/2).")

        elif btype == "demo_ops":
            st.info(block.get("text", "Demo"))
            act = block.get("action", {})
            n_qub = act.get("n_qubits", 1)
            ops_spec = act.get("ops", [])
            ops = [GateOp(gate=step["gate"], targets=step.get("targets", []), theta=step.get("theta"))
                   for step in ops_spec]
            demo_qc = build_circuit(n_qub, ops)
            sv = simulate_statevector(demo_qc)
            st.code(ascii_circuit(demo_qc))
            if sv is not None:
                pretty = [f"{amp.real:+.4f}{amp.imag:+.4f}i" for amp in sv]
                probs = [float(abs(a)**2) for a in sv]
                st.write("**Statevector:**")
                st.code("[" + ", ".join(pretty) + "]")
                st.write("Probabilities: " + ", ".join([f"p({i})={p:.3f}" for i, p in enumerate(probs)]))
            else:
                st.warning("Statevector not available.")

# --------------- Practice ---------------
with practice_tab:
    st.subheader("Circuit Builder")
    c1, c2 = st.columns([3, 2], vertical_alignment="top")

    with c1:
        with st.form("add_gate_form", clear_on_submit=True):
            gate = st.selectbox("Gate", SINGLE_QUBIT_GATES + TWO_QUBIT_GATES + [MEASURE_GATE])
            theta = None
            targets: List[int] = []

            if gate in SINGLE_QUBIT_GATES:
                t = st.number_input("Target qubit", 0, n_qubits - 1, value=0)
                targets = [int(t)]
                if gate in ("RX", "RY", "RZ"):
                    theta = st.slider("θ (radians)", min_value=-np.pi, max_value=np.pi, value=float(np.pi/2), step=0.01)
            elif gate in TWO_QUBIT_GATES:
                c = st.number_input("Control", 0, n_qubits - 1, value=0)
                t = st.number_input("Target", 0, n_qubits - 1, value=min(1, n_qubits - 1))
                targets = [int(c), int(t)]
                if targets[0] == targets[1]:
                    st.warning("Control and target must be different.")
            else:  # Measure
                targets = []

            submitted = st.form_submit_button("Add gate")
            if submitted:
                try:
                    st.session_state.ops.append(GateOp(gate=gate, targets=targets, theta=theta))
                    st.success(f"Added: {gate} {targets} {'' if theta is None else f'theta={theta:.3f}'}")
                except Exception as e:
                    st.error(str(e))

        if st.session_state.ops:
            rows = [op.to_row() for op in st.session_state.ops]
            st.dataframe(rows, use_container_width=True)
            col_rm1, col_rm2 = st.columns([1, 4])
            with col_rm1:
                if st.button("Remove last"):
                    st.session_state.ops.pop()
            with col_rm2:
                if st.button("Clear all"):
                    st.session_state.ops = []
        else:
            st.caption("No gates added yet.")

    with c2:
        qc = build_circuit(n_qubits, st.session_state.ops)
        st.code(ascii_circuit(qc))

        if st.button("Simulate (with measurement)"):
            counts = simulate_counts(qc, shots=shots)
            if counts:
                fig = px.bar(x=list(counts.keys()), y=list(counts.values()), labels={"x": "bitstring", "y": "count"}, title="Measurement histogram")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No counts. Did you include a Measure gate?")

        if st.button("Statevector (pre-measurement)"):
            sv = simulate_statevector(qc)
            if sv is None:
                st.warning("Statevector unavailable.")
            else:
                pretty = [f"{amp.real:+.4f}{amp.imag:+.4f}i" for amp in sv]
                st.write("**Statevector:**")
                st.code("[" + ", ".join(pretty) + "]")

# --------------- Challenge ---------------
with challenge_tab:
    ch = CONTENT["modules"][module_idx]["challenge"]
    st.subheader("Challenge")
    st.write(ch["description"])
    st.caption(f"Constraint: max_gates = {ch['constraints']['max_gates']}, success fidelity ≥ {ch['success_fidelity']}")

    ops = st.session_state.ops
    if len([o for o in ops if o.gate.upper() != "MEASURE"]) > ch["constraints"]["max_gates"]:
        st.warning("Constraint exceeded: reduce the number of gates (excluding Measure).")

    qc_user = build_circuit(ch["n_qubits"], ops)
    st.code(ascii_circuit(qc_user))

    def target_state_vector(name: str, n_qubits: int) -> Optional[np.ndarray]:
        # Single-qubit named targets
        if n_qubits == 1:
            if name == "plus":
                return np.array([1/np.sqrt(2),  1/np.sqrt(2)], dtype=complex)
            if name == "minus":
                return np.array([1/np.sqrt(2), -1/np.sqrt(2)], dtype=complex)
            if name == "zero":
                return np.array([1.0, 0.0], dtype=complex)
            if name == "one":
                return np.array([0.0, 1.0], dtype=complex)
            # special pseudo-target used in Module 3 for sampling-based check
            if name.startswith("p0_"):
                return None

        # Two-qubit Bell states (|00>, |01>, |10>, |11> order)
        if n_qubits == 2:
            if name == "phi_plus":   # (|00> + |11>)/sqrt(2)
                return (1/np.sqrt(2)) * np.array([1, 0, 0, 1], dtype=complex)
            if name == "phi_minus":  # (|00> - |11>)/sqrt(2)
                return (1/np.sqrt(2)) * np.array([1, 0, 0, -1], dtype=complex)
            if name == "psi_plus":   # (|01> + |10>)/sqrt(2)
                return (1/np.sqrt(2)) * np.array([0, 1, 1, 0], dtype=complex)
            if name == "psi_minus":  # (|01> - |10>)/sqrt(2)
                return (1/np.sqrt(2)) * np.array([0, 1, -1, 0], dtype=complex)

        return None

    def fidelity_to_target(qc: QuantumCircuit, target: np.ndarray) -> Optional[float]:
        sv = simulate_statevector(qc)
        if sv is None or sv.shape != target.shape:
            return None
        overlap = np.vdot(target, sv)
        return float(np.abs(overlap) ** 2)

    if st.button("Evaluate"):
        # Special sampling-based check for Module 3 targets like p0_0.25
        if ch["target_state"].startswith("p0_") and ch["n_qubits"] == 1:
            try:
                target_val = float(ch["target_state"].split("_")[1])
            except Exception:
                target_val = 0.25
            counts_eval = simulate_counts(qc_user, shots=2048)
            total = sum(counts_eval.values()) or 1
            p0 = counts_eval.get("0", 0) / total
            st.write(f"Estimated p(0) from sampling: {p0:.3f} (target ≈ {target_val:.2f})")
            if abs(p0 - target_val) <= 0.05:
                st.success("Success by sampling criterion! 🎯")
                st.session_state.scores[ch["id"]] = max(st.session_state.scores.get(ch["id"], 0.0), 1.0)
            else:
                st.info("Try adjusting your circuit (hint: RY controls p(0) = cos²(θ/2)).")
            st.stop()

        tsv = target_state_vector(ch["target_state"], ch["n_qubits"])
        if tsv is None:
            st.error("Undefined target state.")
        else:
            fid = fidelity_to_target(qc_user, tsv)
            if fid is None:
                st.error("Could not compute fidelity. Ensure you evaluate the pre-measurement state.")
            else:
                st.metric("Fidelity", f"{fid:.4f}")
                if fid >= ch["success_fidelity"]:
                    st.success("Success! 🎉")
                    st.session_state.scores[ch["id"]] = max(st.session_state.scores.get(ch["id"], 0.0), fid)
                else:
                    st.info("Keep trying. Hint: adjust gates to match the target state.")

# --------------- Sandbox ---------------
with sandbox_tab:
    st.subheader("Free Playground")
    st.caption("Use the circuit from Practice and experiment freely.")
    qc = build_circuit(n_qubits, st.session_state.ops)
    st.code(ascii_circuit(qc))
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Sandbox: simulate measurement"):
            counts = simulate_counts(qc, shots=shots)
            if counts:
                fig = px.bar(x=list(counts.keys()), y=list(counts.values()), labels={"x": "bitstring", "y": "count"}, title="Measurement histogram")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No counts. Add a Measure gate.")
    with col_b:
        if st.button("Sandbox: statevector"):
            sv = simulate_statevector(qc)
            if sv is None:
                st.warning("Statevector unavailable.")
            else:
                pretty = [f"{amp.real:+.4f}{amp.imag:+.4f}i" for amp in sv]
                st.code("[" + ", ".join(pretty) + "]")

# --------------- Progress ---------------
with progress_tab:
    st.subheader("Progress & Badges (MVP)")
    scores = st.session_state.get("scores", {})
    if not scores:
        st.caption("No records yet. Try the Challenge tab!")
    else:
        for k, v in scores.items():
            st.write(f"**{k}** → best fidelity: {v:.4f}")
    st.info("Note: in this MVP, progress is in-memory only. We can persist to SQLite next.")
