# Quantum Learn & Practice (EN)

Interactive learning app for quantum computing + circuit practice with Qiskit & Streamlit.

## Features
- **Learn**: Rich module content (Module 1: Qubit Basics, Module 2: Single-Qubit Gates).
- **Practice**: Circuit Builder with single/two-qubit gates and measurement.
- **Challenge**: Target-state challenges with fidelity scoring.
- **Sandbox**: Free playground.
- **Progress**: In-memory best-fidelity tracking (MVP).

## Quickstart
```bash
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL shown (typically http://localhost:8501).

## Project Structure
```
.
├── app.py
├── requirements.txt
├── README.md
├── learn/
│   └── content_en.json
├── logic/
│   └── (helpers embedded in app.py for MVP)
├── components/
│   └── (reserved for future UI components)
└── .github/workflows/
    └── streamlit-check.yml (CI: import & syntax check)
```

## License
MIT
