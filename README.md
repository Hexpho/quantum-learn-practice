# 🧠 Quantum Learn & Practice

An **interactive learning app** built with **Streamlit** and **Qiskit**, designed to teach and let you practice **quantum computing** concepts hands-on.  
It combines **visual explanations**, **live circuit simulations**, and **fidelity-based challenges** — from basic qubits to entanglement.

---

## 🚀 Features

### 📘 Learn
- Interactive lessons for each topic  
- Markdown explanations with mathematical context  
- Live **statevector** and **Bloch sphere** visualizations  
- Built-in **multiple-choice quizzes** and **guided exercises**

### 🧪 Practice
- A **Quantum Circuit Builder** — add, reorder, and remove gates visually  
- Simulate with **Qiskit AerSimulator**  
- View **measurement histograms**, **statevectors**, and **Bloch coordinates**

### 🎯 Challenge Mode
- Module-specific targets to reach (e.g., prepare |+⟩ or |Φ⁺⟩)
- Automatic **fidelity evaluation** to measure your success
- Progress stored in-memory (extendable to database)

### 🧰 Sandbox
- Free experimentation space for custom quantum circuits

---

## 🧩 Learning Modules

| Module | Title | Concepts Covered | Practice Focus |
|:--|:--|:--|:--|
| **1** | Qubit Basics | Superposition, Dirac notation, Born rule, Bloch sphere | Create |+⟩ and interpret statevectors |
| **2** | Single-Qubit Gates | X/Y/Z, RX/RY/RZ rotations | Control rotation angles to achieve target probabilities |
| **3** | Measurement & Statistics | Sampling, shots, noise, estimation variance | Observe histogram stabilization and tune p(0) ≈ 0.25 |
| **4** | Entanglement (Two-Qubit Systems) | CX, CZ, Bell states, correlations | Create |Φ⁺⟩ and measure perfect correlations |

> 🔜 *Future modules planned:*  
> Module 5 – Deutsch-Jozsa Algorithm  
> Module 6 – Noise and Quantum Error Basics

---

## 🧮 Technologies Used

- **Python 3.10+**
- **[Streamlit](https://streamlit.io/)** — interactive web UI  
- **[Qiskit](https://qiskit.org/)** — quantum circuit simulation  
- **Plotly + Matplotlib** — histograms and Bloch sphere plotting  
- **NumPy** — statevector math and fidelity computation  

---

## 🖥️ Installation & Quick Start

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/quantum-learn-practice.git
cd quantum-learn-practice
 (Optional) Create a virtual environment
python -m venv .venv
.\.venv\Scripts\activate  # on Windows
# or
source .venv/bin/activate  # on Linux/Mac
Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
Launch the app
streamlit run app.py

🧩 Project Structure
quantum-learn-practice/
│
├── app.py                        # Main Streamlit app
├── requirements.txt
├── README.md
├── .gitignore
│
├── learn/
│   └── content_en.json           # Learning module data
│
├── logic/                        # (reserved for reusable helper scripts)
├── components/                   # (reserved for UI components)
│
└── .github/
    └── workflows/
        └── streamlit-check.yml   # CI import check
🧠 Learning Outcomes

After completing all modules, learners will:

Understand quantum state representation (|ψ⟩ = α|0⟩ + β|1⟩)

Build intuition for Bloch sphere rotations and measurements

Construct and analyze simple two-qubit entangled systems

Interpret simulation results statistically

Be prepared to explore advanced algorithms (Deutsch-Jozsa, Grover, etc.)

Contributing

Contributions are welcome!
Feel free to:

Add new modules (see learn/content_en.json format)

Improve UI elements or visualizations

Add persistence for user progress (e.g., SQLite)

Fork → Create branch → Commit → Pull Request 
📜 License

MIT License © 2025
Developed by Baturalp Bilen Burmaoğlu
Acknowledgements

Built using:

Qiskit

Streamlit

Plotly

NumPy
