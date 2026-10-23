# ⚡ Energy Intelligence System

An interactive, data-driven energy management dashboard designed to analyze energy consumption, explore predictive insights, and support smarter energy optimization.

🌐 **Live Demo:** [Energy Intelligence System](https://energy-intelligence-system-aadtz5ac9eeipwfgbul6mz.streamlit.app/)

💻 **GitHub Repository:** [energy-intelligence-system](https://github.com/ashlin-theres/energy-intelligence-system)

## 📌 Overview

The Energy Intelligence System is a Python-based web application built with Streamlit. It provides an interactive platform for exploring energy consumption patterns, analyzing data, examining predictive insights, and simulating energy optimization strategies.

The project brings together data analytics, machine learning concepts, visualization, and occupancy-related intelligence in a single dashboard.

## 🎯 Objectives

- Analyze energy consumption data through interactive visualizations.
- Identify consumption patterns and trends.
- Explore predictive energy analysis.
- Simulate energy optimization scenarios.
- Investigate occupancy-aware energy management.
- Explore Wi-Fi data for additional intelligence and analysis.
- Present energy-related insights through a user-friendly dashboard.

## ✨ Key Features

### 📊 1. Overview Dashboard
Provides a central view of energy-related information through dashboard metrics and visualizations.

### 📈 2. Energy Analytics
Supports the exploration of energy consumption data, patterns, and trends using interactive charts.

### 🔮 3. Predictive Mode
Provides a dedicated interface for exploring predictive energy analysis.

### 🧠 4. AI Insights
Presents an interface for examining data-driven insights related to energy intelligence.

### ⚙️ 5. Optimization Simulator
Allows users to explore energy optimization scenarios and examine their potential outcomes.

### 🏢 6. Occupancy-Aware Energy Control
Explores how occupancy-related information can support smarter energy management decisions.

### 📡 7. Wi-Fi Intelligence
Includes Wi-Fi data analysis using the project's Wi-Fi datasets to explore additional patterns and insights.

### 📷 8. Person Detection
Includes a computer-vision component using YOLO and OpenCV to explore person detection for occupancy-related applications.

*Note: Actual feature availability and outputs depend on the configured datasets, model, and deployment environment.*

## 🛠️ Technology Stack

- **Programming Language:** Python
- **Web Framework:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Data Visualization:** Plotly
- **Machine Learning:** Scikit-learn, where used by the application
- **Computer Vision:** OpenCV, Ultralytics YOLO
- **Version Control:** Git and GitHub
- **Deployment:** Streamlit Community Cloud

## 📂 Project Structure

```text
energy-intelligence-system/
│
├── app.py
├── inspect_data.py
├── prepare_data.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── sample_energy.csv
├── WiFiData.csv
├── WiFiSurvey.csv
│
└── yolov8n.pt
```

## 🚀 Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/ashlin-theres/energy-intelligence-system.git
```

### 2. Navigate to the project folder

```bash
cd energy-intelligence-system
```

### 3. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Launch the application

```bash
streamlit run app.py
```

The application will open in your browser at the local address displayed in the terminal, usually `http://localhost:8501`.

## 📊 Datasets

The project directory includes the following datasets:

- `sample_energy.csv` — sample energy-related data.
- `WiFiData.csv` — numerical Wi-Fi dataset.
- `WiFiSurvey.csv` — Wi-Fi survey dataset.

Ensure that the required datasets are available at the file paths expected by the application. Dataset units, measurement periods, and data quality should be verified before using the results for real-world energy or cost estimates.

## 🌱 Potential Applications

- Building energy monitoring and analysis.
- Exploring energy-saving strategies.
- Occupancy-aware energy management research.
- Data-driven energy planning.
- Educational projects involving data analytics and machine learning.

## 🔭 Future Improvements

- Integrate live energy meter data.
- Improve predictive model evaluation and reporting.
- Add configurable energy-saving recommendations.
- Enhance real-time occupancy monitoring.
- Add more detailed performance metrics and validation.
- Improve deployment support for browser-based camera access.

## 👩‍💻 Developer

**Ashlin Theres James**

B.Tech — Artificial Intelligence and Data Science

- GitHub: [@ashlin-theres](https://github.com/ashlin-theres)
- LinkedIn: [Ashlin Theres James](https://linkedin.com/in/ashlin-theres-james-77b19132b)

  **Devika Swarup**

B.Tech — Artificial Intelligence and Data Science

- GitHub: [@ashlin-theres](https://github.com/ashlin-theres)
- LinkedIn: [Ashlin Theres James](https://linkedin.com/in/ashlin-theres-james-77b19132b)

  **Aiswarya kallayil Rajesh**

B.Tech — Artificial Intelligence and Data Science

- GitHub: [@ashlin-theres](https://github.com/ashlin-theres)
- LinkedIn: [Ashlin Theres James](https://linkedin.com/in/ashlin-theres-james-77b19132b)

## 🤝 Contributions

Suggestions, feedback, and improvements are welcome. Feel free to explore the repository and contribute to making energy analysis and optimization more accessible.

## 📄 License

No license has been specified yet. Please add an appropriate open-source license if you intend to permit reuse, modification, or redistribution.

---

**Energy Intelligence System — Turning energy data into actionable insights.** ⚡
