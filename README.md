# Warehouse & Delivery Performance — Graduation Project

## Scenario
A regional delivery company serving Assiut and nearby areas is receiving complaints about late deliveries. Management wants to understand on-time performance, compare regions and couriers, control shipping cost, and identify practical next actions.

The dataset is synthetic and intended for teaching. It contains 1,212 rows before cleaning (1,200 generated orders plus 12 exact duplicate rows), missing values, invalid ratings, and extreme shipping costs.

## Files
- `warehouse_delivery_raw.csv`: intentionally messy raw dataset.
- `app.py`: Streamlit dashboard.
- `requirements.txt`: Python dependencies.
- `trainer_guide.pdf`: two-hour guided lesson plan, prompts, code snippets, and answer cues.
- `analysis_starter.py`: compact analysis script covering NumPy, Pandas, Matplotlib, Seaborn, and Plotly.

## Run the dashboard
1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Run `pip install -r requirements.txt`
4. Run `streamlit run app.py`

## Suggested lesson outcome
Students should be able to import, inspect, clean, analyze, visualize, and communicate findings from a real-world-style operational dataset. Do not present synthetic-data findings as evidence about a real company.
