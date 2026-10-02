# The Server Whisperer

**Every failure whispers first.**

A machine learning system that learns what a healthy server looks like and warns before a failure happens, with a lead time, a plain-language reason, and a suggested action. Built as a final-year MCA data science project (Brainware University).

## What it does
- **Whisper level:** an Isolation Forest learns normal behaviour and measures drift early, with no failure labels needed.
- **Failure alert:** an XGBoost model predicts "failure within the next 60 minutes" and raises an alert only when it is confident.
- **Explanations:** every alert says why (SHAP) and suggests an action (suggestion only, nothing is changed automatically).
- **Live demo dashboard:** a "Break a server" button injects a memory leak, disk fill, or CPU spike and replays the incident, comparing the old threshold rule with The Server Whisperer.
- **Sonification:** a calm hum turns discordant, then beeps, as a server drifts toward failure.

## Results
Evaluated on held-out simulated servers that no model was trained or tuned on. To check that the result is not a lucky split, the whole comparison was repeated on 5 different random splits of the servers (average +/- spread across splits). Each model's alert threshold was chosen on separate validation servers to stay within the threshold rule's false-alarm budget.

| | Threshold rule (>90%) | Isolation Forest | XGBoost |
|---|---|---|---|
| Failures caught | 30/30 | 30/30 | 30/30 |
| Median warning (minutes) | 59 +/- 16 | 221 +/- 22 | 120 +/- 36 |
| False alarms per server-day | 0.26 +/- 0.03 | 0.12 +/- 0.08 | 0.08 +/- 0.09 |
| CPU spike warning (min) | 5 +/- 0.4 | 38 +/- 1.4 | 43 +/- 6.5 |
| Disk fill warning (min) | 129 +/- 1.6 | 508 +/- 82 | 341 +/- 165 |
| Memory leak warning (min) | 59 +/- 16 | 221 +/- 22 | 132 +/- 33 |

On average XGBoost produced about 70% fewer false alarms than the threshold rule and gave roughly twice as much warning, with the biggest gain on CPU spikes (about 43 minutes against 5). All three methods caught every failure, so the difference is in timing and noise, not detection. The spread between splits is large, so treat individual numbers as indicative.

## Quick start (Windows PowerShell)
```powershell
git clone https://github.com/Ansrr/server-whisperer.git
cd server-whisperer
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app\app.py
```
The trained models are included, so the dashboard runs straight away.

To rebuild everything from scratch:
```powershell
python src\simulator.py   # simulate a fleet of servers
python src\features.py    # engineer features
python src\baseline.py    # threshold rule baseline
python src\train.py       # train and compare models
python src\explain.py     # SHAP chart and sample explanations
```

## Project structure
| Path | Purpose |
|---|---|
| `src/simulator.py` | simulated server metrics with three fault types |
| `src/features.py` | rolling averages, slopes, drift, failure-within-60-minutes target |
| `src/baseline.py` | threshold rule baseline |
| `src/train.py` | Isolation Forest and XGBoost, split by server |
| `src/explain.py` | SHAP-based reasons and suggested actions |
| `src/sonify.py` | turns server health into sound |
| `app/app.py` | Streamlit dashboard |
| `models/`, `reports/` | trained models, results, SHAP chart |

## Limitations
- The data is simulated, so faults are cleaner than real ones. Next step: test on real data such as Backblaze drive stats.
- Each split tests on only 6 failing servers, and the same 24 failing servers are reshuffled between splits, so the 30 catches are not independent. Treat numbers as indicative.
- The alert gives a probability, not a countdown. A time-to-failure regressor is future work.
- The suggested actions are rule-based, driven by which resource shows the strongest signal.

## Acknowledgements
Early data exploration used the open-source Numenta Anomaly Benchmark (NAB).
