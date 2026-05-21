# ScoutSync ML ⚾🤖

> A Cross-League Predictive Analytics & Translation Engine for Professional Baseball Scouting.

[![Live Demo](https://img.shields.io/badge/Demo-Streamlit_Cloud-FF4B4B?logo=streamlit)](https://scoutsync-ml-sumemxdakhf3zpuagqcyae.streamlit.app/)

[![CI/CD Pipeline](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF?logo=github-actions)](https://github.com/AnishAwasthi/scoutsync-ml/actions)

## 💡 The Core Problem

Evaluating talent across international (NPB, KBO) and amateur (NCAA, Cape Cod) leagues is plagued by competitive and environmental bias. A 98 mph fastball in college faces different contact profiles than in the MLB; hitting a ball 420 feet at high altitude (Tokyo Dome, Coors Field) distorts raw power metrics. 

**ScoutSync ML** strips away these structural biases. It ingests raw tracking metrics (exit velocity, launch angle, spin rates, vertical approach angles), dynamically accounts for local stadium altitudes/air density, and outputs context-neutral Major League baseline projections ($wOBA$, $ERA$) alongside 90% confidence intervals.

## 🛠️ System Architecture & Engineering Highlights

* **Predictive Core:** Optimized gradient-boosted trees (XGBoost) trained on historic translation baselines, outputting non-linear projection curves rather than static point-estimates.

* **Explainable AI (XAI):** Integrated **SHAP (Shapley Additive exPlanations)** natively into the evaluation layer to dissect the "black box" model, exposing the exact weight allocations (e.g., mapping stadium air density penalties vs. raw exit velocity bonuses) for scouts.

* **Resilient Infrastructure:** Implemented a thread-safe, lazy-loading database abstraction layer utilizing SQLAlchemy to decouple heavy analytical data pipelines from frontend rendering loops, guaranteeing sub-second page latency.