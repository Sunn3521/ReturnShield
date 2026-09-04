# ReturnShield AI

Cost-sensitive return-abuse risk and response agent for merchants.

## Overview

ReturnShield helps merchants reduce losses from abusive returns while minimizing unnecessary friction for legitimate customers.

For each return request, the system uses information available at the time of the request to estimate abuse risk, calculate expected merchant loss, and select an operational action:

- AUTO_APPROVE — low risk
- VERIFY — moderate risk; request additional evidence
- MANUAL_REVIEW — high risk; route to manual review

The product combines machine learning, cost-sensitive decisioning, coordinated-account analysis, real-time monitoring, and an AI agent.

## What the product includes

- Point-in-time synthetic event simulation
- Leakage-safe 30-day and 90-day historical features
- Logistic Regression baseline
- XGBoost risk model
- Probability calibration
- Validation-based cost-sensitive policy optimization
- Held-out temporal test evaluation
- Risk explanations
- NetworkX-based coordinated-account detection
- Live transaction REST API
- Continuous synthetic live transactions with changing risk regimes
- Real-time operations dashboard
- Return Investigator
- Coordinated Account / Abuse Ring Explorer
- Live charts, tables, and KPI updates
- Hover information on graph and chart elements
- Time-bounded CSV export
- ChatGPT-style AI Chat
- ReturnShield Agent API with tool calling
- Multiple AI provider options
- Right-side AI Settings panel

## System Architecture

```mermaid
flowchart TD
    A[User] --> B[AI Chat]
    B --> C[AI Provider Layer]

    C --> D[Google Gemini]
    C --> E[Groq]
    C --> F[OpenRouter]
    C --> G[Hugging Face]
    C --> H[Ollama]
    C --> I[OpenAI]

    D --> J[ReturnShield Agent API]
    E --> J
    F --> J
    G --> J
    H --> J
    I --> J

    J --> K[Returns]
    J --> L[Customers]
    J --> M[Clusters]
    J --> N[Risk / Policy Engine]

    N --> O[AUTO_APPROVE]
    N --> P[VERIFY]
    N --> Q[MANUAL_REVIEW]
