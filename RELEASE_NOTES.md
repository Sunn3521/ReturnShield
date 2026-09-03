# ReturnShield v38 — Release Notes

This build consolidates the current hackathon product behavior and documentation.

## UI

- Chat UI uses a ChatGPT-style conversation layout.
- Chat header and AI Settings control remain visible.
- AI Settings opens as a right-side drawer.
- Suggested-query buttons are removed.
- The chat composer stays available at the bottom.
- The conversation is the only scrolling region in the Chat tab.

## Live operations

- Built-in synthetic REST server continuously generates return transactions.
- Live operational data is designed for approximately one-second updates.
- Charts, KPIs, tables, investigation data, and cluster data use the live source.
- Visual refresh behavior is designed to avoid full-page refreshes and distracting fade transitions.

## Investigation

- Return Investigator retains the complete investigation workflow.
- Live return-request selection is refreshed without replacing the investigation workflow itself.
- Risk evidence and operational response fields remain available.

## Analytics

- Policy Actions uses a pie chart.
- Chart and graph elements provide contextual hover information where supported.
- Cluster Explorer presents coordinated-account relationships in a more interpretable form.

## AI Chat

- Chat uses the ReturnShield Agent API as the application/tool layer.
- Supported LLM providers include OpenRouter, Groq, Google Gemini, Hugging Face, Ollama, and OpenAI.
- The LLM can answer general questions and use allow-listed ReturnShield tools for current data and supported actions.
- The LLM does not override the deterministic risk policy.

## Documentation

The README and demo guide have been updated to describe the current architecture, live behavior, Chat UI, provider setup, investigation workflow, evaluation methodology, and hackathon demo flow.
