# Copilot Instructions

This workspace contains a Dockerized model evaluation gallery app.

- Production is an Ubuntu PC with an NVIDIA GPU.
- Real inference and evaluation must run in `production-cuda` mode.
- This app is intended to run only on the Ubuntu NVIDIA production runtime.
- Images are required for datasets; JSON annotations are optional.
- Images-only uploads create inference-only jobs.
- Images plus JSON annotations create evaluation-and-inference jobs.
- Keep runtime data under `storage/` and do not commit uploaded models, datasets, jobs, or generated results.
- Keep Docker packaging current with backend and frontend changes.