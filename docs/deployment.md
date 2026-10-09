# Deployment guide: UPC VM

How the API runs on the team's UPC virtual machine (Ubuntu, 1 CPU, 2 GB RAM, 24 GB
disk), and how to set it up, update it and debug it.

## Architecture

```
 Client (browser /docs, curl, demo)
        │  HTTPS  astra.quick2query.com
        ▼
 Cloudflare ──tunnel──┐
                      ▼
 ┌──────────────────────── UPC VM (Ubuntu) ────────────────────────┐
 │  cloudflared  ──▶  uvicorn 127.0.0.1:8000 (1 worker)             │
 │                     └─ FastAPI app, model loaded at startup      │
 │  systemd: starts uvicorn at boot, restarts it if it crashes      │
 │  ~/taed2-astra: git checkout + .venv (uv) + models/model.pkl     │
 └──────────────────────────────────────────────────────────────────┘
        ▲ git pull (code)                  ▲ dvc pull (model only)
     GitHub                              DagsHub DVC remote
```

How this fits with the training pipeline, and the design choices behind it, is in
[system_design.md](system_design.md).

| Decision | Why |
|----------|-----|
| **No training on the VM.** It downloads the exact `model.pkl` in `dvc.lock` | That is the file the release gates tested. Retraining would produce an untested model with a different MD5, and the full dataset does not fit comfortably in 2 GB of RAM |
| **No data on the VM** | Serving does not need it, and a public-facing server should hold no patient records |
| **systemd** rather than `nohup`/`tmux` | Survives logout, reboot and crashes. Logs go to the journal |
| **Cloudflare tunnel** for public access | A public HTTPS name, `astra.quick2query.com`, for a VM on the university network. `cloudflared` connects outwards to Cloudflare, which terminates TLS, so the VM needs no certificate of its own |
| **No reverse proxy** (no nginx) | The tunnel forwards straight to uvicorn, which listens only on localhost, so the only way in is through Cloudflare. One less service to install and keep running. Trade-off: no proxy caps the request body size, so a huge body reaches the app before the batch limit rejects it |
| **No IP addresses in the logs** | uvicorn runs with `--no-access-log`; the app logs method, path, status and latency only. Client IPs are personal data that operating the service does not need |
| **One worker** | One CPU, and each worker holds its own copy of the model and libraries |
| **No Docker** | One service on one VM. A container would add a build step and an image registry without isolating anything we need |
| **`uv sync --frozen`** | Installs the exact versions in `uv.lock`, including the scikit-learn that pickled the model |

## One-time setup

Run these on the VM over SSH. Each step only has to be done once per VM.

**1. System packages and uv**

```bash
sudo apt-get update && sudo apt-get install -y git make curl
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

**2. Clone the repository.** Use a read-only GitHub deploy key, so no personal
credentials live on the VM:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/github_deploy -N "" -C "astra-vm"
cat ~/.ssh/github_deploy.pub   # add it in GitHub: repo Settings → Deploy keys (read-only)
printf "Host github.com\n  IdentityFile ~/.ssh/github_deploy\n" >> ~/.ssh/config
git clone git@github.com:taed2-2627q1-gced-upc/taed2-astra.git
cd taed2-astra
```

**3. Runtime dependencies and DVC credentials.** `--no-dev` skips the test and lint
tools. DagsHub issues one token: use it for both fields. It is written to
`.dvc/config.local`, which is gitignored. No `.env` is needed, because the API does not use MLflow.

```bash
uv sync --frozen --no-dev
uv run --no-dev dvc remote modify --local dagshub access_key_id     <token>
uv run --no-dev dvc remote modify --local dagshub secret_access_key <token>
```

**4. Model and service**

```bash
make vm-setup          # pulls models/model.pkl (only the model, no data)
make service-install   # systemd unit from deploy/astra-api.service, enabled at boot
```

Public access goes through a Cloudflare tunnel whose public hostname forwards to
`http://localhost:8000`.

<!-- TODO(team): document how the Cloudflare tunnel was created (cloudflared install and
public hostname), without the tunnel token. -->

**5. Check it**

```bash
make smoke                                        # on the VM, straight to uvicorn
make smoke API_URL=https://astra.quick2query.com  # from anywhere, through Cloudflare
```

From your laptop, open <https://astra.quick2query.com/docs>.

## Releasing a new version

After a pull request is merged into `main` (and its model pushed with `dvc push`):

```bash
cd ~/taed2-astra && make deploy
```

`make deploy` pulls the code, syncs dependencies, pulls the model, restarts the
service and runs the smoke test. If the smoke test fails, the release did not work.
Check `/model`: its `model_md5` must match `models/model.pkl` in `dvc.lock` on `main`.

## Operating the service

| Task | Command |
|------|---------|
| Is it running? | `make status` |
| Follow the request log | `make logs` |
| Restart | `sudo systemctl restart astra-api` |
| Stop / start | `sudo systemctl stop astra-api` / `sudo systemctl start astra-api` |
| Run in the foreground to debug | `sudo systemctl stop astra-api && make serve` |

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `make status` shows `activating (auto-restart)` | Startup failed. `make logs` shows why | See the next two rows |
| Log: `FileNotFoundError: No model at ...` | The model was never pulled | `make vm-setup` |
| Log: `Model features ... do not match params.yaml` | Code and model come from different commits | `make deploy` so both match `main` |
| `dvc pull` reports missing files | Whoever last ran `dvc repro` did not `dvc push` | Push from that machine, then retry |
| Cloudflare error page (`502`, `1033`) | uvicorn or `cloudflared` is down on the VM | `make status`, `make logs`, `systemctl status cloudflared` |
