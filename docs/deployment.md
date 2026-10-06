# Deployment guide: UPC VM

How the API runs on the team's UPC virtual machine (Ubuntu, 1 CPU, 2 GB RAM, 24 GB
disk), and how to set it up, update it and debug it.

## Architecture

```
 Client (browser /docs, curl, demo)
        │  HTTP :80
        ▼
 ┌──────────────────────── UPC VM (Ubuntu) ────────────────────────┐
 │  nginx :80  ──proxy──▶  uvicorn 127.0.0.1:8000 (1 worker)        │
 │                          └─ FastAPI app, model loaded at startup │
 │  systemd: starts uvicorn at boot, restarts it if it crashes      │
 │  ~/taed2-astra: git checkout + .venv (uv) + models/model.pkl     │
 └──────────────────────────────────────────────────────────────────┘
        ▲ git pull (code)                  ▲ dvc pull (model only)
     GitHub                              DagsHub DVC remote
```

| Decision | Why |
|----------|-----|
| **No training on the VM.** It downloads the exact `model.pkl` in `dvc.lock` | That is the file the release gates tested. Retraining would produce an untested model with a different MD5, and the full dataset does not fit comfortably in 2 GB of RAM |
| **No data on the VM** | Serving does not need it, and a public-facing server should hold no patient records |
| **systemd** rather than `nohup`/`tmux` | Survives logout, reboot and crashes. Logs go to the journal |
| **nginx** in front of uvicorn | Only nginx is exposed. Uvicorn listens on localhost. It is also the place to add HTTPS or rate limiting later |
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

**4. Model, service and proxy**

```bash
make vm-setup          # pulls models/model.pkl (only the model, no data)
make service-install   # systemd unit from deploy/astra-api.service, enabled at boot
make nginx-install     # nginx site from deploy/nginx.conf on port 80
```

If the firewall is active (`sudo ufw status`), open HTTP with `sudo ufw allow 'Nginx HTTP'`.

**5. Check it**

```bash
make smoke API_URL=http://localhost     # on the VM, through nginx
```

From your laptop (on the UPC network or VPN, if the VM has no public IP), open
`http://<vm-address>/docs` or run `make smoke API_URL=http://<vm-address>`.

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
| `502 Bad Gateway` from nginx | uvicorn is down while nginx is up | `make status`, `make logs` |
| Browser cannot connect at all | Port 80 closed, or not on the UPC network/VPN | `sudo ufw allow 'Nginx HTTP'`, check the VPN |
