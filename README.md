# Personalised Audio Deepfake Detection

A two-stream audio deepfake detector with optional per-speaker personalisation,
plus the Next.js webapp used to run the user study described in the
accompanying dissertation.

The detector fuses a WavLM-based stream and the AASIST anti-spoofing model and
adds a FiLM block that conditions the fused embedding on a small speaker
prototype built from a few seconds of bonafide audio.

## Layout

```
vdetect/          core detector (model, training, evaluation, FastAPI service, CLI)
aasist/           vendored copy of the official AASIST implementation (see Credits)
scripts/          eval clients (In-the-Wild, WaveFake) and study operations scripts
webapp/           Next.js + Supabase study webapp (Phase 1 enrolment, Phase 2 listening test)
deploy/           Caddyfile, PM2 ecosystem, macOS LaunchAgents used on the study host
assets/           checkpoints and enrolment DB (created at runtime; not in git)
```

## Setup

Requires Python 3.10+, Node 22 (only for the webapp), and ffmpeg on PATH for
browser-recorded webm files.

```bash
bash setup.sh                # creates .venv and installs requirements.txt
source .venv/bin/activate
```

For the deepfake generation script, also run `bash scripts/install_f5tts.sh`.

### Dataset

Training and evaluation expect ASVspoof 2019 LA at
`asvspoof_dataset/LA/LA/`:

```
asvspoof_dataset/LA/LA/
  ASVspoof2019_LA_cm_protocols/
  ASVspoof2019_LA_train/flac/
  ASVspoof2019_LA_dev/flac/
  ASVspoof2019_LA_eval/flac/
```

Download from https://datashare.ed.ac.uk/handle/10283/3336.

### AASIST weights

The vendored AASIST repo ships its config but not the pretrained weights.
Place `AASIST.pth` at `aasist/models/weights/AASIST.pth` (the official release
is on the AASIST repo's releases page).

## Train

WavLM baseline:

```bash
bash train.sh
# or:
python -m vdetect.train_wavlm \
  --data-root asvspoof_dataset/LA/LA \
  --epochs 10 --batch-size 8 \
  --out assets/checkpoints/wavlm_baseline.pt
```

Two-stream fusion head (needs the WavLM checkpoint from above + AASIST weights):

```bash
bash train_fusion.sh
```

Episodic FiLM fine-tuning:

```bash
python -m vdetect.train_fusion \
  --data-root asvspoof_dataset/LA/LA \
  --episodic \
  --init-fusion assets/checkpoints/two_stream.pt \
  --out assets/checkpoints/two_stream_film.pt
```

## Evaluate

```bash
# Compare WavLM, AASIST, fusion on the dev split:
bash evaluate_all.sh

# Single model:
python -m vdetect.evaluate_wavlm \
  --data-root asvspoof_dataset/LA/LA \
  --weights assets/checkpoints/wavlm_baseline.pt --split dev
```

Cross-dataset clients (talk to a running API):

```bash
python scripts/inthewild_eval_client.py \
  --api-url http://127.0.0.1:8000 \
  --dataset-dir ./release_in_the_wild \
  --max-per-class 2000 \
  --output results_inthewild.json

python scripts/wavefake_eval_client.py \
  --api-url http://127.0.0.1:8000 \
  --wavefake-dir ./wavefake_dataset \
  --bonafide-dir ./LJSpeech-1.1/wavs \
  --output results_wavefake.json
```

## Use

### CLI

```bash
python -m vdetect.cli detect path/to/audio.wav \
  --weights assets/checkpoints/wavlm_baseline.pt

python -m vdetect.cli batch-detect samples/ \
  --weights assets/checkpoints/wavlm_baseline.pt --output results.json

python -m vdetect.cli enroll alice \
  --audios a1.wav a2.wav a3.wav \
  --weights assets/checkpoints/two_stream.pt

python -m vdetect.cli info assets/checkpoints/wavlm_baseline.pt
```

### API

```bash
python -m vdetect            # serves on 127.0.0.1:8000 (override with VDETECT_HOST/PORT)
```

Then `POST /detect` with an audio file, `POST /enroll` with 3-5 samples and a
`speaker_id`, or `GET /health`.

### Webapp

```bash
cd webapp
cp .env.local.example .env.local
# fill in SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, VDETECT_BASE_URL, NEXT_PUBLIC_SITE_URL
npm install
npm run dev
```

Apply the migrations under `webapp/supabase/migrations/` to your Supabase
project before first use.

## Devices

The code automatically selects CUDA, then MPS (Apple Silicon), then CPU.
On Apple Silicon set `PYTORCH_ENABLE_MPS_FALLBACK=1` for the few ops that
still need CPU fallback.

## Credits

This project uses [AASIST](https://github.com/clovaai/aasist) by Jung et al.,
included verbatim under `aasist/` with its original MIT licence and NOTICE
file. The official release accompanies:

> Jung, J., Heo, H.-S., Tak, H., Shim, H.-j., Chung, J. S., Lee, B.-J.,
> Yu, H.-J. and Evans, N. (2021). *AASIST: Audio Anti-Spoofing using Integrated
> Spectro-Temporal Graph Attention Networks*. arXiv:2110.01200.

The WavLM stream uses Microsoft's `wavlm-base-plus` checkpoint via
[transformers](https://github.com/huggingface/transformers):

> Chen, S. et al. (2022). *WavLM: Large-Scale Self-Supervised Pre-Training for
> Full Stack Speech Processing*. IEEE J-STSP.

Training and evaluation use the ASVspoof 2019 LA dataset:

> Wang, X. et al. (2020). *ASVspoof 2019: A Large-Scale Public Database of
> Synthesized, Converted and Replayed Speech*. Computer Speech & Language,
> 64, 101114.

Cross-dataset evaluations use:

> Müller, N. M. et al. (2022). *Does Audio Deepfake Detection Generalize?*
> INTERSPEECH 2022, pp. 2783-2787. (In-the-Wild)
>
> Frank, J. and Schönherr, L. (2021). *WaveFake: A Data Set to Facilitate
> Audio Deepfake Detection*. NeurIPS Datasets and Benchmarks.

## Licence

The original code in this repository is MIT-licensed. The vendored AASIST code
under `aasist/` is © NAVER Corp. and is distributed under its own MIT licence
(see `aasist/LICENSE` and `aasist/NOTICE`).
