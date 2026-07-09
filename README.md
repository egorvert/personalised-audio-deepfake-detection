# Personalised Audio Deepfake Detection

A two-stream audio deepfake detector with optional per-speaker personalisation,
plus the Next.js webapp used to run the user study described in the
accompanying dissertation.

The detector fuses a WavLM-based stream and the AASIST anti-spoofing model and
adds a FiLM block that conditions the fused embedding on a small speaker
prototype built from a few seconds of bonafide audio.

The full dissertation — motivation, architecture, evaluation, and the user study —
is in [`docs/personalised-audio-deepfake-detection-dissertation.pdf`](docs/personalised-audio-deepfake-detection-dissertation.pdf)
(Egor Vert, BSc Computer Science, Queen Mary University of London, 2026).

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

The AASIST anti-spoofing stream is the official implementation, included
verbatim under `aasist/` with its original MIT licence and NOTICE file
([github.com/clovaai/aasist](https://github.com/clovaai/aasist)):

> Jung, J., Heo, H.-S., Tak, H., Shim, H., Chung, J.S., Lee, B.-J., Yu, H.-J.
> and Evans, N. (2021). *AASIST: Audio Anti-Spoofing using Integrated
> Spectro-Temporal Graph Attention Networks*. arXiv:2110.01200.

The WavLM stream uses Microsoft's `wavlm-base-plus` checkpoint via
[transformers](https://github.com/huggingface/transformers):

> Chen, S. et al. (2022). *WavLM: Large-Scale Self-Supervised Pre-Training for
> Full Stack Speech Processing*. IEEE Journal of Selected Topics in Signal
> Processing, 16(6), pp.1505–1518.

The attentive statistics pooling layer follows:

> Okabe, K., Koshinaka, T. and Shinoda, K. (2018). *Attentive Statistics
> Pooling for Deep Speaker Embedding*. INTERSPEECH 2018, pp.2252–2256.

The FiLM speaker-conditioning block follows:

> Perez, E., Strub, F., de Vries, H., Dumoulin, V. and Courville, A. (2017).
> *FiLM: Visual Reasoning with a General Conditioning Layer*.
> arXiv:1709.07871.

Training and evaluation use the ASVspoof 2019 LA dataset:

> Wang, X., Yamagishi, J., Todisco, M., Delgado, H., Nautsch, A., Evans, N.
> et al. (2020). *ASVspoof 2019: A Large-Scale Public Database of Synthesized,
> Converted and Replayed Speech*. arXiv:1911.01601.

Cross-dataset evaluations use:

> Müller, N. M., Czempin, P., Dieckmann, F., Froghyar, A. and Böttinger, K.
> (2026). *Does Audio Deepfake Detection Generalize?* arXiv:2203.16263.
> (In-the-Wild)
>
> Frank, J. and Schönherr, L. (2021). *WaveFake: A Data Set to Facilitate
> Audio Deepfake Detection*. arXiv:2111.02813.

The deepfake clips for the user study were synthesised with F5-TTS:

> Chen, Y., Niu, Z., Ma, Z., Deng, K., Wang, C., Zhao, J., Yu, K. and
> Chen, X. (2025). *F5-TTS: A Fairytaler that Fakes Fluent and Faithful
> Speech with Flow Matching*. arXiv:2410.06885.

## Licence

The original code in this repository is MIT-licensed. The vendored AASIST code
under `aasist/` is © NAVER Corp. and is distributed under its own MIT licence
(see `aasist/LICENSE` and `aasist/NOTICE`).

The dissertation under `docs/` is © Egor Vert 2026 and is licensed separately
under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/):
you are free to share and reference it with attribution.
