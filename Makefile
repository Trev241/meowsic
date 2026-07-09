.PHONY: setup test run verify clean

setup:
	python -m venv .venv
	.venv/Scripts/python -m pip install --upgrade pip
	.venv/Scripts/python -m pip install -e ".[demucs,youtube,dashboard,psola,dev]"
	.venv/Scripts/python -m pip install --upgrade --upgrade-strategy eager yt-dlp
	.venv/Scripts/python -m pip install "torch==2.6.0" "torchaudio==2.6.0" "soundfile>=0.12"

test:
	.venv/Scripts/python -m pytest tests/

run:
	.venv/Scripts/python app.py

verify:
	.venv/Scripts/python -c "import sys; print('Python:', sys.version.split()[0]); import importlib.util; [print(f'{pkg}: {\"installed\" if importlib.util.find_spec(pkg) else \"missing\"}') for pkg in ['torch', 'torchaudio', 'demucs', 'soundfile', 'yt_dlp', 'gradio']]; import torchaudio; print('torchaudio backends:', torchaudio.list_audio_backends() if importlib.util.find_spec('torchaudio') else 'none')"

clean:
	python -c "import shutil; shutil.rmtree('.venv', ignore_errors=True); shutil.rmtree('cache', ignore_errors=True)"
