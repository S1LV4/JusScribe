"""
    Fetches audio and handles transcoding it for it.
"""

import os
import shutil
import subprocess
import sys
import tempfile

from jusScribe.util import PROCESS_TIMEOUT
from jusScribe.ytldp_download import ytdlp_download


def _convert_to_wav(
    inpath: str, outpath: str, speech_normalization: bool = False
) -> None:
    """Converts a file to wav."""
    cmd_audio_filter = ""
    if speech_normalization:
        cmd_audio_filter = "-filter:a speechnorm=e=12.5:r=0.00001:l=1"
    tmpwav = tempfile.NamedTemporaryFile(  # pylint: disable=consider-using-with
        suffix=".wav", delete=False
    )
    tmpwav.close()
    tmpwavepath = tmpwav.name
    audio_encoder = "-acodec pcm_s16le -ar 44100 -ac 1"
    cmd = f'static_ffmpeg -y -i "{inpath}" {cmd_audio_filter} {audio_encoder} "{tmpwavepath}"'
    print(f"Running:\n  {cmd}")
    try:
        subprocess.run(
            cmd, shell=True, check=True, capture_output=True, timeout=PROCESS_TIMEOUT
        )
    except subprocess.CalledProcessError as exc:
        print(f"Falha ao executar {cmd} com erro {exc}")
        print(f"stdout: {exc.stdout}")
        print(f"stderr: {exc.stderr}")
        raise
    os.remove(outpath)
    # os.rename(tmpwavepath, outpath)
    # overwrite file at outpath
    shutil.copyfile(tmpwavepath, outpath)
    assert os.path.exists(outpath), f"O arquivo esperado {outpath} não existe"


def fetch_audio(url_or_file: str, out_wav: str) -> None:
    """Fetches from the internet or from a local file and outputs a wav file."""
    assert out_wav.endswith(".wav")
    if url_or_file.startswith("http") or url_or_file.startswith("ftp"):
        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"Usando o diretório temporário {tmpdir}")
            downloaded_file = ytdlp_download(url_or_file, os.path.abspath(tmpdir))
            print("Baixando arquivo: ", downloaded_file)
            _convert_to_wav(downloaded_file, out_wav, speech_normalization=True)
        sys.stderr.write("Download completo.\n")
        assert os.path.exists(out_wav), f"O arquivo esperado {out_wav} não existe!"
    else:
        assert os.path.isfile(url_or_file)
        abspath = os.path.abspath(url_or_file)
        out_wav_abs = os.path.abspath(out_wav)
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = f'static_ffmpeg -y -i "{abspath}" -acodec pcm_s16le -ar 44100 -ac 1 out.wav'
            sys.stderr.write(f"Executando:\n  {cmd}\n")
            subprocess.run(
                cmd,
                cwd=tmpdir,
                shell=True,
                check=True,
                capture_output=True,
                timeout=PROCESS_TIMEOUT,
            )
            shutil.copyfile(os.path.join(tmpdir, "out.wav"), out_wav_abs)
        assert os.path.exists(out_wav), f"O arquivo esperado {out_wav} não existe!"


def unit_test() -> None:
    """Runs the program."""
    url = "https://youtube.com/shorts/fbg83R24lc4?si"
    fetch_audio(url, "out.wav")


if __name__ == "__main__":
    unit_test()
