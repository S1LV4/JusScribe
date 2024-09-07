from colorama import init, Fore, Style
from difflib import get_close_matches
import json
import os
from pathlib import Path

init(autoreset=True)  # Inicializa colorama

import atexit
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import warnings
from enum import Enum
from pathlib import Path
from typing import Optional

import static_ffmpeg  # type: ignore
from appdirs import user_config_dir  # type: ignore

from jusScribe.audio import fetch_audio
from jusScribe.insanely_fast_whisper import run_insanely_fast_whisper
from jusScribe.logger import log_error
from jusScribe.util import chop_double_extension, sanitize_filename
from jusScribe.whisper import get_computing_device, run_whisper

DISABLED_WARNINGS = [
    ".*set_audio_backend has been deprecated.*",
    ".*torchaudio._backend.set_audio_backend has been deprecated.*",
]

IS_GITHUB = os.environ.get("GITHUB_ACTIONS", "false") == "true"

for warning in DISABLED_WARNINGS:
    warnings.filterwarnings("ignore", category=UserWarning, message=warning)

os.environ["PYTHONIOENCODING"] = "utf-8"

CACHE_FILE = os.path.join(user_config_dir("transcript-anything", "cache", roaming=True))

PERMS = (
    stat.S_IRUSR
    | stat.S_IRGRP
    | stat.S_IROTH
    | stat.S_IWOTH
    | stat.S_IWUSR
    | stat.S_IWGRP
)


class Device(Enum):
    """Device enum."""

    CPU = "cpu"
    CUDA = "cuda"
    INSANE = "insane"

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def from_str(device: str) -> "Device":
        """Returns the device from a string."""
        if device == "cpu":
            return Device.CPU
        if device == "cuda":
            return Device.CUDA
        if device == "insane":
            return Device.INSANE
        raise ValueError(f"Unknown device {device}")


def make_temp_wav() -> str:
    """
    Makes a temporary mp3 file and returns the path to it.
    """
    tmp = tempfile.NamedTemporaryFile(  # pylint: disable=consider-using-with
        suffix=".wav", delete=False
    )

    tmp.close()

    def cleanup() -> None:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)

    atexit.register(cleanup)
    return tmp.name


def fix_subtitles_path(_path: str) -> str:
    """Fixes windows subtitles path, which is weird."""
    if sys.platform != "win32":
        return _path
    # On Windows, ffmpeg 5 requires the path to be escaped.
    # For example, "C:\Users\user\file.srt" should be "C\\:/\Users/\user/\file.srt".
    # See https://stackoverflow.com/questions/60440793/how-can-i-use-windows-absolute-paths-with-the-movie-filter-on-ffmpeg
    path = Path(_path)
    # get the C:\ part
    drive = path.drive
    # get the \Users\user\file.srt part
    path = path.relative_to(drive)
    drive_fixed = str(drive).replace(":", "\\\\:")
    new_token = "/\\"
    old_token = "\\"
    path_fixed = str(path).replace(old_token, new_token)
    out_path = drive_fixed + path_fixed
    return out_path


def get_video_name_from_url(url: str) -> str:
    """
    Returns the video name.
    """
    assert url.startswith("http"), f" URL inválido {url}"

    # Try and the title of the video using yt-dlp
    # If that fails, use the basename of the url
    cmd_list: list[str] = []
    cmd_list.extend(["yt-dlp", "--get-title", f"{url}"])

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            cmd_str = subprocess.list2cmdline(cmd_list)
            print(f"Executando:\n  {cmd_str}")
            env = os.environ.copy()
            # env["set PYTHONIOENCODING=utf-8"]
            env["PYTHONIOENCODING"] = "utf-8"
            cp = subprocess.run(
                cmd_list,
                check=True,
                capture_output=True,
                universal_newlines=True,
                cwd=temp_dir,
                env=env,
            )
            stdout = cp.stdout
            lines = stdout.split("\n")
            lines = [line.strip() for line in lines]
            for line in lines:
                if "OSError" in line:
                    continue
                line = line[:80]
                return sanitize_filename(line)
            log_error("yt-dlp falhou ao obter o título, usando o nome da base em vez disso.")
            return os.path.basename(url)
    except subprocess.CalledProcessError:
        log_error("yt-dlp falhou ao obter o título, usando o nome da base em vez disso.")
        return os.path.basename(url)
    except Exception as exc:
        log_error(f"yt-dlp falhou com {exc}, usando o nome da base em vez disso.")
        return os.path.basename(url)


def check_python_in_range() -> None:
    """
    Returns whether the python version is in range.
    """
    # valid ranges are (3, 10, 0) to (3, 11, X)
    in_range = sys.version_info >= (3, 10, 0) and sys.version_info < (3, 12, 0)
    if not in_range:
        msg = f"# ALERTA: A versão do Python {sys.version_info} não está no intervalo (3, 10, 0) a (3, 11, X)."
        header = "\n\n#" + "#" * len(msg)
        footer = "#" + "#" * len(msg) + "\n\n"
        msg += "\n# ~JusScribe pode não funcionar corretamente. Por favor, use uma versão suportada do Python."
        warnings.warn(f"{header}\n{msg}\n{footer}\n")

# Carregar configurações
config_path = Path(os.path.expanduser("~")) / "Documents/jusScribe" / "config.json"
try:
    with open(config_path, 'r', encoding='utf-8') as config_file:
        CONFIG = json.load(config_file)
except FileNotFoundError:
    print(f"Arquivo de configuração não encontrado em {config_path}. Usando configurações padrão.")
    CONFIG = {
        "common_words": [],
        "legal_terms": [],
        "default_model": "small",
        "default_language": "Portuguese",
        "default_device": "cpu",
        "default_task": "transcribe"
    }

# Combinar palavras comuns e termos legais
COMMON_WORDS = CONFIG["common_words"] + CONFIG["legal_terms"]

def correct_words(text: str, word_list: list[str]) -> tuple[str, list[tuple[str, str]]]:
    """Corrige palavras mal identificadas no texto transcrito e retorna as correções."""
    words = text.split()
    corrected_words = []
    corrections = []
    for word in words:
        if word.lower() not in word_list:
            matches = get_close_matches(word.lower(), word_list, n=1, cutoff=0.8)
            if matches:
                corrected_words.append(matches[0])
                corrections.append((word, matches[0]))
            else:
                corrected_words.append(word)
        else:
            corrected_words.append(word)
    return ' '.join(corrected_words), corrections

def transcribe(
    url_or_file: str,
    output_dir: Optional[str] = None,
    model: Optional[str] = CONFIG["default_model"],
    task: Optional[str] = CONFIG["default_task"],
    language: Optional[str] = CONFIG["default_language"],
    device: Optional[str] = CONFIG["default_device"],
    embed: bool = False,
    hugging_face_token: Optional[str] = None,
    other_args: Optional[list[str]] = None,
    initial_prompt: Optional[str] = None,
) -> str:
    """
    Runs the program.
    """
    # add the paths for any dependent tools that may rely on ffmpeg
    static_ffmpeg.add_paths()
    check_python_in_range()
    if not os.path.isfile(url_or_file) and embed:
        raise NotImplementedError(
            "A incorporação é suportada apenas para arquivos locais "
            + "Faça download do arquivo primeiro."
        )
    # cache = DiskLRUCache(CACHE_FILE, 16)
    basename = os.path.basename(url_or_file)
    if not basename or basename == ".":  # if url_or_file is a directory
        # Defense against paths with a trailing /, for example:
        # https://example.com/, which will yield a basename of "".
        basename = os.path.basename(os.path.dirname(url_or_file))
        basename = sanitize_filename(basename)
    output_dir_was_generated = False
    if output_dir is None:
        output_dir_was_generated = True
        if url_or_file.startswith("http"):
            outname = get_video_name_from_url(url_or_file)
            output_dir = "text_" + sanitize_filename(outname)
        else:
            output_dir = "text_" + os.path.splitext(basename)[0]
    if output_dir_was_generated and language is not None:
        output_dir = os.path.join(output_dir, language)
    print(f"making dir {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    tmp_wav = make_temp_wav()
    assert os.path.isdir(
        output_dir
    ), f"O caminho  {output_dir} não foi encontrado ou não é um diretório."
    # tmp_mp3 = os.path.join(output_dir, "out.mp3")
    fetch_audio(url_or_file, tmp_wav)
    assert os.path.exists(tmp_wav), f"O caminho {tmp_wav} não existe."
    # filemd5 = md5(file.encode("utf-8")).hexdigest()
    # key = f"{file}-{filemd5}-{model}"
    # cached_data = cache.get_json(key)
    # print(f"Todo: cached data: {cached_data}")
    device = device or get_computing_device()
    device_enum = Device.from_str(device)
    if device_enum == Device.CUDA:
        print("#######################################")
        print("######### ACELERADO POR GPU! ##########")
        print("#######################################")
    elif device_enum == Device.INSANE:
        print("#######################################")
        print("####### MODO DE GPU INSANO! ###########")
        print("#######################################")
    elif device_enum == Device.CPU:
        print("ALERTA: A tarefa não foi acelerada por GPU, e está usando CPU (10x mais lento!)")
    else:
        raise ValueError(f"Dispositivo desconhecido {device}")
    print(f"Usando o dispositivo {device}")
    model_str = f"{model}" if model else ""
    task_str = f"{task}" if task else "transcrever"
    language_str = f"{language}" if language else ""

    print(f"{Fore.CYAN}Executando o sussurro em {tmp_wav} (instalará modelos na primeira execução){Style.RESET_ALL}")
    with tempfile.TemporaryDirectory() as tmpdir:
        if device_enum == Device.INSANE:
            run_insanely_fast_whisper(
                input_wav=Path(tmp_wav),
                model=model_str,
                output_dir=Path(tmpdir),
                task=task_str,
                language=language_str,
                hugging_face_token=hugging_face_token,
                other_args=other_args,
                initial_prompt=initial_prompt,
            )
        else:
            run_whisper(
                input_wav=Path(tmp_wav),
                device=str(device),
                model=model_str,
                output_dir=Path(tmpdir),
                task=task_str,
                language=language_str,
                other_args=other_args,
                initial_prompt=initial_prompt,
            )

        files = [os.path.join(tmpdir, name) for name in os.listdir(tmpdir)]
        srt_file: Optional[str] = None
        for file in files:
            # Change the filename to remove the double extension
            file_name = os.path.basename(file)
            base_path = os.path.dirname(file)
            new_file = os.path.join(base_path, chop_double_extension(file_name))
            _, ext = os.path.splitext(new_file)
            if "speaker.json" in new_file:  # pass through speaker.json
                outfile = os.path.join(output_dir, "speaker.json")
            else:
                outfile = os.path.join(output_dir, f"out{ext}")
            if os.path.exists(outfile):
                os.remove(outfile)
            assert os.path.isfile(file), f"O caminho do arquivo {file} não existe"
            assert not os.path.exists(outfile), f"O caminho {outfile} já existe."
            shutil.move(file, outfile)
            if ext == ".srt":
                srt_file = outfile
                # Corrigir palavras no arquivo SRT
                with open(srt_file, 'r', encoding='utf-8') as f:
                    srt_content = f.read()
                corrected_content, corrections = correct_words(srt_content, COMMON_WORDS)
                with open(srt_file, 'w', encoding='utf-8') as f:
                    f.write(corrected_content)
                
                # Imprimir as correções
                if corrections:
                    print(f"\n{Fore.YELLOW}Palavras corrigidas:{Style.RESET_ALL}")
                    for original, corrected in corrections:
                        print(f"{Fore.RED}{original}{Style.RESET_ALL} -> {Fore.GREEN}{corrected}{Style.RESET_ALL}")
                else:
                    print(f"\n{Fore.GREEN}Nenhuma palavra precisou ser corrigida.{Style.RESET_ALL}")
        output_dir = os.path.abspath(output_dir)
        assert srt_file is not None, "Nenhum arquivo .srt encontrado."
        srt_file = os.path.abspath(srt_file)
        if embed:
            assert os.path.isfile(url_or_file), f"O caminho  {url_or_file} não existe!"
            out_mp4 = os.path.join(output_dir, "out.mp4")
            embed_ffmpeg_cmd_list = [
                "static_ffmpeg",
                "-y",
                "-i",
                url_or_file,
                "-i",
                srt_file,
                "-vf",
                f"subtitles={fix_subtitles_path(srt_file)}",
                out_mp4,
            ]
            embed_ffmpeg_cmd = subprocess.list2cmdline(embed_ffmpeg_cmd_list)
            print(f"{Fore.CYAN}Executando:\n  {embed_ffmpeg_cmd}{Style.RESET_ALL}")
            rtn = subprocess.call(embed_ffmpeg_cmd_list, universal_newlines=True)
            if rtn != 0:
                warnings.warn(f"{Fore.RED}ffmpeg falhou com o código de retorno {rtn}{Style.RESET_ALL}")

    print(f"{Fore.GREEN}Transcrição concluída!{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Arquivos salvos em: {output_dir}{Style.RESET_ALL}")

    # Mostrar o conteúdo do arquivo .srt
    srt_file = Path(output_dir) / "out.srt"
    if srt_file.exists():
        print(f"\n{Fore.CYAN}Conteúdo da transcrição:{Style.RESET_ALL}")
        print(srt_file.read_text(encoding="utf-8"))

    return output_dir


if __name__ == "__main__":
    # test case for twitter video
    # transcribe(url_or_file="https://twitter.com/wlctv_ca/status/1598895698870951943")
    try:
        # transcribe(url_or_file="https://www.youtube.com/live/gBHFFM7-aCk?feature=share", output_dir="test")
        transcribe(
            url_or_file=r"E:\james_o_keefe_struggle_sessions.mp4",
            output_dir=r"E:\test2",
        )
    except KeyboardInterrupt:
        print("Keyboard interrupt")
        sys.exit(1)
