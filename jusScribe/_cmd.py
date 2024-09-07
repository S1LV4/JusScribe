"""
    Entry point for running the transcribe-anything prgram.
"""

# flake8: noqa E501
# pylint: disable=too-many-branches,import-outside-toplevel

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

# appdirs is used to get the cache directory
from appdirs import user_cache_dir  # type: ignore

from jusScribe.parse_whisper_options import parse_whisper_options
from jusScribe.whisper import get_computing_device

HERE = Path(os.path.abspath(os.path.dirname(__file__)))
WHISPER_OPTIONS = HERE / "WHISPER_OPTIONS.json"

os.environ["PYTHONIOENCODING"] = "utf-8"

WHISPER_MODEL_OPTIONS = [
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-legacy",
    "large",
    "large-v2",
    "large-v3",
    "distil-whisper/distil-large-v2",
]


def get_whisper_options() -> dict:
    """Get whisper options.""" ""
    if not WHISPER_OPTIONS.exists():
        whisper_options = parse_whisper_options()
        string = json.dumps(whisper_options, indent=4)
        WHISPER_OPTIONS.write_text(string)
        return whisper_options
    file_age = os.path.getmtime(WHISPER_OPTIONS)
    if file_age > 60 * 60 * 24 * 7:  # 1 week
        whisper_options = parse_whisper_options()
        string = json.dumps(whisper_options, indent=4)
        WHISPER_OPTIONS.write_text(string)
    return whisper_options


def parse_arguments() -> argparse.Namespace:
    """Parse arguments."""
    whisper_options = get_whisper_options()
    device = get_computing_device()
    help_str = (
        f'jusScribe está usando: "{device}".'
        ' Quaisquer argumentos não reconhecidos serão passados "como estão" para a WhisperAI.'
    )
    parser = argparse.ArgumentParser(description=help_str)
    parser.add_argument(
        "url_or_file",
        help="Forneça o caminho do arquivo local (c:\\caminho\\arquivo.wav) ou URL (youtube/etc)",
        nargs="?",
    )
    parser.add_argument(
        "--query-gpu-json-path",
        help=(
            "Consulta a GPU e armazena no caminho fornecido,"
            "!ATENÇÃO!: leva muito tempo no primeiro carregamento!"
        ),
        type=Path,
    )
    parser.add_argument(
        "--output_dir",
        help="Forneça o nome do diretório de saída, o padrão é o nome do arquivo.",
        default=None,
    )
    parser.add_argument(
        "--model",
        help='Defina oModelo do "Whisper" | Padrão: "small"',
        default="small",
        choices=WHISPER_MODEL_OPTIONS,
    )
    parser.add_argument(
        "--task",
        help="Realizar transcrição ou tradução.",
        default="transcribe",
        choices=whisper_options["task"],
    )
    parser.add_argument(
        "--language",
        help='língua do áudio de origem, Ex. Portuguese | Padrão: Portuguese',
        default="Portuguese",
        choices=[None, "Portuguese"] + whisper_options["language"],
    )
    parser.add_argument(
        "--device",
        help="dispositivo a ser usado para processamento, por padrão será selecionado automaticamente CUDA se disponível, caso contrário, CPU",
        default="cuda",
        choices=[None, "cuda", "cpu", "insane"],
    )
    parser.add_argument(
        "--hf_token",
        help="token de acesso do huggingface para download de modelos",
        default=None,
    )
    parser.add_argument(
        "--save_hf_token",
        help="salvar token de acesso do huggingface em um arquivo para uso futuro",
        action="store_true",
    )
    parser.add_argument(
        "--diarization_model",
        help=(
            "Nome do modelo/checkpoint pré-treinado para realizar a diarização."
            + " (padrão: pyannote/speaker-diarization). Só funciona para --device insane."
        ),
        default="pyannote/speaker-diarization-3.1",
    )
    parser.add_argument(
        "--timestamp",
        help=(
            "O Whisper suporta timestamps em nível de bloco ou de palavra. (padrão: bloco)."
            + " Only works for --device insane."
        ),
        choices=["chunk", "word"],
        default=None,
    )
    parser.add_argument(
        "--embed",
        help="se deve incorporar o arquivo de tradução no arquivo de saída",
        action="store_true",
    )
    parser.add_argument(
        "--initial_prompt",
        help="Prompt inicial para o modelo Whisper",
        default="Este vídeo contém conteúdo relacionado a processos judiciais, incluindo audiências, depoimentos de testemunhas, gravações de provas como conversas ou declarações em vídeo. O conteúdo transcrito pode ser utilizado como evidência ou parte de um procedimento legal, portanto, a precisão é essencial. Preste atenção especial à identificação de pessoas envolvidas, como réus, advogados, testemunhas e juízes. Ao transcrever, leve em consideração que os diálogos podem incluir linguagem formal e termos técnicos jurídicos, além de referências a eventos legais e depoimentos que podem ser relevantes para deliberações futuras. A transcrição exata de falas, interrupções e expressões relevantes deve ser priorizada."
        #default="The attached in portuguese videos for transcription contain content related to judicial proceedings, including court hearings, witness testimonies, and recorded evidence such as conversations or video statements. This content may be used as evidence or part of a legal process. During transcription, please consider that the dialogues may involve formal language, legal jargon, and references to legal events. Pay close attention to accurately identifying the individuals involved, such as defendants, lawyers, witnesses, and judges. Ensure that all spoken content is transcribed accurately, as it may be used in future legal deliberations or proceedings. Transcribe in Portuguese with high precision!"
        # Os vídeos anexados em português para transcrição contêm conteúdo relacionado a processos judiciais, incluindo audiências judiciais, depoimentos de testemunhas e provas gravadas, como conversas ou depoimentos em vídeo. 
        # Este conteúdo pode ser usado como prova ou parte de um processo legal. Durante a transcrição, considere que os diálogos podem envolver linguagem formal, jargão jurídico e referências a eventos jurídicos. 
        # Preste muita atenção para identificar com precisão os indivíduos envolvidos, como réus, advogados, testemunhas e juízes. 
        # Certifique-se de que todo o conteúdo falado seja transcrito com precisão, pois poderá ser utilizado em futuras deliberações ou procedimentos legais.
        # transcreva em português com alta precisão!",
    )
    args, unknown = parser.parse_known_args()
    if args.url_or_file is None and args.query_gpu_json_path is None:
        print("Nenhum arquivo ou URL fornecido")
        parser.print_help()
        sys.exit(1)
    args.unknown = unknown
    return args


def main() -> int:
    """Main entry point for the command line tool."""
    args = parse_arguments()
    unknown = args.unknown
    if args.query_gpu_json_path is not None:
        from jusScribe.insanely_fast_whisper import get_cuda_info

        json_str = get_cuda_info().to_json_str()
        path: Path = args.query_gpu_json_path
        path.write_text(json_str, encoding="utf-8")
        return 0
    if args.model == "large-legacy":
        args.model = "large"
    elif args.model == "large":
        print(
            "Utilizando o modelo large-v3 por padrão para --model large,"
            + "use --model large-legacy para o modelo antigo"
        )
        args.model = "large-v3"
    elif args.model is None and args.device == "insane":
        print("Utilizando o modelo large-v3 por padrão para --device insane")
        args.model = "large-v3"

    hf_token_path = Path(user_cache_dir(), "hf_token.txt")
    if args.hf_token is None:
        args.hf_token = os.environ.get("HF_TOKEN", None)
        if args.hf_token is None and hf_token_path.exists():
            # read from file
            args.hf_token = hf_token_path.read_text(encoding="utf-8").strip() or None
        if args.hf_token is None:
            args.diarization_model = None
    if args.save_hf_token:
        hf_token_path.write_text(args.hf_token or "", encoding="utf-8")
        print("Token do Hugging Face salvo em", hf_token_path)

    # For now, just stuff --diarization_model and --timestamp into unknown
    if args.diarization_model:
        if args.device != "insane":
            print(
                "-diarization_model só funciona com --device insane. Ignorando --diarization_model"
            )
        else:
            unknown.append(f"--diarization_model {args.diarization_model}")

    if args.timestamp:
        if args.device != "insane":
            print("--timestamp só funciona com --device insane. Ignorando --timestamp")
        else:
            unknown.append(f"--timestamp {args.timestamp}")


    if unknown:
        print(f"Argumentos passados para o backend do whisper: {unknown}")
    print(f"Executando a transcrição do arquivo: {args.url_or_file}")
    try:
        from jusScribe.api import transcribe

        transcribe(
            url_or_file=args.url_or_file,
            output_dir=args.output_dir,
            model=args.model if args.model != "None" else None,
            task=args.task,
            language=args.language if args.language != "None" else None,
            device=args.device,
            embed=args.embed,
            hugging_face_token=args.hf_token,
            other_args=unknown,
            initial_prompt=args.initial_prompt,
        )
    except KeyboardInterrupt:
        print("Interrompido pelo usuário")
        return 1
    except Exception as e:  # pylint: disable=broad-except
        stack = traceback.format_exc()
        sys.stderr.write(f"Erro: {e}\n{stack}\n Erro: enquanto processava {args.url_or_file}\n")
        return 1
    return 0


if __name__ == "__main__":
    # push sys argv prior to call
    sys.argv.append("test.wav")
    sys.argv.append("--model")
    sys.argv.append("large")
    sys.exit(main())
