import argparse
import os
import sys
import time
from pathlib import Path

from transcribe.core import save_txt, transcribe_media


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe video and audio files using whisper.cpp.",
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Path(s) to media file(s) (video or audio)",
    )
    parser.add_argument(
        "--model",
        default="large-v3",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size (default: large-v3)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language code (e.g. 'en'). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: same as each input file)",
    )
    parser.add_argument(
        "--speakers",
        action="store_true",
        default=False,
        help="Enable speaker diarization (identify different speakers)",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Force a specific number of speakers (default: auto-detect)",
    )
    args = parser.parse_args()

    # Token is read from environment only — never from CLI args (visible in
    # process listings and shell history).
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if args.speakers and not hf_token:
        print(
            "Error: speaker diarization requires a HuggingFace token. "
            "Set the HF_TOKEN or HUGGINGFACE_TOKEN environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate all files exist before starting
    media_paths = []
    for f in args.files:
        p = Path(f)
        if not p.exists():
            print(f"Error: file not found: {p}", file=sys.stderr)
            sys.exit(1)
        media_paths.append(p)

    total = len(media_paths)
    failed = 0

    for idx, media_path in enumerate(media_paths, 1):
        output_dir = Path(args.output_dir) if args.output_dir else media_path.parent
        stem = media_path.stem

        if total > 1:
            print(f"\n[{idx}/{total}] Transcribing: {media_path}")
        else:
            print(f"Transcribing: {media_path}")
        print(f"Model: {args.model}")
        print(f"Language: {args.language or 'auto-detect'}")
        if args.speakers:
            print("Speaker diarization: enabled")

        start = time.time()
        try:
            result = transcribe_media(
                str(media_path),
                model_size=args.model,
                language=args.language,
                output_dir=str(output_dir),
                diarize=args.speakers,
                num_speakers=args.num_speakers,
                hf_token=hf_token,
            )
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            failed += 1
            continue

        elapsed = time.time() - start

        print(f"Detected language: {result['language']}")
        print(f"Segments: {len(result['segments'])}")
        if result.get("speakers"):
            print(f"Speakers: {result['speakers']}")
        print(f"Time: {elapsed:.1f}s")

        path = save_txt(result, output_dir / f"{stem}.txt")
        print(f"Saved: {path}")

    if total > 1:
        print(f"\nBatch complete: {total - failed} succeeded, {failed} failed")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
