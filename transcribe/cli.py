import argparse
import sys
import time
from pathlib import Path

from transcribe.core import save_json, save_txt, transcribe_video


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe video files using faster-whisper.",
    )
    parser.add_argument("video", help="Path to the video file")
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
        help="Output directory (default: same as video file)",
    )
    parser.add_argument(
        "--format",
        default="all",
        choices=["txt", "json", "all"],
        help="Output format (default: all)",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else video_path.parent
    stem = video_path.stem

    print(f"Transcribing: {video_path}")
    print(f"Model: {args.model}")
    print(f"Language: {args.language or 'auto-detect'}")

    start = time.time()
    result = transcribe_video(
        str(video_path),
        model_size=args.model,
        language=args.language,
        output_dir=str(output_dir),
    )
    elapsed = time.time() - start

    print(f"Detected language: {result['language']}")
    print(f"Segments: {len(result['segments'])}")
    print(f"Time: {elapsed:.1f}s")

    saved = []
    if args.format in ("txt", "all"):
        path = save_txt(result, output_dir / f"{stem}.txt")
        saved.append(str(path))
    if args.format in ("json", "all"):
        path = save_json(result, output_dir / f"{stem}.json")
        saved.append(str(path))

    for p in saved:
        print(f"Saved: {p}")


if __name__ == "__main__":
    main()
