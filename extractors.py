"""Local speech and slide extraction, loaded only when captions fail."""

from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
import math
import os
import re
import shutil


def timestamp(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def clean_speech_text(text):
    """Remove known sound cues; preserve meaningful bracketed content."""
    cue = (r"(?:music|applause|clapping|laughter|laughing|silence|inaudible|"
           r"background music|instrumental music|soft music|music playing|"
           r"\u0938\u0902\u0917\u0940\u0924|\u0924\u093e\u0932\u093f\u092f\u093e\u0902|"
           r"\u0924\u093e\u0932\u093f\u092f\u093e\u0901)")
    text = re.sub(r"[\[(]\s*" + cue + r"\s*[\])]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[\u266a\u266b]+", " ", text)
    # Ignore timestamps when checking whether anything meaningful remains.
    meaningful = re.sub(r"\[?\b\d{1,2}:\d{2}(?::\d{2})?\b\]?", " ", text)
    if re.fullmatch(r"(?:\s|[.,!\-]|" + cue + r")*", meaningful, flags=re.IGNORECASE):
        return ""
    return " ".join(text.split())


def download_media(url, directory, *, audio):
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import DownloadError

    options = {
        "format": "bestaudio/best" if audio else
                  "bestvideo[height<=720][ext=mp4]/best[height<=720]/best",
        "outtmpl": str(Path(directory) / ("audio.%(ext)s" if audio else "video.%(ext)s")),
        "noplaylist": True,
        "quiet": True,
        "socket_timeout": 30,
        "retries": 5,
        "fragment_retries": 5,
        "extractor_retries": 3,
        "continuedl": True,
        "skip_unavailable_fragments": False,
        "noprogress": True,
        "http_chunk_size": 1024 * 1024,
        "color": {"stdout": "never", "stderr": "never"},
    }
    runtimes = {}
    for runtime in ("deno", "node"):
        executable = shutil.which(runtime)
        if executable:
            runtimes[runtime] = {"path": executable}
    if runtimes:
        options["js_runtimes"] = runtimes
    # Download a single stream: no FFmpeg merge or conversion required.
    # Re-extract on failure to refresh expired URLs and resume the .part file.
    for attempt in range(3):
        try:
            with YoutubeDL(options) as downloader:
                info = downloader.extract_info(url, download=True)
                path = Path(downloader.prepare_filename(info))
            break
        except DownloadError as exc:
            if attempt == 2:
                message = re.sub(r"\x1b\[[0-9;]*m", "", str(exc))
                raise RuntimeError(
                    "Media download failed after 3 attempts. "
                    "Check your connection and update yt-dlp[default]. " + message
                ) from exc
    if not path.is_file():
        raise RuntimeError("The media download did not produce a file.")
    return path


@lru_cache(maxsize=1)
def whisper_model():
    from faster_whisper import WhisperModel

    return WhisperModel(os.getenv("WHISPER_MODEL", "base"),
                        device="cpu", compute_type="int8")


def transcribe_audio(path):
    segments, _ = whisper_model().transcribe(
        str(path), vad_filter=True, condition_on_previous_text=False,
    )
    # VAD and no-speech probability reduce hallucinations on silent videos/music.
    return "\n".join(
        f"[{timestamp(segment.start)}] {clean_speech_text(segment.text)}"
        for segment in segments
        if clean_speech_text(segment.text) and segment.no_speech_prob < 0.6
    )


@lru_cache(maxsize=2)
def ocr_reader(languages=("en", "hi")):
    import easyocr

    return easyocr.Reader(list(languages), gpu=False, quantize=True)


def extract_slides(path, report, interval=5, languages=("en", "hi")):
    import cv2

    if interval <= 0:
        raise ValueError("Frame interval must be positive.")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError("Could not open the downloaded video for OCR.")
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(fps) or fps <= 0 or not math.isfinite(frame_count) or frame_count <= 0:
            raise RuntimeError("Could not determine video duration for OCR.")
        duration = frame_count / fps
        reader = ocr_reader(languages)
        extracted = []
        readable_frames = 0
        for seconds in range(0, math.ceil(duration), interval):
            capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
            success, frame = capture.read()
            if not success:
                continue
            readable_frames += 1
            total = math.ceil(duration / interval)
            report(f"Reading slides: frame {seconds // interval + 1}/{total} "
                   f"({timestamp(seconds)} / {timestamp(duration)})")
            # Bound detector cost independently of the sampling interval.
            height, width = frame.shape[:2]
            scale = min(1.0, 1280 / max(height, width))
            if scale < 1:
                frame = cv2.resize(frame, (round(width * scale), round(height * scale)),
                                   interpolation=cv2.INTER_AREA)
            text = " ".join(reader.readtext(
                frame, detail=0, paragraph=True, decoder="greedy",
                canvas_size=1280, mag_ratio=1.0, contrast_ths=0.0,
            )).strip()
            if text:
                extracted.append(f"[Slide at {timestamp(seconds)}] {text}")
        if not readable_frames:
            raise RuntimeError("No video frames could be decoded for OCR.")
        return "\n\n".join(extracted)
    finally:
        capture.release()


def extract_fallback_content(url, report=None, *, ocr_languages=("en", "hi"), ocr_interval=5, force_ocr=False):
    """Try speech first, then OCR; always clean up downloaded media."""
    report = report or (lambda message: None)
    errors = []
    with TemporaryDirectory(prefix="youtube-rag-") as directory:
        if not force_ocr:
            try:
                report("Downloading audio for Whisper...")
                audio = download_media(url, directory, audio=True)
                report("Transcribing speech with Whisper (first run downloads the model)...")
                text = transcribe_audio(audio)
                if clean_speech_text(text):
                    report("Content source: Whisper speech transcription")
                    return text
                report("No usable speech found. Trying slide OCR...")
            except Exception as exc:
                errors.append(f"Speech extraction failed: {exc}")
                report("Speech extraction unavailable. Trying slide OCR...")

        try:
            report("Downloading video to read slides...")
            video = download_media(url, directory, audio=False)
            report("Loading OCR (first run downloads the models)...")
            text = extract_slides(video, report, interval=ocr_interval, languages=ocr_languages)
            if text.strip():
                report("Content source: slide OCR")
                return text
            errors.append("No readable slide text was found.")
        except Exception as exc:
            errors.append(f"Slide extraction failed: {exc}")

    raise RuntimeError("No usable content could be extracted. " + " ".join(errors))
