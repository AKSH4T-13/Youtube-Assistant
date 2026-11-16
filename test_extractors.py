import unittest
from unittest.mock import patch

import extractors


class FallbackTests(unittest.TestCase):
    def test_force_ocr_skips_speech_and_audio_download(self):
        with patch.object(extractors, "download_media", return_value="video") as download, \
             patch.object(extractors, "transcribe_audio") as speech, \
             patch.object(extractors, "extract_slides", return_value="actual slide text"):
            self.assertEqual(extractors.extract_fallback_content("url", force_ocr=True),
                             "actual slide text")
            speech.assert_not_called()
            self.assertEqual(download.call_count, 1)
            self.assertFalse(download.call_args.kwargs["audio"])

    def test_sound_cues_are_not_usable_speech(self):
        for text in ("[Music] [Applause]", "(music) ♪♫", "Music Applause", "[संगीत]"):
            self.assertEqual(extractors.clean_speech_text(text), "")

    def test_sound_cues_do_not_erase_real_speech(self):
        self.assertEqual(extractors.clean_speech_text("[Music] Burning fuels causes warming."),
                         "Burning fuels causes warming.")
        self.assertEqual(extractors.clean_speech_text("[CO2] traps heat"), "[CO2] traps heat")

    def test_speech_success_skips_video_and_ocr(self):
        with patch.object(extractors, "download_media", return_value="audio") as download, \
             patch.object(extractors, "transcribe_audio", return_value="spoken content"), \
             patch.object(extractors, "extract_slides") as slides:
            self.assertEqual(extractors.extract_fallback_content("url"), "spoken content")
            self.assertTrue(download.call_args.kwargs["audio"])
            self.assertEqual(download.call_count, 1)
            slides.assert_not_called()

    def test_empty_speech_uses_ocr_and_cleans_downloads(self):
        directories = []

        def download(url, directory, *, audio):
            directories.append(directory)
            return "audio" if audio else "video"

        with patch.object(extractors, "download_media", side_effect=download), \
             patch.object(extractors, "transcribe_audio", return_value=""), \
             patch.object(extractors, "extract_slides", return_value="slide content"):
            self.assertEqual(extractors.extract_fallback_content("url"), "slide content")
        self.assertEqual(len(directories), 2)
        self.assertFalse(extractors.Path(directories[0]).exists())

    def test_speech_failure_still_tries_ocr(self):
        with patch.object(extractors, "download_media", return_value="media"), \
             patch.object(extractors, "transcribe_audio", side_effect=RuntimeError("decoder")), \
             patch.object(extractors, "extract_slides", return_value="slide content"):
            self.assertEqual(extractors.extract_fallback_content("url"), "slide content")

    def test_all_failures_report_both_stages(self):
        with patch.object(extractors, "download_media", return_value="media"), \
             patch.object(extractors, "transcribe_audio", side_effect=RuntimeError("speech error")), \
             patch.object(extractors, "extract_slides", side_effect=RuntimeError("ocr error")):
            with self.assertRaisesRegex(RuntimeError, "speech error.*ocr error"):
                extractors.extract_fallback_content("url")

    def test_silence_and_empty_slides_report_no_content(self):
        with patch.object(extractors, "download_media", return_value="media"), \
             patch.object(extractors, "transcribe_audio", return_value=""), \
             patch.object(extractors, "extract_slides", return_value=""):
            with self.assertRaisesRegex(RuntimeError, "No readable slide text"):
                extractors.extract_fallback_content("url")


if __name__ == "__main__":
    unittest.main()
