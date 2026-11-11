"""验证断流重试与独立目录续传；不访问网络或下载真实模型。"""
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from download_large import download_files, retry_network


class ChunkedEncodingError(Exception):
    pass


class DownloadRetryTests(unittest.TestCase):
    @patch("download_large.time.sleep")
    def test_resume_same_new_directory_after_broken_stream(self, sleep):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            completed = directory / "model-00001.safetensors"
            completed.write_bytes(b"completed")
            partial = directory / ".cache" / "part.incomplete"
            partial.parent.mkdir()
            partial.write_bytes(b"start")
            calls = []

            def backend(**kwargs):
                calls.append(kwargs)
                if kwargs["filename"] == completed.name:
                    return str(completed)
                if sum(row["filename"] == "model-00002.safetensors" for row in calls) == 1:
                    raise ChunkedEncodingError("IncompleteRead")
                self.assertEqual(partial.read_bytes(), b"start")
                result = directory / kwargs["filename"]
                result.write_bytes(b"start-end")
                return str(result)

            files = [SimpleNamespace(rfilename=completed.name, size=9),
                     SimpleNamespace(rfilename="model-00002.safetensors", size=9)]
            with redirect_stdout(io.StringIO()):
                download_files(directory, files, backend, 3, ChunkedEncodingError)
            self.assertEqual(completed.read_bytes(), b"completed")
            self.assertEqual(len(calls), 3)
            self.assertTrue(all(row["force_download"] is False for row in calls))
            self.assertTrue(all(row["local_dir"] == str(directory) for row in calls))
            self.assertTrue(all(row["cache_dir"] == str(directory / ".cache" / "isolated-hub")
                                for row in calls))
            sleep.assert_called_once_with(5)

    @patch("download_large.time.sleep")
    def test_exhausted_retries_preserve_original_error(self, sleep):
        original = ChunkedEncodingError("broken body")
        operation = Mock(side_effect=original)
        with redirect_stdout(io.StringIO()), self.assertRaises(ChunkedEncodingError) as caught:
            retry_network("shard", operation, 2, ChunkedEncodingError)
        self.assertIs(caught.exception, original)
        self.assertEqual(operation.call_count, 2)
        sleep.assert_called_once_with(5)

    @patch("download_large.time.sleep")
    def test_permanent_http_error_is_not_retried(self, sleep):
        error = ChunkedEncodingError("not found")
        error.response = SimpleNamespace(status_code=404)
        operation = Mock(side_effect=error)
        with self.assertRaises(ChunkedEncodingError):
            retry_network("metadata", operation, 5, ChunkedEncodingError)
        operation.assert_called_once()
        sleep.assert_not_called()

    def test_wrong_file_size_stops_before_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "shard.safetensors"
            path.write_bytes(b"short")
            with redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                download_files(directory, [SimpleNamespace(rfilename=path.name, size=99)],
                               Mock(return_value=str(path)), 1, ChunkedEncodingError)


if __name__ == "__main__":
    unittest.main()
