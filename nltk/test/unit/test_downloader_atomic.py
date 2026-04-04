import os
import shutil
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

from nltk.downloader import Downloader, Package


class TestDownloaderAtomic(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        # Size and unzipped_size must exactly match the length of b"data" (4 bytes)
        self.pkg = Package(
            id="abc",
            url="http://example.com/abc.zip",
            size=4,
            unzipped_size=4,
            filename="abc.zip",
            checksum="mock_md5",
            unzip=False,
        )

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch("nltk.downloader.md5_hexdigest", return_value="mock_md5")
    @patch("time.sleep", side_effect=time.sleep)
    @patch("nltk.downloader.urlopen")
    def test_concurrent_downloads_cooperate(self, mock_urlopen, mock_sleep, mock_md5):
        """Verify multiple parallel calls result in exactly ONE network request."""

        def side_effect(*args, **kwargs):
            # A 100ms delay ensures all threads hit the loop while the file is downloading
            time.sleep(0.1)
            mock_resp = MagicMock()
            mock_resp.read.side_effect = [b"data", b""]
            return mock_resp

        mock_urlopen.side_effect = side_effect
        dl = Downloader(download_dir=self.test_dir)
        num_concurrent = 3

        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [
                executor.submit(dl.download, self.pkg, quiet=True)
                for _ in range(num_concurrent)
            ]
            results = [f.result() for f in futures]

        self.assertTrue(all(results))
        # PROOF: Our atomic lock prevents redundant downloads
        self.assertEqual(mock_urlopen.call_count, 1)
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "abc.zip")))
