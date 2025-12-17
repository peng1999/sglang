import re
import unittest

import requests

from sglang.test.test_disaggregation_utils import TestDisaggregationBase
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
)


class TestDisaggregationMetrics(TestDisaggregationBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST

        # Non blocking start servers
        cls.start_prefill()
        cls.start_decode()

        # Block until both
        cls.wait_server_ready(cls.prefill_url + "/health")
        cls.wait_server_ready(cls.decode_url + "/health")

        cls.launch_lb()

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--tp",
            "1",
            "--enable-metrics",
        ]
        prefill_args += cls.transfer_backend + cls.rdma_devices
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
        )

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--tp",
            "1",
            "--base-gpu-id",
            "1",
            "--enable-metrics",
        ]
        decode_args += cls.transfer_backend + cls.rdma_devices
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
        )

    def test_num_running_reqs_metric_labels(self):
        response = requests.post(
            self.lb_url + "/generate",
            json={
                "text": "Hello",
                "sampling_params": {"temperature": 0, "max_new_tokens": 8},
            },
            timeout=180,
        )
        self.assertEqual(response.status_code, 200)

        prefill_metrics = requests.get(self.prefill_url + "/metrics").text
        decode_metrics = requests.get(self.decode_url + "/metrics").text

        prefill_match = re.search(
            r'sglang:num_running_reqs\{[^}]*engine_type="prefill"[^}]*\}\s+([-+]?\d+(?:\.\d+)?)',
            prefill_metrics,
        )
        self.assertIsNotNone(
            prefill_match,
            "num_running_reqs gauge for prefill engine not found in metrics",
        )

        decode_match = re.search(
            r'sglang:num_running_reqs\{[^}]*engine_type="decode"[^}]*\}\s+([-+]?\d+(?:\.\d+)?)',
            decode_metrics,
        )
        self.assertIsNotNone(
            decode_match,
            "num_running_reqs gauge for decode engine not found in metrics",
        )


if __name__ == "__main__":
    unittest.main()
