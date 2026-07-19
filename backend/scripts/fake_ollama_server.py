"""Standalone fake Ollama /api/chat server for manual end-to-end verification.

Run with: python scripts/fake_ollama_server.py
Then point OLLAMA_BASE_URL at http://localhost:11434 (its default) and run
the real backend against it - no real Ollama/model weights needed.
"""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer


def _reply_for(prompt: str) -> str:
    if "recommended_clarification_rounds" in prompt:
        return json.dumps(
            {
                "readiness_score": 62,
                "evaluation_feedback": ["SMS delivery failure path is undefined"],
                "recommended_clarification_rounds": 1,
            }
        )
    # Check QA-matrix marker BEFORE the BA-refiner marker: qa_matrix_builder's
    # prompt embeds the polished_spec text, which legitimately contains
    # "Core Calculation Framework" as a heading (from the BA response below),
    # so checking BA's marker first would misclassify the QA call too.
    if "senior QA Engineer" in prompt:
        return json.dumps(
            {
                "gaps_found": False,
                "questions": [],
                "test_matrix": [
                    {
                        "id": "TC-1",
                        "category": "sunny_day",
                        "title": "Transfer within daily limit succeeds",
                        "steps": [
                            {
                                "step_number": 1,
                                "action": "Submit a $500 transfer",
                                "expected_result": "Transfer completes",
                            }
                        ],
                        "status": "new",
                        "included": True,
                    }
                ],
            }
        )
    if "Core Calculation Framework" in prompt:
        return json.dumps(
            {
                "ambiguous": False,
                "questions": [],
                "polished_spec": "## Input Validation\nOTP must be 6 digits.\n\n"
                "## Core Calculation Framework\nDaily limit is $5000.\n\n"
                "## Network Architecture\nLedger call times out after 10s.\n\n"
                "## State Lifecycles\npending -> processing -> completed.",
            }
        )
    return "## TC-1\n| Step # | Action | Expected Result |\n| --- | --- | --- |\n| 1 | Submit a $500 transfer | Transfer completes |"


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        prompt = body.get("messages", [{}])[0].get("content", "")
        content = _reply_for(prompt)
        response = json.dumps({"message": {"content": content}}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    server = HTTPServer(("localhost", 11434), Handler)
    print("Fake Ollama server listening on http://localhost:11434")
    server.serve_forever()
