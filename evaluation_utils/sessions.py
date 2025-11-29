import requests
import time

from evaluation_utils.commons import BASE_URL, API_TOKEN
from agents.config import TwentyFourtyEightAgent


class Session:
    def __init__(self, session_id: str | None = None, renderer=None):
        self.session_id = session_id
        self.renderer = renderer
    
    def create(self):
        if self.renderer:
            self.renderer.event("Creating session...")

        response = requests.post(
            f"{BASE_URL}/sessions",
            headers={"Authorization": f"Token {API_TOKEN}"},
            params={"track": TwentyFourtyEightAgent.TRACK}
        )
        if not response.ok:
            self.renderer.event(f"Failed to create session: {response.text}")
            raise Exception(f"Failed to create session: {response.text}")
        self.session_id = response.json()["task_id"]
        submission_id = str(response.json()["submission_id"])

        if self.renderer:
            self.renderer.event(f"Session created: {self.session_id}")
            self.renderer.event(f"Submission ID: {submission_id}")
            # Live-update the config panel with new identifiers
            try:
                self.renderer.set_session_info(session_id=self.session_id, submission_id=submission_id)
            except Exception:
                print("Failed to set session info")
                # Non-fatal: UI update should not break session creation
                pass
    
    def get(self):
        response = requests.get(
            f"{BASE_URL}/sessions/{self.session_id}",
            headers={"Authorization": f"Token {API_TOKEN}"}
        )
        if not response.ok:
            self.renderer.event(f"Failed to get session: {response.text}")
            raise Exception(f"Failed to get session: {response.text}")
        
        data = response.json()
        self.submission_id = str(data["submission_id"])
        self.session_id = data["task_id"]
        if self.renderer:
            # Live-update the config panel with new identifiers
            try:
                self.renderer.set_session_info(session_id=self.session_id, submission_id=self.submission_id)
            except Exception as e:
                print(f"Failed to set session info: {e}")
                # Non-fatal: UI update should not break session creation
                pass

        return data
    
    def stop(self):
        response = requests.delete(
            f"{BASE_URL}/sessions/{self.session_id}",
            headers={"Authorization": f"Token {API_TOKEN}"}
        )
        return response.json()
    
    def wait_for_start(self, poll_interval: float = 1.0, timeout: float = 1500.0):
        start = time.time()
        last_status = None

        while True:
            status = self.get()["last_status"]

            # Only log status changes to avoid spam
            if status != last_status:
                if self.renderer:
                    self.renderer.event(f"Game server instance: {status}")
                last_status = status

            if status == "RUNNING":
                if self.renderer:
                    self.renderer.event("Game server has started")
                break
            
            if status in ["STOPPED"]:
                raise Exception("Session stopped. Start a new session next time.")

            if time.time() - start > timeout:
                raise TimeoutError("Timed out waiting for task to start")

            time.sleep(poll_interval)
