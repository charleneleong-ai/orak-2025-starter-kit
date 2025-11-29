from multiprocessing import Process
import os
import subprocess
import time
import shutil
import json

from evaluation_utils.commons import GAME_SERVER_PORTS, GAME_DATA_DIR
from evaluation_utils.renderer import Renderer


class GameLauncher:
    def __init__(self, renderer: Renderer, games: list[str] | None = None):
        self.renderer = renderer
        # If no specific games are provided, default to all known games
        self.games = games or list(GAME_SERVER_PORTS.keys())
        self.game_servers_procs = {}
        self.output_files = {}

        # Initialize all game servers as queued in the renderer
        for game in self.games:
            self.renderer.set_server_status(game, "queued")
            self.renderer.set_score(game, 0)

    def __del__(self):
        self.force_stop_all_games()
    
    def clean_game_data_dir(self):
        if os.path.exists(GAME_DATA_DIR):
            shutil.rmtree(GAME_DATA_DIR)
        os.makedirs(GAME_DATA_DIR)

    def _update_scores_from_disk(self):
        """Update renderer with scores read from disk."""
        for game in self.games:
            results_path = os.path.join(GAME_DATA_DIR, game, "game_results.json")
            score_val = 0
            try:
                if os.path.exists(results_path):
                    with open(results_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        score_val = int(data.get("score", 0))
            except Exception:
                score_val = 0
            self.renderer.set_score(game, score_val)
    
    def launch_game_server(self, game_name: str):
        if game_name in self.game_servers_procs:
            return self.game_servers_procs[game_name]

        self.renderer.set_server_status(game_name, "launching")

        app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        game_server_dir = os.path.join(app_dir, "evaluation_utils", "mcp_game_servers", game_name)
        game_server_script = os.path.join(game_server_dir, "server.py")
        game_data_dir = os.path.join(GAME_DATA_DIR, game_name)
        if not os.path.exists(game_data_dir):
            os.makedirs(game_data_dir)
        cmd = [
            "python",
            game_server_script,
        ]
        env = os.environ.copy()
        env["PORT"] = str(GAME_SERVER_PORTS[game_name])
        env["GAME_DATA_DIR"] = game_data_dir
        env["PYTHONPATH"] = os.path.join(app_dir, "evaluation_utils") + os.pathsep + app_dir
        env["GAME_ID"] = game_name

        log_file_path = os.path.join(game_data_dir, "game_server.log")
        self.output_files[game_name] = open(log_file_path, "w")

        proc = subprocess.Popen(cmd, env=env, stdout=self.output_files[game_name], stderr=self.output_files[game_name])
        self.game_servers_procs[game_name] = proc

        return proc

    def start_game_servers(self, games: list[str] | None = None):
        self.renderer.event("Initializing game servers...")

        game_list = games or self.games

        for game_name in game_list:
            self.launch_game_server(game_name)
            time.sleep(0.5)

        time.sleep(1.5)
        self.renderer.event("All game servers launched successfully")
    
    def clean_up_game_server(self, game_name: str):
        if os.path.exists(os.path.join(GAME_DATA_DIR, game_name, "game_results.json")):
            
            self.game_servers_procs[game_name].terminate()
            self.game_servers_procs[game_name].wait()
            
            del self.game_servers_procs[game_name]
            
            if game_name in self.output_files:
                self.output_files[game_name].close()
                del self.output_files[game_name]
    
    def stop_game_server(self, game_name: str, silent: bool = False):
        if game_name in self.game_servers_procs:
            if self.game_servers_procs[game_name].poll() is not None:
                self.clean_up_game_server(game_name)
                return

            if not silent:
                self.renderer.event(f"Shutting down {game_name}")
            # Only set to "stopped" if not already in a terminal state
            current_status = self.renderer.state.server_status_by_game.get(game_name)
            if current_status not in ("completed", "failed", "stopped"):
                self.renderer.set_server_status(game_name, "stopped")
            self.clean_up_game_server(game_name)

    def force_stop_all_games(self):
        for game_name in list(self.game_servers_procs.keys()):
            self.stop_game_server(game_name, silent=True)
    
    def wait_for_games_to_finish(self):
        completed_games: set[str] = set()
        total_games = len(self.game_servers_procs)

        while len(completed_games) < total_games:
            time.sleep(10)
            for game_name, proc in self.game_servers_procs.items():
                if game_name in completed_games:
                    continue

                results_path = os.path.join(GAME_DATA_DIR, game_name, "game_results.json")
                return_code = proc.poll()

                if return_code is not None:
                    if return_code != 0 or not os.path.exists(results_path):
                        self.renderer.warn(f"Game server {game_name} crashed with return code {return_code}")
                        self.renderer.set_server_status(game_name, "failed")
                        self.force_stop_all_games()
                        return

                if os.path.exists(results_path):
                    time.sleep(5)  # give a buffer for any pending writes
                    self.renderer.set_server_status(game_name, "completed")
                    self._update_scores_from_disk()
                    self.renderer.event(f"Game {game_name} completed")
                    completed_games.add(game_name)



if __name__ == "__main__":
    from evaluation_utils.renderer import get_renderer

    renderer = get_renderer()
    renderer.start(local=True)

    try:
        game_launcher = GameLauncher(renderer)
        game_launcher.start_game_servers()
        game_launcher.wait_for_games_to_finish()
    finally:
        game_launcher.force_stop_all_games()
        renderer.stop()
