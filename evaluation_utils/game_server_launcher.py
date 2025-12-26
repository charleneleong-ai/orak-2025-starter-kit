from multiprocessing import Process
import os
import subprocess
import time
import shutil
import json
import omegaconf
from dotenv import load_dotenv

from evaluation_utils.commons import GAME_SERVER_PORTS, GAME_DATA_DIR
from evaluation_utils.renderer import get_renderer, Renderer
from config.base import Settings
from config.utils import load_hydra_settings

load_dotenv()

class GameLauncher:
    def __init__(self, renderer: Renderer, settings: Settings | None = None):
        self.renderer = renderer
        self.settings = settings
        # If no specific games are provided, default to all known games
        self.games = self.load_games() or list(GAME_SERVER_PORTS.keys())
        self.game_servers_procs = {}
        self.output_files = {}

        # Initialize all game servers as queued in the renderer
        for game in self.games:
            self.renderer.set_server_status(game, "queued")
            self.renderer.set_score(game, 0)

    def __del__(self):
        self.force_stop_all_games()
        
    def load_games(self) -> list[str]:
        self.games = []
        for g in list(GAME_SERVER_PORTS.keys()):
            for s in self.settings.__dict__.keys():
                self.renderer.info(f"Checking if game {g} is enabled in settings")
                if g == s and getattr(self.settings, g) is not None:
                    self.renderer.event(f"Adding game {g} to game launcher")
                    self.games.append(g)
        return self.games
    
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
            
        # Generate config file from settings if available
        config_path = None
        if self.settings:
            env_config = None
            # Dynamically get the game config from settings using the game_name
            game_config = getattr(self.settings, game_name, None)
            if game_config and hasattr(game_config, "env"):
                env_config = game_config.env
            self.renderer.event(f"Using config for {game_name}: {env_config}")
            if env_config and env_config is not None:
                config_path = os.path.join(game_data_dir, "config.yaml")
                
                # Convert dataclass/pydantic model to dict
                if hasattr(env_config, "model_dump"):
                    data = env_config.model_dump()
                else:
                    from dataclasses import asdict
                    data = asdict(env_config)
                
                # Restructure for game server config format
                common_fields = ["env_name", "log_path"]
                yaml_data = {k: v for k, v in data.items() if k in common_fields}
                yaml_data["env"] = {k: v for k, v in data.items() if k not in common_fields}
                
                # Save using OmegaConf
                cfg = omegaconf.OmegaConf.create(yaml_data)
                omegaconf.OmegaConf.save(cfg, config_path)
                self.renderer.event(f"Generated config for {game_name} at {config_path}...")
                
        if not config_path:
             raise ValueError(f"Configuration for {game_name} is missing in settings. Cannot start game server.")

        cmd = [
            "python",
            game_server_script,
        ]
        
        cmd.extend(["--config", config_path])
            
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
        self.renderer.event(f"Initializing game servers {games}...")

        game_list = games or self.games

        for game_name in game_list:
            self.launch_game_server(game_name)
            # Avoid long per-game delays; servers should come up in parallel.
            # A tiny stagger helps prevent resource spikes on some systems.
            time.sleep(0.05)

        time.sleep(1.5)
        self.renderer.event("All game servers launched successfully")
    
    def clean_up_game_server(self, game_name: str):
        """
        Terminate a game server process and close any associated resources.

        Important: cleanup must NOT depend on the existence of game_results.json.
        Servers can crash or runs can be interrupted before results are written,
        and we still need to ensure processes and file handles are released.
        """
        proc = self.game_servers_procs.get(game_name)
        if proc is not None:
            try:
                # If still running, try graceful terminate first, then hard kill.
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=5)
                else:
                    # Reap process if already exited.
                    try:
                        proc.wait(timeout=0.2)
                    except Exception:
                        pass
            finally:
                # Always remove the proc entry even if terminate/kill raised.
                self.game_servers_procs.pop(game_name, None)

        f = self.output_files.pop(game_name, None)
        if f is not None:
            try:
                f.close()
            except Exception:
                pass
    
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
            # Iterate over a snapshot in case we stop/cleanup while iterating.
            for game_name, proc in list(self.game_servers_procs.items()):
                if game_name in completed_games:
                    continue

                results_path = os.path.join(GAME_DATA_DIR, game_name, "game_results.json")
                return_code = proc.poll()

                if return_code is not None:
                    # If the process exited cleanly, allow a small grace period for the
                    # results file to appear (avoid false "crash" on delayed writes).
                    if return_code == 0 and not os.path.exists(results_path):
                        grace_deadline = time.time() + 2.0
                        while time.time() < grace_deadline and not os.path.exists(results_path):
                            time.sleep(0.1)

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
    renderer = get_renderer()
    renderer.start(local=True)
    
    # Load settings for standalone execution
    settings = load_hydra_settings("gemini")

    try:
        game_launcher = GameLauncher(renderer, settings=settings)
       
        renderer.event(f"Starting game servers for games: {game_launcher.games}...")
        game_launcher.start_game_servers()
        game_launcher.wait_for_games_to_finish()
    finally:
        game_launcher.force_stop_all_games()
        renderer.stop()
