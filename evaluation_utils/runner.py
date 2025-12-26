import asyncio
import os
import time
import json
import backoff
from typing import Any

import base64
from io import BytesIO

# from evaluation_utils.sessions import Session
from evaluation_utils.game_env import GameEnv
from evaluation_utils.commons import GAME_SERVER_PORTS, GAME_DATA_DIR
from evaluation_utils.game_server_launcher import GameLauncher
from evaluation_utils.renderer import Renderer
from evaluation_utils.sessions import Session
from config.base import Settings
from config.utils import load_agent_map
from loguru import logger

def pil_image_to_base64(image_object):
    """
    Converts a PIL Image object to a base64 string.
    """
    if image_object is None:
        return None
    # Create a buffer in memory
    buffered = BytesIO()
    # Save the image object to the buffer in JPEG format
    # Note: If the original image has an RGBA mode, convert it to RGB first 
    # if saving as JPEG, as JPEG does not support the alpha channel.
    if image_object.mode == 'RGBA':
        image_object = image_object.convert('RGB')
        
    image_object.save(buffered, format="JPEG")
    # Get the value from the buffer and encode it in base64
    img_bytes = buffered.getvalue()
    encoded_string = base64.b64encode(img_bytes).decode('utf-8')
    return encoded_string


class Runner:
    def __init__(
        self,
        session_id: str | None = None,
        local: bool = False,
        renderer: Renderer | None = None,
        games: list[str] | None = None,
        grpc_host: str = "localhost",
        grpc_ports: dict[str, int] | None = None,
        manage_local_game_servers: bool = True,
        settings: Settings | None = None,
    ):
        self.local = local
        self.renderer = renderer
        self.manage_local_game_servers = manage_local_game_servers
        self.settings = settings
        self.agent_map = load_agent_map(self.settings)

        # Determine which games to run
        if self.local:
            # Local mode: allow selecting a subset of games
            if games is None:
                self.games = list(GAME_SERVER_PORTS.keys())
            else:
                invalid = [g for g in games if g not in GAME_SERVER_PORTS]
                if invalid:
                    raise ValueError(f"Unknown game(s) requested: {', '.join(invalid)}")
                self.games = games
        else:
            # Remote mode: always run all games; per-game selection is only supported locally
            self.games = list(GAME_SERVER_PORTS.keys())

        self.scores = {game: 0 for game in self.games}
        self.session_file = None
        self._session_provided_by_user = session_id is not None
        self._should_delete_session_file = False

        if self.local:
            mode_suffix = " (managing servers)" if self.manage_local_game_servers else " (attach mode)"
            self.renderer.event(f"Running in LOCAL mode{mode_suffix}")

            ports = grpc_ports or GAME_SERVER_PORTS
            missing = [g for g in self.games if g not in ports]
            if missing:
                raise ValueError(f"Missing port(s) for game(s): {', '.join(missing)}")

            self.grpc_addresses = {game: f"{grpc_host}:{ports[game]}" for game in self.games}
            self.game_launcher = GameLauncher(renderer, settings=self.settings) if self.manage_local_game_servers else None
        else:
            self.renderer.event("Running in REMOTE mode")
            self.session = Session(session_id=session_id, renderer=self.renderer)
            self._should_delete_session_file = not self._session_provided_by_user

            session_dir = os.path.join(os.getcwd(), ".aicrowd")
            session_file = os.path.join(session_dir, "session_id")
            self.session_file = session_file
            os.makedirs(session_dir, exist_ok=True)

            # If no session-id provided, check persisted session file
            if self.session.session_id is None and os.path.exists(session_file):
                try:
                    with open(session_file, "r", encoding="utf-8") as f:
                        previous_session_id = f.read().strip()
                except Exception:
                    previous_session_id = ""

                if previous_session_id:
                    if self.renderer.confirm(
                        f"Found previous session [bold]{previous_session_id}[/bold]. Continue it?",
                        default=True
                    ):
                        self.renderer.event(f"Continuing previous session: {previous_session_id}")
                        self.session.session_id = previous_session_id
                    else:
                        # Stop previous session before creating a new one
                        self.renderer.event(f"Stopping previous session: {previous_session_id}")
                        try:
                            temp = Session(previous_session_id, renderer=self.renderer)
                            temp.stop()
                        except Exception:
                            pass

            # Create a new session if we still don't have one
            if self.session.session_id is None:
                self.renderer.event("Creating new session...")
                self.session.create()
                self.renderer.event(f"Session created: {self.session.session_id}")
                try:
                    with open(session_file, "w", encoding="utf-8") as f:
                        f.write(self.session.session_id)
                except Exception:
                    pass
            else:
                # Persist provided/continued session id
                try:
                    with open(session_file, "w", encoding="utf-8") as f:
                        f.write(self.session.session_id)
                except Exception:
                    pass
            self.renderer.event(f"Waiting for session {self.session.session_id} to start...")
            self.session.wait_for_start()
            self.renderer.event(f"Session {self.session.session_id} is ready")
    
    async def evaluate_all_games(self):
        if self.local:
            # Only start the subset of games selected for this run (unless attaching)
            if self.manage_local_game_servers and self.game_launcher is not None:
                self.game_launcher.start_game_servers(self.games)
            # self.renderer.event("Waiting for game servers to be ready...")

        self.renderer.event(f"Starting parallel evaluation of {len(self.scores)} games")

        all_games_succeeded = True
        try:
            # Evaluate all selected games in parallel
            tasks = [asyncio.create_task(self.start_game(game_name)) for game_name in self.games]
            await asyncio.gather(*tasks)

            self.renderer.event("All games completed successfully")
        except Exception:
            all_games_succeeded = False
            raise
        finally:
            if self.local:
                if self.manage_local_game_servers and self.game_launcher is not None:
                    self.renderer.event("Stopping all game servers...")
                    self.game_launcher.force_stop_all_games()
            self._cleanup_session_file(all_games_succeeded)

    @backoff.on_exception(backoff.constant, Exception, max_time=3000, max_tries=300, interval=10)
    def wait_for_client_connect(self, env: GameEnv):
        """Establish connection and register session with the game server."""
        env.connect()

    async def _wait_for_client_connect_async(self, env: GameEnv) -> None:
        """
        Async wrapper for session registration.

        GameEnv.connect() and its retry/backoff logic are synchronous and can block;
        running it in a worker thread allows other games to progress concurrently.
        """
        await asyncio.to_thread(self.wait_for_client_connect, env)

    async def _call_in_thread(self, fn, *args, **kwargs) -> Any:
        """Run a blocking callable in a worker thread and yield control to the event loop."""
        return await asyncio.to_thread(fn, *args, **kwargs)

    async def start_game(self, game_name: str):
        self.renderer.set_server_status(game_name, "launching")
        game_display_name = game_name.replace("_", " ").title()

        self.renderer.event(f"{game_display_name}: Initializing agent")

        if self.local:
            grpc_address = self.grpc_addresses[game_name]
        else:
            grpc_address = self.session.get()["grpc_addresses"][game_name]
        agent = self.agent_map[game_name]
        logger.info(f"Starting evaluation for game {game_name} using gRPC address {grpc_address}")
        env = GameEnv(grpc_address)

        self.renderer.event(f"{game_display_name}: Waiting for client to connect...")
        await self._wait_for_client_connect_async(env)
        self.renderer.event(f"{game_display_name}: Connected successfully, starting game loop")
        self.renderer.set_server_status(game_name, "running")

        try:
            self.renderer.start_game_timer(game_name)

            # Prepare per-iteration state logging
            game_data_dir = os.path.join(GAME_DATA_DIR, game_name)
            os.makedirs(game_data_dir, exist_ok=True)

            # Configure agent logging and save model declaration
            if hasattr(agent, "set_log_dir"):
                agent.set_log_dir(game_data_dir)
            
            if hasattr(agent, "get_model_declaration"):
                try:
                    model_decl = agent.get_model_declaration()
                    with open(os.path.join(game_data_dir, "model_declaration.json"), "w") as f:
                        json.dump(model_decl, f, indent=2)
                except Exception as e:
                    logger.error(f"Failed to save model declaration: {e}")

            game_states_path = os.path.join(game_data_dir, "game_states.jsonl")
            states_f = open(game_states_path, "a", encoding="utf-8")

            game_config = await self._call_in_thread(env.get_game_config)
            
            # Prefer max_episodes from settings if available
            if self.settings:
                game_settings = getattr(self.settings, game_name, None)
                if game_settings and hasattr(game_settings, "env") and hasattr(game_settings.env, "max_episodes"):
                    max_episodes = game_settings.env.max_episodes
                else:
                    max_episodes = game_config.get("max_episodes")
            else:
                max_episodes = game_config.get("max_episodes")

            try:
                # Game loop
                iteration = game_config.get("current_step", 0)
                episode = game_config.get("current_episode", 0)
                avg_score = 0
                while episode < max_episodes:
                    iteration += 1
                    obs = await self._call_in_thread(env.load_obs)
                    action = await self._call_in_thread(agent.act, obs)
                    result = await self._call_in_thread(env.dispatch_final_action, action)
                    finished = bool(result.get("is_finished"))
                    current_score = result.get("score", 0)
                    avg_score = result.get("avg_score", 0)

                    # Append per-iteration JSONL record
                    try:
                        obs["obs_image"] = pil_image_to_base64(obs["obs_image"])
                        result.pop("obs")
                        states_f.write(json.dumps({
                            "iteration": iteration,
                            "obs": obs,
                            "action": action,
                            "result": result,
                            "current_score": current_score
                        }, ensure_ascii=False) + "\n")
                        states_f.flush()
                    except Exception as e:
                        # Do not fail the game loop on logging issues
                        import traceback
                        traceback.format_exc()
                        self.renderer.event(f"{game_display_name}: Error writing game states: {e}, traceback: {traceback.format_exc()}, obs: {obs.keys()}, result: {result.keys()}")
                        pass

                    # Update game progress (score and elapsed time)
                    self.renderer.update_game_progress(game_name, current_score)

                    # Log every 10 iterations or on score changes
                    # if iteration % 10 == 0 or (iteration > 1 and current_score != self.scores.get(game_name, 0)):
                    self.renderer.event(f"{game_display_name}: Step {iteration}, Episode: {episode+1}, Score: {current_score}")

                    if finished:
                        steps_this_episode = iteration
                        
                        # Record episode stats
                        if hasattr(agent, "record_episode_end"):
                            seed = "unknown"
                            # Try to find seed in game info
                            final_game_info = result.get("obs", {}).get("game_info", {})
                            current_game_info = obs.get("game_info", {})
                            
                            if "seed" in final_game_info:
                                seed = final_game_info["seed"]
                            elif "seed" in current_game_info:
                                seed = current_game_info["seed"]
                                
                            agent.record_episode_end(episode + 1, game_name, seed, current_score)

                        episode += 1
                        iteration = 0
                        self.renderer.event(
                            f"{game_display_name}: Game finished after {steps_this_episode} steps with final score: {current_score}"
                        )
                        if episode < max_episodes:
                            self.renderer.event(f"{game_display_name}: Starting new episode... ({episode+1}/{max_episodes})")
                        else:
                            self.renderer.event(f"{game_display_name}: Max episodes reached. Game finished.")

                self.scores[game_name] = avg_score

                # Save evaluation summary
                if hasattr(agent, "get_evaluation_summary"):
                    try:
                        summary = agent.get_evaluation_summary(episode)
                        with open(os.path.join(game_data_dir, "evaluation_summary.json"), "w") as f:
                            json.dump(summary, f, indent=2)
                    except Exception as e:
                        logger.error(f"Failed to save evaluation summary: {e}")

                # Mark game as completed
                self.renderer.complete_game(game_name, avg_score)
            except Exception as e:
                self.renderer.event(f"{game_display_name}: Error: {e}")
                raise
            finally:
                try:
                    states_f.close()
                except Exception:
                    pass
        finally:
            await self._call_in_thread(env.close)

    def _cleanup_session_file(self, all_games_succeeded: bool):
        if (
            self.local
            or not all_games_succeeded
            or not self._should_delete_session_file
            or not self.session_file
        ):
            return

        try:
            os.remove(self.session_file)
            if self.renderer:
                self.renderer.event("Session completed. Cleaning up saved session id.")
        except FileNotFoundError:
            # Already removed or never created; ignore.
            pass
        except OSError as exc:
            if self.renderer:
                self.renderer.event(f"Warning: Failed to delete session file: {exc}")
