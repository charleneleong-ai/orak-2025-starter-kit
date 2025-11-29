import asyncio
import os
import time
import json
import backoff

# from evaluation_utils.sessions import Session
from evaluation_utils.game_env import GameEnv
from evaluation_utils.commons import GAME_SERVER_PORTS, GAME_DATA_DIR
from evaluation_utils.game_server_launcher import GameLauncher
from evaluation_utils.renderer import Renderer
from evaluation_utils.sessions import Session

from agents.config import PokemonAgent, TwentyFourtyEightAgent, SuperMarioAgent, StarCraftAgent, StreetFighterAgent

AGENT_MAP = {
    "pokemon_red": PokemonAgent,
    "twenty_fourty_eight": TwentyFourtyEightAgent,
    "super_mario": SuperMarioAgent,
    "star_craft": StarCraftAgent,
    "street_fighter": StreetFighterAgent,
}


class Runner:
    def __init__(
        self,
        session_id: str | None = None,
        local: bool = False,
        renderer: Renderer | None = None,
        games: list[str] | None = None,
    ):
        self.local = local
        self.renderer = renderer

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
            self.renderer.event("Running in LOCAL mode")
            self.mcp_urls = {game: f"http://localhost:{GAME_SERVER_PORTS[game]}/mcp" for game in self.games}
            self.game_launcher = GameLauncher(renderer, games=self.games)
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
            # Only start the subset of games selected for this run
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
                self.renderer.event("Stopping all game servers...")
                self.game_launcher.force_stop_all_games()
            self._cleanup_session_file(all_games_succeeded)

    @backoff.on_exception(backoff.constant, Exception, max_time=3000, max_tries=300, interval=10)
    async def wait_for_client_connect(self, env: GameEnv):
        async with env.client:
            await env.wait_for_ping()

    async def start_game(self, game_name: str):
        self.renderer.set_server_status(game_name, "launching")
        game_display_name = game_name.replace("_", " ").title()

        self.renderer.event(f"{game_display_name}: Initializing agent")

        if self.local:
            mcp_url = self.mcp_urls[game_name]
        else:
            mcp_url = self.session.get()["mcp_urls"][game_name]
        agent = AGENT_MAP[game_name]()
        env = GameEnv(mcp_url)

        self.renderer.event(f"{game_display_name}: Waiting for client to connect...")    
        await self.wait_for_client_connect(env)
        self.renderer.event(f"{game_display_name}: Connected successfully, starting game loop")
        self.renderer.set_server_status(game_name, "running")

        async with env.client:
            self.renderer.start_game_timer(game_name)

            # wait for game using async sleep
            # await asyncio.sleep(60)

            # Prepare per-iteration state logging
            game_data_dir = os.path.join(GAME_DATA_DIR, game_name)
            os.makedirs(game_data_dir, exist_ok=True)
            game_states_path = os.path.join(game_data_dir, "game_states.jsonl")
            states_f = open(game_states_path, "a", encoding="utf-8")

            game_config = await env.get_game_config()
            max_episodes = game_config.get("max_episodes")

            try:
                # Game loop
                iteration = game_config.get("current_step", 0)
                episode = game_config.get("current_episode", 0)
                avg_score = 0
                while episode < max_episodes:
                    iteration += 1
                    obs = await env.load_obs()
                    action = agent.act(obs)
                    result = await env.dispatch_final_action(action)
                    finished = bool(result.get("is_finished"))
                    current_score = result.get("score", 0)
                    avg_score = result.get("avg_score", 0)

                    # Append per-iteration JSONL record
                    try:
                        states_f.write(json.dumps({
                            "iteration": iteration,
                            "obs": obs,
                            "action": action,
                            "result": result,
                            "current_score": current_score
                        }, ensure_ascii=False) + "\n")
                        states_f.flush()
                    except Exception:
                        # Do not fail the game loop on logging issues
                        pass

                    # Update game progress (score and elapsed time)
                    self.renderer.update_game_progress(game_name, current_score)

                    # Log every 10 iterations or on score changes
                    # if iteration % 10 == 0 or (iteration > 1 and current_score != self.scores.get(game_name, 0)):
                    self.renderer.event(f"{game_display_name}: Step {iteration}, Episode: {episode+1}, Score: {current_score}")

                    if finished:
                        episode += 1
                        iteration = 0
                        self.renderer.event(f"{game_display_name}: Game finished after {iteration} steps with final score: {current_score}")
                        if episode < max_episodes:
                            self.renderer.event(f"{game_display_name}: Starting new episode... ({episode+1}/{max_episodes})")
                        else:
                            self.renderer.event(f"{game_display_name}: Max episodes reached. Game finished.")

                self.scores[game_name] = avg_score
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
