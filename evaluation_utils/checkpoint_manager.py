"""
General checkpoint management system for agents and game states.
Supports saving/loading agent state, game progress, and training history.
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from evaluation_utils.checkpointable import Checkpointable


class CheckpointManager:
    """Manages checkpoints for agents and game states."""

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def get_checkpoint_path(
        self, agent_name: str, game_name: str | None = None, checkpoint_id: str | None = None
    ) -> Path:
        """
        Get path for a checkpoint file.

        Args:
            game_name: Name of the game
            agent_name: Name of the agent
            checkpoint_id: Optional identifier (defaults to timestamp)

        Returns:
            Path to checkpoint file
        """
        if checkpoint_id is None:
            checkpoint_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        game_dir = self.checkpoint_dir / game_name if game_name else self.checkpoint_dir
        game_dir.mkdir(parents=True, exist_ok=True)

        return game_dir / f"{agent_name}_{checkpoint_id}.pkl"

    def get_latest_checkpoint(self, agent_name: str, game_name: str | None = None) -> Path | None:
        """
        Find the most recent checkpoint for a game/agent pair.

        Args:
            game_name: Name of the game
            agent_name: Name of the agent

        Returns:
            Path to latest checkpoint or None if no checkpoints exist
        """
        game_dir = self.checkpoint_dir / game_name if game_name else self.checkpoint_dir
        if not game_dir.exists():
            return None

        checkpoints = list(game_dir.glob(f"{agent_name}_*.pkl"))
        if not checkpoints:
            return None

        # Sort by modification time, most recent first
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return checkpoints[0]

    def save_checkpoint(
        self,
        agent_name: str,
        agent_state: dict[str, Any],
        game_state: dict[str, Any],
        game_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        checkpoint_id: str | None = None,
    ) -> Path:
        """
        Save a checkpoint with agent state, game state, and metadata.

        Args:
            game_name: Name of the game
            agent_name: Name of the agent
            agent_state: Agent's internal state (returned by agent.get_state())
            game_state: Current game state (score, episode, etc.)
            metadata: Additional metadata to save
            checkpoint_id: Optional identifier

        Returns:
            Path where checkpoint was saved
        """
        checkpoint_path = self.get_checkpoint_path(agent_name, game_name, checkpoint_id)

        checkpoint = {
            "game_name": game_name,
            "agent_name": agent_name,
            "timestamp": datetime.now().isoformat(),
            "agent_state": agent_state,
            "game_state": game_state,
            "metadata": metadata or {},
        }

        try:
            with open(checkpoint_path, "wb") as f:
                pickle.dump(checkpoint, f)

            # Also save a human-readable JSON summary
            summary_path = checkpoint_path.with_suffix(".json")
            summary = {
                "game_name": game_name,
                "agent_name": agent_name,
                "timestamp": checkpoint["timestamp"],
                "game_state": game_state,
                "metadata": metadata or {},
            }
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)

            logger.info(f"Checkpoint saved: {checkpoint_path}")
            return checkpoint_path

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise

    def save_agent_checkpoint(
        self,
        agent: Checkpointable,
        game_state: dict[str, Any],
        game_name: str | None = None,
        checkpoint_id: str | None = None,
    ) -> Path:
        """
        Save checkpoint directly from a Checkpointable agent.

        Args:
            agent: Agent implementing Checkpointable protocol
            game_name: Name of the game
            game_state: Current game state (score, episode, etc.)
            checkpoint_id: Optional identifier

        Returns:
            Path where checkpoint was saved
        """
        # Verify agent has required methods (runtime check)
        if not isinstance(agent, Checkpointable):
            raise TypeError(
                "Agent must implement Checkpointable protocol (have get_state, load_state, get_checkpoint_metadata methods)"
            )

        agent_name = agent.__class__.__name__
        agent_state = agent.get_state()
        metadata = agent.get_checkpoint_metadata()
        # If no explicit checkpoint_id provided, generate one with step count
        if checkpoint_id is None:
            total_steps = 0
            if game_state and "total_steps" in game_state:
                total_steps = game_state["total_steps"]
            elif metadata and "total_steps" in metadata:
                total_steps = metadata["total_steps"]

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_id = f"step_{total_steps}_{timestamp}"

        return self.save_checkpoint(
            game_name=game_name,
            agent_name=agent_name,
            agent_state=agent_state,
            game_state=game_state,
            metadata=metadata,
            checkpoint_id=checkpoint_id,
        )

    def load_checkpoint(self, checkpoint_path: Path) -> dict[str, Any]:
        """
        Load a checkpoint from file.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Checkpoint dictionary containing agent_state, game_state, and metadata
        """
        try:
            with open(checkpoint_path, "rb") as f:
                checkpoint = pickle.load(f)

            logger.info(f"Checkpoint loaded: {checkpoint_path}")
            logger.info(f"  Game: {checkpoint['game_name']}, Agent: {checkpoint['agent_name']}")
            logger.info(f"  Timestamp: {checkpoint['timestamp']}")

            return checkpoint

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

    def load_agent_checkpoint(
        self,
        agent: Checkpointable,
        checkpoint_path: Path,
    ) -> dict[str, Any]:
        """
        Load checkpoint directly into a Checkpointable agent.

        Args:
            agent: Agent implementing Checkpointable protocol
            checkpoint_path: Path to checkpoint file

        Returns:
            Full checkpoint data (including game_state, metadata)
        """
        # Verify agent has required methods (runtime check)
        if not isinstance(agent, Checkpointable):
            raise TypeError(
                "Agent must implement Checkpointable protocol (have get_state, load_state, get_checkpoint_metadata methods)"
            )

        checkpoint = self.load_checkpoint(checkpoint_path)
        agent.load_state(checkpoint["agent_state"])

        logger.info(f"Agent state restored from checkpoint: {checkpoint_path}")

        return checkpoint

    def load_latest_agent_checkpoint(
        self,
        agent: Checkpointable,
        game_name: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Load the latest checkpoint for an agent/game into the agent.

        Args:
            agent: Agent implementing Checkpointable protocol
            game_name: Name of the game

        Returns:
            Full checkpoint data if found, None otherwise
        """
        agent_name = agent.__class__.__name__
        latest = self.get_latest_checkpoint(agent_name, game_name)

        if latest is None:
            logger.info(f"No checkpoint found for {game_name}/{agent_name}")
            return None

        return self.load_agent_checkpoint(agent, latest)

    def list_checkpoints(
        self, game_name: str | None = None, agent_name: str | None = None
    ) -> list[dict[str, Any]]:
        """
        List all available checkpoints, optionally filtered by game/agent.

        Args:
            game_name: Optional game name filter
            agent_name: Optional agent name filter

        Returns:
            List of checkpoint info dictionaries
        """
        checkpoints = []

        search_dir = self.checkpoint_dir / game_name if game_name else self.checkpoint_dir

        if not search_dir.exists():
            return []

        pattern = f"{agent_name}_*.json" if agent_name else "*.json"

        for json_file in search_dir.rglob(pattern):
            try:
                with open(json_file) as f:
                    summary = json.load(f)
                    summary["checkpoint_path"] = json_file.with_suffix(".pkl")
                    checkpoints.append(summary)
            except Exception as e:
                logger.warning(f"Failed to read checkpoint summary {json_file}: {e}")

        # Sort by timestamp, most recent first
        checkpoints.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        return checkpoints

    def delete_checkpoint(self, checkpoint_path: Path):
        """Delete a checkpoint and its summary."""
        try:
            checkpoint_path.unlink(missing_ok=True)
            checkpoint_path.with_suffix(".json").unlink(missing_ok=True)
            logger.info(f"Checkpoint deleted: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to delete checkpoint: {e}")
            raise

    def cleanup_old_checkpoints(
        self, agent_name: str, game_name: str | None = None, keep_last_n: int = 5
    ):
        """
        Keep only the N most recent checkpoints for a game/agent pair.

        Args:
            game_name: Name of the game
            agent_name: Name of the agent
            keep_last_n: Number of recent checkpoints to keep
        """
        game_dir = self.checkpoint_dir / game_name if game_name else self.checkpoint_dir
        if not game_dir.exists():
            return

        checkpoints = list(game_dir.glob(f"{agent_name}_*.pkl"))
        if len(checkpoints) <= keep_last_n:
            return

        # Sort by modification time, oldest first
        checkpoints.sort(key=lambda p: p.stat().st_mtime)

        # Delete oldest checkpoints
        for checkpoint in checkpoints[:-keep_last_n]:
            self.delete_checkpoint(checkpoint)

        logger.info(f"Cleaned up old checkpoints, kept {keep_last_n} most recent")
