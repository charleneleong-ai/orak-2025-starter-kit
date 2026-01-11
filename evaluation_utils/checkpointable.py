"""
Protocol for checkpointable agents.

Using Protocol instead of ABC to avoid multiple inheritance issues
with weave.Model and Pydantic.
"""

from typing import Protocol, Any, runtime_checkable


@runtime_checkable
class Checkpointable(Protocol):
    """
    Protocol for agents that support checkpointing.
    
    Any class implementing these three methods is considered Checkpointable,
    regardless of inheritance. This allows type checking without requiring
    inheritance from a base class.
    
    Usage:
        class MyAgent:
            def get_state(self) -> dict[str, Any]:
                return {"my_data": self.data}
            
            def load_state(self, state: dict[str, Any]):
                self.data = state["my_data"]
            
            def get_checkpoint_metadata(self) -> dict[str, Any]:
                return {"version": "1.0"}
        
        # Type checker will accept MyAgent as Checkpointable
        agent: Checkpointable = MyAgent()
    """
    
    def get_state(self) -> dict[str, Any]:
        """
        Get the current state of the agent for checkpointing.
        
        Should return a dictionary containing all state needed to restore
        the agent, including:
        - Memory/learning state
        - Statistics and metrics
        - Configuration
        - Any accumulated knowledge
        
        Returns:
            Dictionary containing all agent state.
        """
        ...
    
    def load_state(self, state: dict[str, Any]) -> None:
        """
        Load agent state from a checkpoint.
        
        Should restore the agent to the exact state it was in when
        get_state() was called.
        
        Args:
            state: State dictionary previously returned by get_state()
        """
        ...
    
    def get_checkpoint_metadata(self) -> dict[str, Any]:
        """
        Get metadata to save with checkpoint (for summaries/inspection).
        
        Should return a dictionary with human-readable summary information
        about the agent and its training progress. This is used for the
        JSON checkpoint summaries.
        
        Returns:
            Dictionary with metadata (e.g., episode count, scores, etc.)
        """
        ...
