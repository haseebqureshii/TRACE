import json
from pathlib import Path

import pytest

from src.guardrails.dst import LessonState, DialogueStateManager


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_fixture(file_name: str) -> str:
    """Load content from a fixture file."""
    fixture_path = FIXTURES_DIR / file_name
    with open(fixture_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_json_fixture(file_name: str) -> dict:
    """Load JSON content from a fixture file."""
    fixture_path = FIXTURES_DIR / file_name
    with open(fixture_path, "r", encoding="utf-8") as f:
        return json.load(f)


class TestDialogueStateManager:
    def test_initial_state_rendering(self):
        # Load initial state from fixture
        initial_state_data = load_json_fixture("initial_state.json")
        
        # Create LessonState and DialogueStateManager
        state = LessonState(
            target_language=initial_state_data["target_language"],
            current_scenario=initial_state_data["current_scenario"],
            current_sub_goal=initial_state_data["current_sub_goal"],
            turn_count=initial_state_data["turn_count"]
        )
        manager = DialogueStateManager(state)
        
        # Build system context
        actual_prompt = manager.build_system_context()
        
        # Load expected prompt from fixture
        expected_prompt = load_fixture("expected_system_prompt_initial.txt")
        
        # Verify output matches fixture data
        assert actual_prompt == expected_prompt

    def test_state_updates_and_rendering(self):
        # Load initial state from fixture
        initial_state_data = load_json_fixture("initial_state.json")
        
        # Create LessonState and DialogueStateManager
        state = LessonState(
            target_language=initial_state_data["target_language"],
            current_scenario=initial_state_data["current_scenario"],
            current_sub_goal=initial_state_data["current_sub_goal"],
            turn_count=initial_state_data["turn_count"]
        )
        manager = DialogueStateManager(state)
        
        # Load updated sub-goal from fixture
        updated_sub_goal = load_fixture("updated_sub_goal.txt")
        
        # Update sub-goal
        manager.update_sub_goal(updated_sub_goal)
        
        # Increment turn 3 times to reach turn_count = 3
        for _ in range(3):
            manager.increment_turn()
        
        # Build system context
        actual_prompt = manager.build_system_context()
        
        # Load expected prompt from fixture
        expected_prompt = load_fixture("expected_system_prompt_updated.txt")
        
        # Verify output matches fixture data
        assert actual_prompt == expected_prompt

    def test_update_sub_goal_method(self):
        # Load initial state from fixture
        initial_state_data = load_json_fixture("initial_state.json")
        
        # Create LessonState and DialogueStateManager
        state = LessonState(
            target_language=initial_state_data["target_language"],
            current_scenario=initial_state_data["current_scenario"],
            current_sub_goal=initial_state_data["current_sub_goal"],
            turn_count=initial_state_data["turn_count"]
        )
        manager = DialogueStateManager(state)
        
        # Load updated sub-goal from fixture
        updated_sub_goal = load_fixture("updated_sub_goal.txt")
        
        # Update sub-goal
        manager.update_sub_goal(updated_sub_goal)
        
        # Verify state was updated
        assert manager.state.current_sub_goal == updated_sub_goal

    def test_increment_turn_method(self):
        # Load initial state from fixture
        initial_state_data = load_json_fixture("initial_state.json")
        
        # Create LessonState and DialogueStateManager
        state = LessonState(
            target_language=initial_state_data["target_language"],
            current_scenario=initial_state_data["current_scenario"],
            current_sub_goal=initial_state_data["current_sub_goal"],
            turn_count=initial_state_data["turn_count"]
        )
        manager = DialogueStateManager(state)
        
        # Initial turn count should be 0
        assert manager.state.turn_count == 0
        
        # Increment turn once
        manager.increment_turn()
        assert manager.state.turn_count == 1
        
        # Increment turn twice more
        manager.increment_turn()
        manager.increment_turn()
        assert manager.state.turn_count == 3