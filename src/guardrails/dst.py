from pydantic import BaseModel


class LessonState(BaseModel):
    target_language: str
    current_scenario: str
    current_sub_goal: str
    turn_count: int = 0


class DialogueStateManager:
    def __init__(self, state: LessonState):
        self.state = state

    def update_sub_goal(self, new_goal: str) -> None:
        self.state.current_sub_goal = new_goal

    def increment_turn(self) -> None:
        self.state.turn_count += 1

    def build_system_context(self) -> str:
        return (
            f"You are an AI language tutor helping a student learn {self.state.target_language}. \n\n"
            f"Current scenario: {self.state.current_scenario}\n"
            f"Current sub-goal: {self.state.current_sub_goal}\n"
            f"Turn number: {self.state.turn_count}\n\n"
            "Please assist the student with this specific sub-goal within the given scenario."
        )