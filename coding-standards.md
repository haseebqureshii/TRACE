# General

* Understand the task before making changes.
* Ask for clarification if requirements are ambiguous.
* Prefer simple, maintainable solutions over unnecessary complexity.
* Follow the project's existing architecture and coding style.

# Code Quality

* Write modular, reusable, and readable code.
* Use descriptive names for variables, functions, and classes.
* Keep functions and classes focused on a single responsibility.
* Avoid duplication (DRY) and unnecessary dependencies.
* Limit changes to the requested scope.

# Reliability

* Handle errors gracefully.
* Consider edge cases and validate inputs where appropriate.
* Preserve backward compatibility unless instructed otherwise.

# Verification

* Verify assumptions by inspecting the codebase or using available MCP tools instead of guessing.
* When using third-party libraries, consult their documentation or source through available MCP tools whenever possible.
* Run or recommend appropriate tests before considering the task complete.

# Communication

* Explain important design decisions and trade-offs briefly.
* State assumptions explicitly.
* If uncertain, say so rather than inventing an answer.

# Final Review

* Review the implementation for correctness, consistency, maintainability, and completeness before presenting it.

# AI Research & Guardrail Integrity

* Never modify test assertions, expected fixtures, or evaluation criteria to force a failing test to pass.
* Maintain strict separation between Input Rails (evaluating user prompts before generation) and Output Rails (evaluating LLM outputs after generation).
* For Output Rail unit tests, use static or mocked off-topic response strings rather than expecting the LLM to fail on purpose.
