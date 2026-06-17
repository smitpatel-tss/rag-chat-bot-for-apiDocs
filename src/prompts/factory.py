from src.prompts.registry import PROMPT_REGISTRY


class PromptFactory:

    @staticmethod
    def get(name: str, version: str = "v1"):
        if name not in PROMPT_REGISTRY:
            raise ValueError(f"Prompt not found: {name}")

        versions = PROMPT_REGISTRY[name]

        if version not in versions:
            raise ValueError(f"Version not found: {name}:{version}")

        return versions[version]