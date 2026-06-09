import sys
from pathlib import Path
from langchain_core.messages import HumanMessage

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.llm.factory import LLMFactory

def verify_factory_pipeline():
    print("Requesting model instance from LLMFactory...")

    try:
        model = LLMFactory.get_model()
        print("Model loaded...")

        messages = [HumanMessage(content="Respond with: 'Factory interface is operational'")]

        response = model.invoke(messages)
        print(f"Model Output: {response.content}")

    except Exception as e:
        print(f"\nPipeline Test Failed: {e}\n")

if __name__ == "__main__":
    verify_factory_pipeline()