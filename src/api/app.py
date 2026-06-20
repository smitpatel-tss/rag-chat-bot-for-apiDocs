import atexit
import logging
import time
from functools import wraps

from flask import Flask, request, jsonify
from flask_cors import CORS

from src.config.settings import settings
from src.llm.factory import LLMFactory
from src.database.pg_store import PGVectorStore
from src.chat.engine import RAGChatEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

db_string: str = settings.VECTOR_DB_URL

vector_store: PGVectorStore = PGVectorStore(
    connection_string=db_string
)

try:
    vector_store.connect()
except Exception as e:
    logger.error(f"Failed to connect to vector store: {e}")
    raise

atexit.register(vector_store.close)

chat_engine = RAGChatEngine(
    vector_store=vector_store,
    llm=LLMFactory.get_model(),
    embedder=LLMFactory.get_embedding_model(),
    table_name=settings.VECTOR_DB_COLLECTION
)

def log_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        elapsed_time = end_time - start_time
        logger.info(f"-- [{elapsed_time:.4f} s] for endpoint '{func.__name__}'")
        
        return result
    return wrapper


@app.route("/api/chat", methods=["POST"])
@log_execution_time
def chat():
    try:
        data = request.get_json(silent=True)

        if data is None:
            return jsonify({"error": "Invalid or missing JSON payload."}), 400

        user_query = data.get("query")

        if not user_query or not user_query.strip():
            return jsonify({"error": "The 'query' field cannot be empty."}), 400

        result = chat_engine.generate_answer(
            user_query=user_query.strip()
        )

        return jsonify(result), 200

    except ValueError as e:
        logger.warning(f"Validation error: {e}")

        return jsonify({
            "error": str(e)
        }), 400

    except Exception as e:
        logger.error(
            f"Error processing chat request: {e}",
            exc_info=True
        )

        return jsonify({"error": "An internal server error occurred while processing your request."}), 500


@app.route("/health", methods=["GET"])
@log_execution_time
def health():
    return jsonify({
        "status": "healthy"
    }), 200


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )