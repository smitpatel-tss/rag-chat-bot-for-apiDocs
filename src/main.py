from flask import Flask, request, jsonify
from pydantic import ValidationError
from api.schemas.chat import ChatRequest, ChatResponse, Citation
from config.settings import settings

app = Flask(__name__)

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "environment": settings.APP_ENV}), 200

@app.route("/api/chat", methods=["POST"])
def chat():
   
    json_data = request.get_json(silent=True)
    if json_data is None:
        return jsonify({"error": "Missing JSON body"}), 400

    try:
        chat_req = ChatRequest(**json_data)
    except ValidationError as e:
        return jsonify({"error": "Validation Error", "details": e.errors()}), 422

    app.logger.info(f"Received query: '{chat_req.query}' from page: '{chat_req.current_page_path}'")
    
    mock_response = ChatResponse(
        answer=f"You asked: '{chat_req.query}' while looking at {chat_req.current_page_path}. This is a placeholder response.",
        sources=[
            Citation(title="Users API Guide", url=f"{chat_req.current_page_path}#overview")
        ]
    )

    return jsonify(mock_response.model_dump()), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=(settings.APP_ENV == "development"))