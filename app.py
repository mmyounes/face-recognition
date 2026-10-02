from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import uuid
import traceback
import database
import face_recognition
from couchbase.options import QueryOptions
from dotenv import load_dotenv

load_dotenv()



app = Flask(__name__)
CORS(app)


face_recognition_scope = os.getenv("FACE_RECOGNITION_SCOPE")
face_recognition_collection = os.getenv("FACE_RECOGNITION_COLLECTION")
celeb_bucket = os.getenv("CELEB_BUCKET")
# Max cosine distance for a face to count as a match (0 = identical, lower is stricter)
verify_distance_threshold = float(os.getenv("VERIFY_DISTANCE_THRESHOLD", "0.2"))
cluster = database.get_cluster()
face_collection = database.get_bucket().scope(face_recognition_scope).collection(face_recognition_collection)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    name = request.form.get("name")
    image_file = request.files.get("image")

    if not name:
        return jsonify({"success": False, "reason": "Name is required"}), 400

    if not image_file:
        return jsonify({"success": False, "reason": "Image is required"}), 400

    
    # Convert image to OpenCV format
    img = face_recognition.convert_image_to_cv2(image_file)

    is_valid, reason = face_recognition.validate_face(img)
    
    if not is_valid:
        print("❌ " + reason)
        return jsonify({
                "success": False,
                "reason": reason
            }), 400

    # ---- EXTRACT EMBEDDING using Facenet512 ----
    emb = face_recognition.generate_embedding(image_file)
    
    # ---- SAVE embedding in CB----
    docId = f"{uuid.uuid4()}_{name}"
    doc = {
        "name": name,
        "embedding": emb
    }
    face_collection.upsert(docId, doc)
    
    return jsonify({
        "success": True,
        "filename": docId
    })


@app.route("/verify")
def verify():
    return render_template("verify.html")

@app.route("/verify", methods=["POST"])
def verify_post():
    image_file = request.files.get("image")
    
    if not image_file:
        return jsonify({"success": False, "reason": "Image is required"}), 400
    
    # Convert image to OpenCV format
    img = face_recognition.convert_image_to_cv2(image_file)

    is_valid, reason = face_recognition.validate_face(img)
    
    if not is_valid:
        print("❌ " + reason)
        return jsonify({
                "success": False,
                "reason": reason
            }), 400
    
    # ---- EXTRACT EMBEDDING using Facenet512 ----
    emb = face_recognition.generate_embedding(image_file)
    
    collection_name = f"`{os.getenv('COUCHBASE_BUCKET')}`.`{face_recognition_scope}`.`{face_recognition_collection}`"
    statement = f"""
        SELECT name, APPROX_VECTOR_DISTANCE(embedding, $vquery, "COSINE") as distance
        FROM {collection_name}
        ORDER BY APPROX_VECTOR_DISTANCE(embedding, $vquery, "COSINE")
        LIMIT 1;
    """
    
    # vector search
    try:
        result = cluster.query(statement, QueryOptions(named_parameters={"vquery": emb}))
        for row in result:
            print(f"Name: {row['name']}, Distance: {row['distance']}")
            distance = row["distance"]
            score = round((1 - distance) * 100, 1)
            if distance < verify_distance_threshold:
                return jsonify({
                    "success": True,
                    "reason": f"Welcome, {row['name']}. Please proceed to the lounge.",
                    "name": row["name"],
                    "distance": distance,
                    "score": score
                })
            return jsonify({
                "success": False,
                "reason": "We couldn't verify your access. Please try again or speak with our lounge concierge for assistance.",
                "distance": distance,
                "score": score
            })
    except Exception as e:
        traceback.print_exc()
        print(f"\033[91mQuery failed: {e}\033[0m")
        return jsonify({
            "success": False,
            "reason": "Internal server error during verification. Contact Admin."
        }), 500

    
    return jsonify({
        "success": False,
        "reason": "We couldn't verify your access. Please try again or speak with our lounge concierge for assistance."
    })
    


@app.route('/celeb')
def celeb():
    return render_template('celeb.html')


@app.route('/celeb', methods=['POST'])
def compare_celeb():
    # 1. Get the image from the request
    image_file = request.files.get('photo')
    
    # Convert image to OpenCV format
    img = face_recognition.convert_image_to_cv2(image_file)

    is_valid, reason = face_recognition.validate_face(img)
    
    if not is_valid:
        print("❌ " + reason)
        return jsonify({"error": reason}), 400

    
    # ---- EXTRACT EMBEDDING using Facenet512 ----
    emb = face_recognition.generate_embedding(image_file)
    statement = f"""
        SELECT name, filename, APPROX_VECTOR_DISTANCE(embedding, $vquery, "COSINE") as distance
        FROM `{celeb_bucket}`
        ORDER BY APPROX_VECTOR_DISTANCE(embedding, $vquery, "COSINE")
        LIMIT 1;
    """
    
    print(statement)
    
    # vector search
    try:
        result = cluster.query(statement, QueryOptions(named_parameters={"vquery": emb}))
        for row in result:
            print(f"Name: {row['name']}, Distance: {row['distance']}")
            return jsonify({
                "celeb_name": row['name'],
                "celeb_image_url": f"/static/images/celeb/{row['filename']}.jpg",
                "distance": row['distance'],
                "score": round((1 - row['distance']) * 100, 1)
            })
    except Exception as e:
        traceback.print_exc()
        print(f"\033[91mQuery failed: {e}\033[0m")
        return jsonify({"error": "Internal server error during verification. Contact Admin."}), 500
    
    return jsonify({"error": "No celebrity match found."}), 500


if __name__ == "__main__":
    app.run(
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true"
    )

