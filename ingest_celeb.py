import csv
from datetime import timedelta
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions
import os
from deepface import DeepFace
from dotenv import load_dotenv

load_dotenv()

user = os.getenv("COUCHBASE_USER")
password = os.getenv("COUCHBASE_PASSWORD")
conn_string = os.getenv("COUCHBASE_HOST")

auth = PasswordAuthenticator(user, password)
options = ClusterOptions(auth)
options.apply_profile('wan_development')
cluster = Cluster(conn_string, options)
cluster.wait_until_ready(timedelta(seconds=5))
bucket = cluster.bucket(os.getenv("CELEB_BUCKET"))
collection = bucket.default_collection()

base_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(base_dir, "Labels.csv"), newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        collection.upsert(row["filename"], row)


dataset_path = os.path.join(base_dir, "static", "images", "celeb")
celeb = os.listdir(dataset_path)
count = 1

for person in celeb:
    person_path = os.path.join(dataset_path, person)
    emb = DeepFace.represent(img_path=person_path, model_name="Facenet512", enforce_detection=False)[0]["embedding"]
    docId = person.split(".")[0]
    print(docId)
    doc = collection.get(docId).content_as[dict]
    doc["embedding"] = emb
    collection.upsert(docId, doc)
    print (f"Processed {count}/{len(celeb)}: {person}")
    count += 1
    
print(count-1, "celebrities processed and embeddings stored in Couchbase ✅")