# Face Recognition with Couchbase Vector Search

A small Flask web app that shows face recognition built on **[DeepFace](https://github.com/serengil/deepface)** (RetinaFace detector + Facenet512 embeddings) and **Couchbase Hyperscale Vector Indexes** queried with SQL++ (`APPROX_VECTOR_DISTANCE`).

It covers two use cases:

1. **Register & verify a face.** Register a person by name with a webcam photo or an uploaded image. Later, verify a new photo against everyone you've registered. The app returns the closest match and a **match score**, and grants or denies access based on a distance threshold. The demo UI is themed as an airport lounge check-in ("Skyline Lounge").
2. **Celebrity look-alike.** Take a webcam photo and find the celebrity in a preloaded dataset of about 150 public figures whose face is most similar to yours.

---

## How it works

```
Browser (webcam / upload)
        │  image
        ▼
Flask (app.py)
        │
        ├─ face_recognition.validate_face()   ← RetinaFace: exactly one face, frontal,
        │                                       not cropped, large enough, not tilted
        ├─ face_recognition.generate_embedding() ← Facenet512 → 512-dim vector
        │
        ▼
Couchbase
   ├─ register:  upsert {name, embedding}
   └─ verify / look-alike:
        SELECT name, APPROX_VECTOR_DISTANCE(embedding, $vquery, "COSINE") AS distance
        FROM <keyspace> ORDER BY APPROX_VECTOR_DISTANCE(...) LIMIT 1
        (served by a Hyperscale Vector Index on `embedding`)
```

- **Distance** is the cosine distance between the two embeddings: `0` means identical, and larger means less similar.
- **Score** is `(1 - distance) × 100`, shown as a percentage.
- When you verify, a match is accepted when `distance < VERIFY_DISTANCE_THRESHOLD` (default `0.2`).

### Project layout

| Path | Purpose |
|---|---|
| `app.py` | Flask app and all HTTP routes |
| `face_recognition.py` | Face quality validation and Facenet512 embedding generation |
| `database.py` | Couchbase connection (singleton cluster and bucket) |
| `ingest_celeb.py` | One-off script that loads the celebrity dataset and embeddings into Couchbase |
| `Labels.csv` | Celebrity names and image file names (`id,name,empty,filename`) |
| `static/images/celeb/` | Celebrity reference images, served back to the browser as the match photo |
| `templates/index.html` | Register page |
| `templates/verify.html` | Verify page |
| `templates/celeb.html` | Celebrity look-alike page |
| `.env.example` | Template for every config value the app needs |

---

## Prerequisites

- **Python 3.11 or 3.12.** TensorFlow, which deepface pulls in, may not yet support the newest Python release.
- **A Couchbase cluster that supports Hyperscale Vector Indexes.** This means Couchbase Server **8.0+**, or a **Capella** cluster running 8.0+. The Data, Query and Index services must be enabled.
- **A webcam and a modern browser.** Camera access only works on `http://localhost` or over HTTPS.
- **About 1 GB of disk** for TensorFlow and the model weights. DeepFace downloads the Facenet512 and RetinaFace weights to `~/.deepface` the first time they're used.

---

## Setup

### 1. Clone and install

```bash
git clone <your-repo-url> face-recognition
cd face-recognition

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your values:

| Variable | Description | Example |
|---|---|---|
| `COUCHBASE_HOST` | Cluster connection string | `couchbases://cb.xxxx.cloud.couchbase.com` (Capella) or `couchbase://localhost` |
| `COUCHBASE_USER` | Database user. Needs read/write and query access to both buckets. | `app_user` |
| `COUCHBASE_PASSWORD` | Password for that user | |
| `COUCHBASE_BUCKET` | Bucket for registered faces | `face_recognition` |
| `FACE_RECOGNITION_SCOPE` | Scope inside that bucket | `faces` |
| `FACE_RECOGNITION_COLLECTION` | Collection inside that scope | `users` |
| `VERIFY_DISTANCE_THRESHOLD` | Max cosine distance accepted as a verified match | `0.2` |
| `CELEB_BUCKET` | Bucket for the celebrity dataset (uses `_default` collection) | `celeb` |
| `PORT` | Port Flask listens on | `5000` |
| `FLASK_DEBUG` | `true` to enable Flask debug/reload | `false` |

> `.env` is git-ignored. Never commit real credentials.

### 3. Prepare Couchbase

Create these in the Couchbase UI or Capella console:

1. A bucket named `COUCHBASE_BUCKET`, with a scope `FACE_RECOGNITION_SCOPE` and a collection `FACE_RECOGNITION_COLLECTION` inside it.
2. A bucket named `CELEB_BUCKET`. Only its `_default` scope and collection are used.
3. A database user with access to both buckets. On Capella, also add your IP to the cluster's **allowed IP list**.

Optionally, create primary indexes so you can browse the documents in Query Workbench. The commands below use the default names from `.env.example`; swap in yours:

```sql
CREATE PRIMARY INDEX IF NOT EXISTS ON `face_recognition`.`faces`.`users`;
CREATE PRIMARY INDEX IF NOT EXISTS ON `celeb`;
```

> Don't create the vector indexes yet. A Hyperscale Vector Index trains its centroids on the vectors already in the collection, so it has to be created **after** data is loaded. See [step 6](#6-create-the-hyperscale-vector-indexes).

### 4. Load the celebrity dataset

This step is only needed for the look-alike use case. Run it once:

```bash
python ingest_celeb.py
```

The script:

1. Upserts every row of `Labels.csv` into `CELEB_BUCKET`, using the `filename` column as the document key.
2. Generates a Facenet512 embedding for each image in `static/images/celeb/` and adds it to the matching document as `embedding`.

It prints progress for each image. The first run is slower because it downloads the model weights.

### 5. Run the app

```bash
python app.py
```

Open <http://localhost:5000>. If you changed `PORT`, use that port instead.

Before you can verify anyone, register at least one face at <http://localhost:5000/>. See [Use case 1](#use-case-1-register-a-face-then-verify-it). Registration doesn't need a vector index. Verification does.

### 6. Create the Hyperscale Vector Indexes

Both lookups (`/verify` and `/celeb`) run `ORDER BY APPROX_VECTOR_DISTANCE(...) LIMIT 1`, which is served by a **Hyperscale Vector Index** on the `embedding` field. Create one index per keyspace, **after** its data exists:

- **Celebrity index:** create it once `ingest_celeb.py` has finished.
- **Face index:** create it after you've registered your first faces.

Run these in Query Workbench, or in the Capella **Data Tools → Query** tab. They use the default names from `.env.example`, so swap in yours:

```sql
-- Use case 1: registered faces
CREATE VECTOR INDEX `idx_faces_embedding_hyperscale`
  ON `face_recognition`.`faces`.`users`(`embedding` VECTOR)
  WITH { "dimension": 512, "similarity": "COSINE", "description": "IVF,SQ8" };

-- Use case 2: celebrity look-alike
CREATE VECTOR INDEX `idx_celeb_embedding_hyperscale`
  ON `celeb`.`_default`.`_default`(`embedding` VECTOR)
  WITH { "dimension": 512, "similarity": "COSINE", "description": "IVF,SQ8" };
```

| Setting | Why |
|---|---|
| `dimension: 512` | Facenet512 produces 512-dimensional embeddings. It must match exactly. |
| `similarity: "COSINE"` | Must match the metric used in the app's `APPROX_VECTOR_DISTANCE(..., "COSINE")` queries |
| `description: "IVF,SQ8"` | IVF partitioning with the default centroid count (about 1 per 1,000 vectors), plus 8-bit scalar quantization |

Check that the indexes have finished building (state `online`):

```sql
SELECT name, keyspace_id, state FROM system:indexes WHERE name LIKE "idx_%_embedding_hyperscale";
```

> **Retraining:** centroids are learned when the index is built. Newly registered faces are still indexed and searchable afterwards. But if the data grows a lot (for example from 10 faces to 100,000), drop and recreate the face index so the centroids reflect the new data.

---

## Use case 1: Register a face, then verify it

### Register: `http://localhost:5000/`

1. Enter the person's **full name**.
2. Do one of these:
   - Click **Take Photo** to capture a webcam frame.
   - Choose an image file and click **Upload Image**.
3. The server checks that the photo is usable. It rejects the photo if:
   - there isn't exactly one face;
   - detection confidence is below 0.95;
   - the face touches the image edge;
   - the face covers less than 10% of the image;
   - the head is tilted.
4. If the photo passes, the server stores the 512-dim embedding (no image) in `COUCHBASE_BUCKET.FACE_RECOGNITION_SCOPE.FACE_RECOGNITION_COLLECTION` as:

   ```json
   { "name": "Jane Doe", "embedding": [0.0123, -0.0456, ...] }
   ```

   The document key is `<uuid>_<name>`. You can register the same person several times with different photos to make matching more robust.

### Verify: `http://localhost:5000/verify`

1. Look into the camera, keeping your face centered and frontal, and click **Verify Access**.
2. The server applies the same validation, generates an embedding, and finds the **single closest** registered face.
3. The result screen shows:
   - **Access Granted**: "Welcome, *name*", plus the **match score** and cosine distance, when `distance < VERIFY_DISTANCE_THRESHOLD`.
   - **Access Denied**: the closest score is shown, but it isn't close enough.
   - A validation message, such as "Face is tilted or not frontal", if the photo was rejected.

**Tuning:** lower `VERIFY_DISTANCE_THRESHOLD` (for example `0.15`) to make it stricter and reduce false accepts. Raise it (for example `0.3`) to make it more lenient and reduce false rejects. Watch the server log: every verification prints the closest name and distance, which helps you choose a value.

## Use case 2: Celebrity look-alike

### `http://localhost:5000/celeb`

1. Complete [step 4](#4-load-the-celebrity-dataset) and create the celebrity index in [step 6](#6-create-the-hyperscale-vector-indexes) first.
2. Face the camera and click **ANALYZE MY FACE**.
3. The server validates your face, generates an embedding, and returns the closest celebrity in `CELEB_BUCKET`. There's no threshold: you always get the best match.
4. The result card shows the celebrity's **name**, their **reference photo** from `static/images/celeb/`, and the **match score** and distance.

**Adding celebrities:**

1. Drop `<filename>.jpg` into `static/images/celeb/`.
2. Add a row `<id>,<Display Name>, ,<filename>` to `Labels.csv`.
3. Re-run `python ingest_celeb.py`.

---

## API reference

All `POST` endpoints take `multipart/form-data`.

| Method & path | Form fields | Success response |
|---|---|---|
| `GET /` | | Register page |
| `POST /upload` | `name`, `image` (file) | `{"success": true, "filename": "<docId>"}` |
| `GET /verify` | | Verify page |
| `POST /verify` | `image` (file) | `{"success": true, "reason": "Welcome, ...", "name": "...", "distance": 0.08, "score": 92.0}` |
| `GET /celeb` | | Look-alike page |
| `POST /celeb` | `photo` (file) | `{"celeb_name": "...", "celeb_image_url": "/static/images/celeb/x.jpg", "distance": 0.31, "score": 69.0}` |

If the face fails validation:

- `/upload` and `/verify` return HTTP 400 with `{"success": false, "reason": "..."}`.
- `/celeb` returns HTTP 400 with `{"error": "..."}`.

Example using `curl`:

```bash
curl -F "name=Jane Doe" -F "image=@jane.jpg" http://localhost:5000/upload
curl -F "image=@jane2.jpg" http://localhost:5000/verify
curl -F "photo=@me.jpg"   http://localhost:5000/celeb
```

---

## Quick Setup with Claude Code

If you use [Claude Code](https://claude.com/claude-code), open a terminal in the repo root, run `claude`, and paste this prompt. It walks you through the whole setup interactively:

```text
Help me set up and run this face recognition project locally. Read README.md,
app.py, database.py, face_recognition.py, ingest_celeb.py and .env.example first,
then guide me step by step, pausing whenever you need input from me:

1. Check my Python version (3.11 or 3.12 required). If it's not suitable, tell me
   how to install a compatible version.
2. Create a virtual environment in .venv and install requirements.txt into it.
   If TensorFlow / tf-keras / numpy versions conflict, diagnose and fix it.
3. Copy .env.example to .env. Ask me for my Couchbase connection string, username
   and password (or whether I'm using Capella vs. a local Couchbase Server 8.0+),
   plus the bucket/scope/collection names I want, and write them into .env.
   Never print my password back or commit .env.
4. Using my credentials, write and run a small Python check that connects to the
   cluster and confirms the face bucket/scope/collection and the celeb bucket
   exist. If anything is missing, give me the exact steps (UI or SQL++) to create
   it, then the optional CREATE PRIMARY INDEX statements from the README, and re-check.
5. Ask whether I want the celebrity look-alike feature. If yes, run
   `python ingest_celeb.py` and confirm documents with embeddings were written,
   then create the celebrity Hyperscale Vector Index from README step 6
   (dimension 512, COSINE) and wait until its state is online.
6. Start the app with `python app.py` and confirm it's listening.
7. Explain how to test both use cases in the browser:
   - Register a face at http://localhost:5000/. Once at least one face is
     registered, create the face Hyperscale Vector Index from README step 6
     (dimension 512, COSINE) and wait until it is online.
   - Verify it at http://localhost:5000/verify and interpret the match score /
     distance (and how VERIFY_DISTANCE_THRESHOLD affects the result).
   - Find a celebrity look-alike at http://localhost:5000/celeb.
If any step fails, read the error, explain the cause, and fix it before moving on.
```

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `UnAmbiguousTimeoutException` / can't connect | Wrong `COUCHBASE_HOST`; Capella needs `couchbases://` and your IP in the allowed list |
| `No index available on keyspace` | Create the Hyperscale Vector Index for that keyspace ([step 6](#6-create-the-hyperscale-vector-indexes)) |
| `CREATE VECTOR INDEX` fails, or the index never comes online | Load data into the collection first (the index trains on existing vectors), and check the cluster runs Couchbase 8.0+ with the Index service |
| `function APPROX_VECTOR_DISTANCE not found` | The cluster is older than Couchbase 8.0. Upgrade it, or use a Capella cluster on 8.0+. |
| Wrong or poor matches after a lot of new registrations | Drop and recreate the face vector index so its centroids are retrained |
| `Please run pip install tf-keras` | `pip install tf-keras` (already in `requirements.txt`) |
| "No face detected" / "Face is too small" | Move closer, face the camera, improve lighting, keep the whole face in frame |
| Camera doesn't start | Open the app via `http://localhost`, not a LAN IP, or serve over HTTPS; allow camera permission |
| First request is very slow | DeepFace is downloading model weights to `~/.deepface`. This only happens once. |
| Verify always denies | Check the logged distance, then raise `VERIFY_DISTANCE_THRESHOLD` slightly or register more photos of the person |

---

## Privacy & data notice

- The app stores **face embeddings and names** only; it never stores uploaded images. Embeddings are still biometric data, so get consent before registering anyone, and follow the privacy laws that apply to you (for example GDPR or BIPA).
- The celebrity images in `static/images/celeb/` are included for demo purposes only. Rights stay with their respective owners. Replace them with images you have the rights to before any public or commercial use.
- This is a demo, not a production access-control system. It has **no liveness or anti-spoofing check**: a photo of a registered person can pass verification.
