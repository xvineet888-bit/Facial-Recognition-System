"""FaceID Flask dashboard: optimized recognition, registered persons and event history."""
from pathlib import Path
from werkzeug.utils import secure_filename
import os, threading, time, shutil, sqlite3, csv, base64, zipfile, tempfile, secrets, hashlib, json
from io import StringIO, BytesIO
from datetime import datetime, date, timedelta
import cv2, face_recognition, numpy as np
from flask import Flask, Response, jsonify, render_template, request, redirect, url_for, flash, session, send_file
from src.face_recognition import recognize_faces
from src.utils import load_encodings, RAW_DIR, save_encodings
from src.prepare_dataset import extract_dataset_encodings
app=Flask(__name__)
ROOT=Path(__file__).resolve().parent

def get_persistent_secret_key():
    """Use an environment secret, or create a private key that survives restarts."""
    configured = os.environ.get("FACE_DASHBOARD_SECRET")
    if configured:
        return configured
    key_path = ROOT / ".faceid_secret_key"
    try:
        if key_path.exists():
            key = key_path.read_text(encoding="utf-8").strip()
            if len(key) >= 64:
                return key
        key = secrets.token_hex(32)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        try:
            fd = os.open(str(key_path), flags, 0o600)
        except FileExistsError:
            existing = key_path.read_text(encoding="utf-8").strip()
            if len(existing) < 64:
                raise RuntimeError(
                    "The stored Flask secret key is invalid. Set FACE_DASHBOARD_SECRET "
                    "or replace .faceid_secret_key with a new private key."
                )
            return existing
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(key)
        return key
    except OSError as exc:
        raise RuntimeError(
            "Cannot persist the Flask secret key. Set FACE_DASHBOARD_SECRET "
            "to a private, stable value before starting the application."
        ) from exc

app.secret_key = get_persistent_secret_key()
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024
HISTORY_DB=ROOT/"data"/"recognition_history.db"
lock=threading.RLock()
event_lock=threading.Lock()
camera_lifecycle_lock = threading.RLock()
camera_threads = []
camera = None
running = False
latest_jpeg = None

latest_frame = None
latest_frame_id = 0
latest_boxes = []
latest_status={"name":"Camera stopped","faces":0,"updated":None}
encodings=labels=None
MATCH_TOLERANCE=.48; RESIZE_FACTOR=.40; PROCESS_EVERY_N_FRAMES=5; JPEG_QUALITY=65
ALLOWED_EXTENSIONS={".jpg",".jpeg",".png",".webp"}
LOG_COOLDOWN_SECONDS=30
last_logged={}

def db():
    HISTORY_DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(HISTORY_DB)
    con.execute("""CREATE TABLE IF NOT EXISTS events(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person TEXT NOT NULL,
        result TEXT NOT NULL,
        event_time TEXT NOT NULL,
        source TEXT NOT NULL DEFAULT 'unknown'
    )""")
    event_columns = {row[1] for row in con.execute("PRAGMA table_info(events)")}
    if "source" not in event_columns:
        con.execute("ALTER TABLE events ADD COLUMN source TEXT NOT NULL DEFAULT 'unknown'")
    con.execute("""CREATE TABLE IF NOT EXISTS group_reports(
        report_id TEXT PRIMARY KEY,
        report_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS system_activity(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        action TEXT NOT NULL,
        details TEXT NOT NULL,
        event_time TEXT NOT NULL
    )""")
    con.commit()
    return con


def log_event(name, source="webcam"):
    """Save a recognition result with source-aware duplicate suppression."""
    now = time.time()
    key = (str(source), str(name))
    with lock, event_lock:
        if now - last_logged.get(key, 0) < LOG_COOLDOWN_SECONDS:
            return
        con = db()
        try:
            result = "Unknown" if str(name) == "Unknown" else "Recognized"
            con.execute(
                "INSERT INTO events (person, result, event_time, source) VALUES (?, ?, ?, ?)",
                (str(name), result, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), str(source))
            )
            con.commit()
            last_logged[key] = now
        finally:
            con.close()


def log_system_action(action, details=""):
    with lock:
        con = db()
        try:
            con.execute(
                "INSERT INTO system_activity(action, details, event_time) VALUES(?,?,?)",
                (action, details[:500], datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            con.commit()
        finally:
            con.close()


def save_group_report(rows):
    with lock:
        """Store potentially large reports server-side; keep only an opaque ID in the cookie."""
        report_id = secrets.token_urlsafe(24)
        previous_id = session.get("group_report_id")
        con = db()
        try:
            if previous_id:
                con.execute("DELETE FROM group_reports WHERE report_id = ?", (previous_id,))
            con.execute(
                "DELETE FROM group_reports WHERE created_at < ?",
                ((datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S"),)
            )
            con.execute(
                "INSERT INTO group_reports(report_id, report_json, created_at) VALUES(?,?,?)",
                (report_id, json.dumps(rows), datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            con.commit()
        finally:
            con.close()
        session["group_report_id"] = report_id

def get_group_report():
    with lock:
        report_id = session.get("group_report_id")
        if not report_id:
            return []
        con = db()
        try:
            row = con.execute(
                "SELECT report_json FROM group_reports WHERE report_id = ?",
                (report_id,)
            ).fetchone()
            return json.loads(row[0]) if row else []
        finally:
            con.close()

def clear_group_report():
    with lock:
        report_id = session.pop("group_report_id", None)
        if report_id:
            con = db()
            try:
                con.execute("DELETE FROM group_reports WHERE report_id = ?", (report_id,))
                con.commit()
            finally:
                con.close()

def safe_spreadsheet_value(value):
    """Prevent spreadsheet formula execution for untrusted text fields."""
    if isinstance(value, str) and value.lstrip(" \t\r\n").startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def management_csrf_token():
    token = session.get("management_csrf")
    if not token:
        token = secrets.token_urlsafe(32)
        session["management_csrf"] = token
    return token


def verify_management_csrf():
    supplied = request.form.get("csrf_token", "")
    expected = session.get("management_csrf", "")
    if not expected or not secrets.compare_digest(supplied, expected):
        raise ValueError("Security check failed. Refresh the page and try again.")


def load_database():
    global encodings, labels
    with lock:
        loaded_encodings, loaded_labels = load_encodings()
        encodings, labels = loaded_encodings, loaded_labels


def get_database_snapshot():
    """Return a consistent encoding/label pair for a recognition request."""
    global encodings, labels
    with lock:
        if encodings is None or labels is None:
            load_database()
        return encodings, labels

def rebuild_database():
    """Rebuild all encodings. Kept for operations that remove dataset entries."""
    global encodings,labels
    new_e,new_l=extract_dataset_encodings()
    save_encodings(new_e,new_l)
    encodings=np.asarray(new_e,dtype=np.float64); labels=np.asarray(new_l)


def existing_image_hashes():
    """Return SHA-256 hashes for registered images to prevent exact duplicates."""
    hashes = set()
    if RAW_DIR.exists():
        for path in RAW_DIR.rglob("*"):
            if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS:
                try:
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    hashes.add(digest)
                except OSError:
                    continue
    return hashes


def append_uploaded_encodings(vectors, person_name):
    """Append newly computed encodings without reprocessing existing photos."""
    global encodings, labels

    if encodings is None or labels is None:
        try:
            current_e, current_l = load_encodings()
        except FileNotFoundError:
            current_e = np.empty((0, 128), dtype=np.float64)
            current_l = np.asarray([], dtype=str)
    else:
        current_e = np.asarray(encodings, dtype=np.float64)
        current_l = np.asarray(labels)

    new_e = np.asarray(vectors, dtype=np.float64).reshape(-1, 128)
    new_l = np.asarray([str(person_name)] * len(new_e))
    updated_e = np.vstack((current_e, new_e))
    updated_l = np.concatenate((current_l, new_l))
    save_encodings(updated_e, updated_l)
    encodings, labels = updated_e, updated_l


def validate_and_encode_upload(data, filename):
    """Decode, validate exactly one face, and compute its encoding once."""
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"{filename} is not a readable image.")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb, model="hog")
    if len(locations) != 1:
        raise ValueError(f"{filename}: expected exactly one face, found {len(locations)}.")
    vectors = face_recognition.face_encodings(rgb, locations)
    if len(vectors) != 1:
        raise ValueError(f"{filename}: face encoding could not be generated.")
    return vectors[0]

def registered_people():
    people=[]
    if RAW_DIR.exists():
        for folder in sorted(p for p in RAW_DIR.iterdir() if p.is_dir()):
            pics=[p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS]
            if pics: people.append({"name":folder.name,"photos":len(pics)})
    return people


def camera_capture_worker():
    global camera, running, latest_jpeg
    global latest_frame, latest_frame_id, latest_boxes

    with lock:
        cap = camera

    try:
        while True:
            with lock:
                if not running:
                    break

            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            with lock:
                if not running:
                    break
                latest_frame = frame.copy()
                latest_frame_id += 1
                boxes = latest_boxes.copy()

            display_frame = frame.copy()
            for top, right, bottom, left, name in boxes:
                color = (40, 190, 120) if name != "Unknown" else (40, 90, 230)
                cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(display_frame, (left, max(0, top - 30)), (right, top), color, -1)
                cv2.putText(
                    display_frame, name, (left + 6, max(18, top - 9)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )

            ok, buffer = cv2.imencode(
                ".jpg", display_frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )
            if ok:
                with lock:
                    if running:
                        latest_jpeg = buffer.tobytes()
    finally:
        if cap:
            cap.release()
        with lock:
            if camera is cap:
                camera = None
            running = False
            latest_jpeg = None
            latest_frame = None
            latest_boxes = []


def camera_recognition_worker():
    global latest_boxes, latest_status
    last_processed_id = 0

    while True:
        with lock:
            if not running:
                break
            frame = latest_frame.copy() if latest_frame is not None else None
            frame_id = latest_frame_id
            # Take a consistent pair; writers replace arrays rather than mutating them.
            current_encodings = encodings
            current_labels = labels

        if frame is None:
            time.sleep(0.01)
            continue
        if frame_id - last_processed_id < PROCESS_EVERY_N_FRAMES:
            time.sleep(0.005)
            continue
        last_processed_id = frame_id

        try:
            small = cv2.resize(
                frame, None, fx=RESIZE_FACTOR, fy=RESIZE_FACTOR,
                interpolation=cv2.INTER_AREA
            )
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb, model="hog")
            vectors = face_recognition.face_encodings(rgb, locations)

            new_boxes, names = [], []
            scale = 1 / RESIZE_FACTOR

            for (top, right, bottom, left), vector in zip(locations, vectors):
                idx = recognize_faces(
                    current_encodings, vector, tolerance=MATCH_TOLERANCE
                ) if current_encodings is not None and len(current_encodings) else None
                name = str(current_labels[idx]) if idx is not None else "Unknown"
                names.append(name)
                new_boxes.append(
                    tuple(int(value * scale) for value in (top, right, bottom, left))
                    + (name,)
                )
                log_event(name, source="webcam")

            with lock:
                latest_boxes = new_boxes
                latest_status = {
                    "name": ", ".join(names) if names else "No face detected",
                    "faces": len(locations),
                    "updated": time.strftime("%H:%M:%S")
                }
        except Exception as exc:
            print("Face recognition error:", exc)
            time.sleep(0.05)


def start_camera():
    global camera, running
    global latest_jpeg, latest_frame, latest_frame_id, latest_boxes, camera_threads

    with camera_lifecycle_lock:
        with lock:
            if running:
                return
            if camera is not None or any(thread.is_alive() for thread in camera_threads):
                raise RuntimeError("The previous camera session is still stopping. Please wait.")

            cap = cv2.VideoCapture(
                0, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
            )
            if not cap.isOpened():
                cap.release()
                raise RuntimeError(
                    "Could not open webcam. Close other camera apps and try again."
                )

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            camera = cap
            running = True
            latest_jpeg = None
            latest_frame = None
            latest_frame_id = 0
            latest_boxes = []
            latest_status.update({"name": "Starting camera", "faces": 0,
                                  "updated": time.strftime("%H:%M:%S")})

            capture_thread = threading.Thread(
                target=camera_capture_worker, daemon=True, name="faceid-camera-capture"
            )
            recognition_thread = threading.Thread(
                target=camera_recognition_worker, daemon=True, name="faceid-camera-recognition"
            )
            camera_threads = [capture_thread, recognition_thread]
            capture_thread.start()
            recognition_thread.start()


def stop_camera():
    """Stop and join camera workers before allowing another camera session."""
    global running, camera, latest_jpeg, latest_frame, latest_boxes, camera_threads
    with camera_lifecycle_lock:
        with lock:
            running = False
            threads = list(camera_threads)
            cap = camera
        # Do not hold the state lock while joining workers.
        for thread in threads:
            if thread.is_alive() and thread is not threading.current_thread():
                thread.join(timeout=3.0)
        with lock:
            if cap is not None and camera is cap:
                # A blocked capture read may still be active; release as a fallback.
                cap.release()
                camera = None
            latest_jpeg = None
            latest_frame = None
            latest_boxes = []
            latest_status.update({"name": "Camera stopped", "faces": 0,
                                  "updated": time.strftime("%H:%M:%S")})
            if not any(thread.is_alive() for thread in camera_threads):
                camera_threads = []
            if camera is None:
                running = False


@app.route("/")
def index(): return render_template("index.html")

@app.route("/identify-image", methods=["GET", "POST"])
def identify_image():
    annotated_image = None
    results = []
    error = None
    if request.method == "POST":
        uploaded = request.files.get("photo")
        if not uploaded or not uploaded.filename:
            error = "Choose an image first."
        elif Path(uploaded.filename).suffix.lower() not in ALLOWED_EXTENSIONS:
            error = "Use a JPG, JPEG, PNG or WEBP image."
        else:
            raw = uploaded.read()
            image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                error = "The selected file is not a readable image."
            else:
                try:
                    current_encodings, current_labels = get_database_snapshot()
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    locations = face_recognition.face_locations(rgb, model="hog")
                    vectors = face_recognition.face_encodings(rgb, locations)
                    if not locations:
                        error = "No faces were detected in this image. Try a clearer, front-facing photo."
                    for (top, right, bottom, left), vector in zip(locations, vectors):
                        name = "Unknown"
                        distance = None
                        if current_encodings is not None and len(current_encodings) > 0:
                            distances = face_recognition.face_distance(np.asarray(current_encodings), vector)
                            best = int(np.argmin(distances))
                            distance = float(distances[best])
                            if distance <= MATCH_TOLERANCE:
                                name = str(current_labels[best])
                        known = name != "Unknown"
                        color = (40, 190, 120) if known else (40, 90, 230)
                        cv2.rectangle(image, (left, top), (right, bottom), color, 3)
                        label = name if distance is None else f"{name} ({distance:.2f})"
                        cv2.rectangle(image, (left, max(0, top-32)), (right, top), color, -1)
                        cv2.putText(image, label, (left+6, max(20, top-10)), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 2)
                        results.append({"name": name, "distance": None if distance is None else round(distance, 4), "status": "Recognized" if known else "Unknown"})
                        log_event(name, source="uploaded_image")
                    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                    if ok:
                        annotated_image = "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")
                    else:
                        error = "Could not prepare the annotated image."
                except Exception as exc:
                    error = f"Image identification failed: {exc}"
    return render_template("image_recognition.html", annotated_image=annotated_image, results=results, error=error)

@app.errorhandler(413)
def image_too_large(_error):
    return "Request is too large. The maximum upload size is 512 MB.", 413




@app.route("/image_recognition", methods=["GET", "POST"])
def image_recognition():
    import os
    import base64
    import cv2
    import numpy as np
    import face_recognition
    from pathlib import Path
    from flask import request, render_template

    annotated_image = None
    results = []
    error = None
    recognition_source = "registered"

    if request.method == "POST":
        recognition_source = request.form.get(
            "recognition_source", "registered"
        )

        if recognition_source not in ("registered", "lfw"):
            recognition_source = "registered"

        if recognition_source == "registered":
            clear_group_report()

        uploaded = request.files.get("photo")

        if not uploaded or not uploaded.filename:
            error = "Please select a photograph."

        elif Path(uploaded.filename).suffix.lower() not in ALLOWED_EXTENSIONS:
            error = "Upload JPG, JPEG, PNG or WEBP only."

        else:
            try:
                raw = uploaded.read()

                image = cv2.imdecode(
                    np.frombuffer(raw, dtype=np.uint8),
                    cv2.IMREAD_COLOR
                )

                if image is None:
                    raise ValueError("Invalid or unreadable image.")

                # Select recognition database
                if recognition_source == "lfw":
                    desktop = ROOT.parent.parent / "Datasets"

                    current_encodings = np.load(
                        desktop / "lfw_gallery_encodings.npy"
                    )
                    current_labels = np.load(
                        desktop / "lfw_gallery_labels.npy"
                    )

                    if len(current_encodings) != len(current_labels):
                        raise ValueError("LFW gallery data is inconsistent.")

                    active_tolerance = 0.48

                else:
                    current_encodings, current_labels = (
                        get_database_snapshot()
                    )
                    active_tolerance = MATCH_TOLERANCE

                # Detect and encode uploaded faces
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                locations = face_recognition.face_locations(
                    rgb, model="hog"
                )

                vectors = face_recognition.face_encodings(
                    rgb, locations
                )

                if not locations:
                    raise ValueError("No faces detected in this photo.")

                for index, ((top, right, bottom, left), vector) in enumerate(
                    zip(locations, vectors), start=1
                ):
                    name = "Unknown"
                    distance = None

                    if (
                        current_encodings is not None
                        and len(current_encodings) > 0
                    ):
                        distances = face_recognition.face_distance(
                            np.asarray(current_encodings),
                            vector
                        )

                        best = int(np.argmin(distances))
                        distance = float(distances[best])

                        if distance <= active_tolerance:
                            name = str(current_labels[best])

                    known = name != "Unknown"
                    color = (40, 190, 120) if known else (40, 90, 230)

                    results.append({
                        "face": index,
                        "name": name,
                        "status": "Recognized" if known else "Unknown",
                        "distance": (
                            round(distance, 4)
                            if distance is not None else None
                        )
                    })

                    # Only registered-person recognition enters normal history
                    if recognition_source == "registered":
                        image_source = (
                            "uploaded_image"
                            if len(locations) == 1
                            else "group_photo"
                        )
                        log_event(name, source=image_source)

                    # Draw face rectangle and identity
                    cv2.rectangle(
                        image, (left, top), (right, bottom), color, 3
                    )

                    label = f"Face {index}: {name}"

                    cv2.rectangle(
                        image,
                        (left, max(0, top - 32)),
                        (right, top),
                        color,
                        -1
                    )

                    cv2.putText(
                        image,
                        label,
                        (left + 5, max(20, top - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2
                    )

                # Convert annotated image for display
                ok, encoded = cv2.imencode(".jpg", image)

                if not ok:
                    raise ValueError(
                        "Could not generate annotated image."
                    )

                annotated_image = (
                    "data:image/jpeg;base64,"
                    + base64.b64encode(
                        encoded.tobytes()
                    ).decode("ascii")
                )

                # Only registered-person results enter normal reports
                if recognition_source == "registered":
                    save_group_report(results)

            except Exception as exc:
                error = str(exc)

    recognized = sum(
        r["status"] == "Recognized" for r in results
    )

    unknown = sum(
        r["status"] == "Unknown" for r in results
    )

    return render_template(
        "image_recognition.html",
        annotated_image=annotated_image,
        results=results,
        recognized=recognized,
        unknown=unknown,
        total=len(results),
        error=error,
        recognition_source=recognition_source
    )



@app.route("/image_recognition/export")
def export_group_report():
    rows = get_group_report()

    if not rows:
        flash("Run a group photo analysis before exporting.", "error")
        return redirect(url_for("group_analysis"))

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Face Number", "Person", "Status", "Face Distance"
    ])

    for row in rows:
        writer.writerow([
            row["face"], safe_spreadsheet_value(row["name"]),
            row["status"], row["distance"]
        ])

    from flask import make_response
    response = make_response(output.getvalue())
    response.headers["Content-Type"] = "text/csv; charset=utf-8"
    response.headers["Content-Disposition"] = (
        "attachment; filename=group_photo_analysis.csv"
    )
    return response

# ================= GROUP PHOTO PDF EXPORT =================

@app.route("/image_recognition/export-pdf")
def export_group_pdf():
    rows = get_group_report()

    if not rows:
        flash("Run a group photo analysis before exporting.", "error")
        return redirect(url_for("group_analysis"))

    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle
        )
    except ImportError:
        flash("Install ReportLab: pip install reportlab", "error")
        return redirect(url_for("group_analysis"))

    output = BytesIO()

    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm
    )

    styles = getSampleStyleSheet()

    recognized = sum(
        row.get("status") == "Recognized" for row in rows
    )

    unknown = sum(
        row.get("status") == "Unknown" for row in rows
    )

    content = [
        Paragraph("Group Photo Analysis Report", styles["Title"]),
        Spacer(1, 15),
        Paragraph(
            f"Total Faces: {len(rows)} | "
            f"Recognized: {recognized} | "
            f"Unknown: {unknown}",
            styles["Normal"]
        ),
        Spacer(1, 15)
    ]

    data = [
        ["Face Number", "Person", "Status", "Face Distance"]
    ]

    for row in rows:
        distance = row.get("distance")

        data.append([
            row.get("face", ""),
            safe_spreadsheet_value(row.get("name", "")),
            row.get("status", ""),
            f"{distance:.4f}"
            if isinstance(distance, (int, float))
            else "N/A"
        ])

    table = Table(
        data,
        repeatRows=1,
        colWidths=[30 * mm, 65 * mm, 42 * mm, 38 * mm]
    )

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#263445")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f1f5f9")]),
        ("PADDING", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
    ]))

    content.append(table)
    document.build(content)

    from flask import make_response

    response = make_response(output.getvalue())
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = (
        "attachment; filename=group_photo_analysis.pdf"
    )

    return response


# ================= GROUP PHOTO EXCEL EXPORT =================

@app.route("/group-analysis/export-excel")
def export_group_excel():
    rows = get_group_report()

    if not rows:
        flash("Run a group photo analysis before exporting.", "error")
        return redirect(url_for("image_recognition"))

    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment
    except ImportError:
        flash("Install openpyxl: pip install openpyxl", "error")
        return redirect(url_for("image_recognition"))

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "image_recognition"

    sheet.append([
        "Face Number",
        "Person",
        "Status",
        "Face Distance"
    ])

    for row in rows:
        sheet.append([
            row.get("face", ""),
            safe_spreadsheet_value(row.get("name", "")),
            row.get("status", ""),
            row.get("distance", "")
        ])

    header_fill = PatternFill(
        fill_type="solid",
        fgColor="263445"
    )

    for cell in sheet[1]:
        cell.font = Font(
            bold=True,
            color="FFFFFF"
        )

        cell.fill = header_fill

        cell.alignment = Alignment(
            horizontal="center"
        )

    widths = {
        "A": 16,
        "B": 28,
        "C": 18,
        "D": 18
    }

    for column, width in widths.items():
        sheet.column_dimensions[column].width = width

    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions

    output = BytesIO()
    workbook.save(output)
    output.seek(0)

    from flask import make_response

    response = make_response(output.getvalue())

    response.headers["Content-Type"] = (
        "application/vnd.openxmlformats-officedocument."
        "spreadsheetml.sheet"
    )

    response.headers["Content-Disposition"] = (
        "attachment; filename=group_photo_analysis.xlsx"
    )

    return response


@app.route("/persons")
def persons():
    with lock:
        people = registered_people()
    return render_template("persons.html", people=people)

@app.route("/persons/add",methods=["POST"])
def add_person():
    with lock:
        if running:
            flash("Stop the camera before adding a person.", "error")
            return redirect(url_for("persons"))
        name=" ".join(request.form.get("name"," ").strip().split())
        files=request.files.getlist("photos")
        if not name or not any(f.filename for f in files):
            flash("Enter a name and select at least one photo.","error")
            return redirect(url_for("persons"))
        if any(c in name for c in r'\/:*?"<>|') or name in {".",".."}:
            flash("Please use a simple name without special path characters.","error")
            return redirect(url_for("persons"))

        folder=RAW_DIR/name
        if folder.exists():
            flash("That person already exists. Choose another name.","error")
            return redirect(url_for("persons"))

        try:
            folder.mkdir(parents=True)
        except OSError as exc:
            flash(f"Could not create the person's folder: {exc}", "error")
            return redirect(url_for("persons"))
        known_hashes = existing_image_hashes()
        added_vectors = []
        added_files = []
        skipped_duplicates = 0
        database_update_started = False
        try:
            for uploaded in files:
                if not uploaded.filename:
                    continue
                suffix=Path(secure_filename(uploaded.filename)).suffix.lower()
                if suffix not in ALLOWED_EXTENSIONS:
                    raise ValueError("Only JPG, JPEG, PNG and WEBP are supported.")
                data=uploaded.read()
                digest=hashlib.sha256(data).hexdigest()
                if digest in known_hashes:
                    skipped_duplicates += 1
                    continue

                vector = validate_and_encode_upload(data, uploaded.filename)
                destination = folder / f"{len(added_files)+1:03d}{suffix}"
                destination.write_bytes(data)
                added_files.append(destination)
                added_vectors.append(vector)
                known_hashes.add(digest)

            if not added_vectors:
                raise ValueError("All selected photos are duplicates of images already registered.")

            # Save the new encodings only; existing dataset images are not reprocessed.
            database_update_started = True
            append_uploaded_encodings(added_vectors, name)
            flash(
                f"Added {name} with {len(added_files)} new photo(s). "
                + (f"Skipped {skipped_duplicates} duplicate(s)." if skipped_duplicates else "")
                ,"success"
            )
        except Exception as e:
            for path in added_files:
                path.unlink(missing_ok=True)
            # Rebuild only if the encoding-file update itself had started.
            if database_update_started:
                try:
                    rebuild_database()
                except Exception as rebuild_error:
                    print("Could not reconcile encodings after failed upload:", rebuild_error)
            shutil.rmtree(folder,ignore_errors=True)
            flash(f"Could not add person: {e}","error")
        return redirect(url_for("persons"))

@app.route("/persons/add-photos/<path:name>", methods=["POST"])
def add_person_photos(name):
    with lock:
        if running:
            flash("Stop the camera before adding photos.", "error")
            return redirect(url_for("persons"))

        folder = (RAW_DIR / name).resolve()
        root = RAW_DIR.resolve()
        if root not in folder.parents or not folder.is_dir():
            flash("Person folder not found.", "error")
            return redirect(url_for("persons"))

        files = request.files.getlist("photos")
        if not any(f.filename for f in files):
            flash("Select at least one photo.", "error")
            return redirect(url_for("persons"))

        known_hashes = existing_image_hashes()
        saved = []
        added_vectors = []
        skipped_duplicates = 0
        database_update_started = False
        try:
            next_number = len([p for p in folder.iterdir() if p.is_file()]) + 1
            for uploaded in files:
                if not uploaded.filename:
                    continue
                suffix = Path(secure_filename(uploaded.filename)).suffix.lower()
                if suffix not in ALLOWED_EXTENSIONS:
                    raise ValueError("Only JPG, JPEG, PNG and WEBP images are supported.")

                data = uploaded.read()
                digest = hashlib.sha256(data).hexdigest()
                if digest in known_hashes:
                    skipped_duplicates += 1
                    continue

                vector = validate_and_encode_upload(data, uploaded.filename)
                while True:
                    destination = folder / f"{next_number:03d}{suffix}"
                    next_number += 1
                    if not destination.exists():
                        break
                destination.write_bytes(data)
                saved.append(destination)
                added_vectors.append(vector)
                known_hashes.add(digest)

            if not added_vectors:
                flash(
                    "No new photos added. All selected photos were already registered."
                    ,"info"
                )
                return redirect(url_for("persons"))

            database_update_started = True
            append_uploaded_encodings(added_vectors, name)

            flash(
                f"Added {len(saved)} new photo(s) to {name}. Recognition database updated. "
                + (f"Skipped {skipped_duplicates} duplicate(s)." if skipped_duplicates else "")
                ,"success"
            )
        except Exception as exc:
            for path in saved:
                path.unlink(missing_ok=True)
            # Rebuild only if the encoding-file update itself had started.
            if database_update_started:
                try:
                    rebuild_database()
                except Exception as rebuild_error:
                    print("Could not reconcile encodings after failed upload:", rebuild_error)
            flash(f"Could not add photos: {exc}","error")

        return redirect(url_for("persons"))

@app.route("/persons/delete/<path:name>",methods=["POST"])
def delete_person(name):
    with lock:
        folder=(RAW_DIR/name).resolve(); root=RAW_DIR.resolve()
        if root not in folder.parents or not folder.is_dir():
            flash("Person folder not found.","error"); return redirect(url_for("persons"))
        if running:
            flash("Stop the camera before deleting a person.","error"); return redirect(url_for("persons"))
        backup = ROOT / "data" / f".faceid-deleting-{secrets.token_hex(8)}"
        try:
            backup.parent.mkdir(parents=True, exist_ok=True)
            folder.rename(backup)
            rebuild_database()
            shutil.rmtree(backup, ignore_errors=True)
            flash(f"Removed {name}.","success")
        except Exception as e:
            if backup.exists(): backup.rename(folder)
            flash(f"Could not remove person: {e}","error")
        return redirect(url_for("persons"))

def valid_date(value):
    try:
        return datetime.strptime(value, "%Y-%m-%d").date() if value else None
    except ValueError:
        return None

@app.route("/history")
def history():
    with lock:
        start = valid_date(request.args.get("start"))
        end = valid_date(request.args.get("end"))
        clauses, params = [], []
        if start:
            clauses.append("date(event_time) >= ?"); params.append(start.isoformat())
        if end:
            clauses.append("date(event_time) <= ?"); params.append(end.isoformat())
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        con=db()
        try:
            rows=con.execute("SELECT id,person,result,event_time,source FROM events"+where+" ORDER BY id DESC LIMIT 500", params).fetchall()
            total=con.execute("SELECT COUNT(*) FROM events"+where, params).fetchone()[0]
            unknown=con.execute("SELECT COUNT(*) FROM events"+where+(" AND " if where else " WHERE ")+"result='Unknown'", params).fetchone()[0]
            people = [row[0] for row in con.execute("SELECT DISTINCT person FROM events ORDER BY person COLLATE NOCASE").fetchall()]
        finally: con.close()
        return render_template(
            "history.html", events=rows, total=total, unknown=unknown,
            start=request.args.get("start", ""), end=request.args.get("end", "")
        )

@app.route("/history/export")
def export_history():
    with lock:
        start = valid_date(request.args.get("start"))
        end = valid_date(request.args.get("end"))
        clauses, params = [], []
        if start:
            clauses.append("date(event_time) >= ?"); params.append(start.isoformat())
        if end:
            clauses.append("date(event_time) <= ?"); params.append(end.isoformat())
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        con=db()
        try:
            rows=con.execute("SELECT id,person,result,event_time,source FROM events"+where+" ORDER BY id DESC", params).fetchall()
        finally: con.close()
        output=StringIO(); writer=csv.writer(output)
        writer.writerow(["Event ID", "Person", "Result", "Date & Time", "Source"])
        writer.writerows(
            (event_id, safe_spreadsheet_value(person), result, event_time, source)
            for event_id, person, result, event_time, source in rows
        )
        from flask import make_response
        response=make_response(output.getvalue())
        response.headers["Content-Type"]="text/csv; charset=utf-8"
        response.headers["Content-Disposition"]="attachment; filename=faceid_recognition_history.csv"
        return response

@app.route("/analytics")
def analytics():
    with lock:
        today=date.today()
        days=[today-timedelta(days=i) for i in range(6,-1,-1)]
        labels_days=[d.strftime("%a") for d in days]
        start_day=days[0].isoformat()
        con=db()
        try:
            total=con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            unknown=con.execute("SELECT COUNT(*) FROM events WHERE result='Unknown'").fetchone()[0]
            known=total-unknown
            daily_rows=con.execute("SELECT date(event_time), SUM(CASE WHEN result!='Unknown' THEN 1 ELSE 0 END), SUM(CASE WHEN result='Unknown' THEN 1 ELSE 0 END) FROM events WHERE date(event_time)>=? GROUP BY date(event_time)", (start_day,)).fetchall()
            daily={row[0]:(row[1] or 0,row[2] or 0) for row in daily_rows}
            known_daily=[daily.get(d.isoformat(),(0,0))[0] for d in days]
            unknown_daily=[daily.get(d.isoformat(),(0,0))[1] for d in days]
            top_rows=con.execute("SELECT person, COUNT(*) AS hits FROM events WHERE result!='Unknown' GROUP BY person ORDER BY hits DESC, person ASC LIMIT 5").fetchall()
        finally: con.close()
        return render_template("analytics.html", total=total, unknown=unknown, known=known,
                               day_labels=labels_days, known_daily=known_daily,
                               unknown_daily=unknown_daily, top_people=top_rows)

def evaluate_model():
    """Evaluate one held-out image per identity against the remaining photos."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

    if running:
        raise RuntimeError("Stop the webcam before running model evaluation.")
    if not RAW_DIR.exists():
        raise RuntimeError("The raw dataset folder was not found.")

    reference_vectors, reference_labels, test_items, skipped = [], [], [], []
    for folder in sorted(p for p in RAW_DIR.iterdir() if p.is_dir()):
        usable = []
        for image_path in sorted(p for p in folder.iterdir()
                                 if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS):
            try:
                image = face_recognition.load_image_file(str(image_path))
                locations = face_recognition.face_locations(image, model="hog")
                if len(locations) != 1:
                    skipped.append(f"{folder.name}/{image_path.name}: expected one face, found {len(locations)}")
                    continue
                vectors = face_recognition.face_encodings(image, locations)
                if len(vectors) != 1:
                    skipped.append(f"{folder.name}/{image_path.name}: encoding failed")
                    continue
                usable.append((image_path.name, vectors[0]))
            except Exception as exc:
                skipped.append(f"{folder.name}/{image_path.name}: {exc}")

        if len(usable) < 2:
            skipped.append(f"{folder.name}: needs at least 2 usable photos; found {len(usable)}")
            continue

        test_name, test_vector = usable[-1]
        for _, vector in usable[:-1]:
            reference_vectors.append(vector)
            reference_labels.append(folder.name)
        test_items.append((folder.name, test_name, test_vector))

    if not test_items or not reference_vectors:
        raise RuntimeError(
            "Not enough data for holdout testing. Each evaluated person needs at least "
            "2 clear photos in data/raw/<PersonName>/."
        )

    y_true, y_pred, timings = [], [], []
    for true_name, filename, vector in test_items:
        started = time.perf_counter()
        distances = face_recognition.face_distance(np.asarray(reference_vectors), vector)
        best_index = int(np.argmin(distances))
        best_distance = float(distances[best_index])
        predicted = reference_labels[best_index] if best_distance <= MATCH_TOLERANCE else "Unknown"
        timings.append((time.perf_counter() - started) * 1000)
        y_true.append(true_name)
        y_pred.append(predicted)

    class_names = sorted(set(y_true))
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_names, average="macro", zero_division=0
    )
    matrix_labels = class_names + ["Unknown"]
    matrix = confusion_matrix(y_true, y_pred, labels=matrix_labels).tolist()

    return {
        "accuracy": round(float(accuracy) * 100, 1),
        "precision": round(float(precision) * 100, 1),
        "recall": round(float(recall) * 100, 1),
        "f1": round(float(f1) * 100, 1),
        "tested": len(test_items),
        "reference_count": len(reference_vectors),
        "person_count": len(class_names),
        "avg_ms": round(float(np.mean(timings)), 2),
        "tolerance": MATCH_TOLERANCE,
        "matrix_labels": matrix_labels,
        "matrix": matrix,
        "skipped": skipped,
    }


@app.route("/evaluation", methods=["GET", "POST"])
def evaluation():
    result = None
    error = None
    if request.method == "POST":
        try:
            result = evaluate_model()
        except Exception as exc:
            error = str(exc)
    return render_template("evaluation.html", result=result, error=error)

@app.route("/api/start",methods=["POST"])
def api_start():
    try:
        get_database_snapshot()
        start_camera(); return jsonify(ok=True,message="Camera started")
    except Exception as e: return jsonify(ok=False,message=str(e)),400

@app.route("/api/stop",methods=["POST"])
def api_stop():
    stop_camera()
    return jsonify(ok=True, message="Camera stopped")

@app.route("/video")
def video():
    def generate():
        while True:
            with lock: frame=latest_jpeg; active=running
            if frame: yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+frame+b"\r\n"
            else: time.sleep(.08 if not active else .03)
    return Response(generate(),mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/identity-count")
def identity_count():
    try:
        _current_encodings, current_labels = get_database_snapshot()
        return jsonify(count=len(set(str(x) for x in current_labels)))
    except Exception:return jsonify(count=0)

@app.route("/api/status")
def status():
    with lock: data=dict(latest_status); data["running"]=running
    return jsonify(data)


# ================= SHARED SYSTEM MANAGEMENT (NO LOGIN) =================

def _data_size(path):
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


@app.route("/system-management")
def system_management():
    with lock:
        people = registered_people()
        con = db()
        try:
            events_count = con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            activities = con.execute(
                "SELECT id, action, details, event_time FROM system_activity ORDER BY id DESC LIMIT 20"
            ).fetchall()
        finally:
            con.close()
        return render_template(
            "system_management.html",
            people=people,
            events_count=events_count,
            total_photos=sum(person["photos"] for person in people),
            data_size=_data_size(ROOT / "data"),
            activities=activities,
            csrf_token=management_csrf_token(),
            camera_running=running,
        )

@app.route("/system-management/activity/<int:activity_id>/delete", methods=["POST"])
def delete_system_activity(activity_id):
    """Delete one selected management activity record without creating a new log entry."""
    with lock:
        try:
            verify_management_csrf()
            con = db()
            try:
                deleted = con.execute(
                    "DELETE FROM system_activity WHERE id = ?",
                    (activity_id,)
                ).rowcount
                con.commit()
            finally:
                con.close()

            if deleted:
                flash("Selected system activity record was permanently deleted.", "success")
            else:
                flash("That activity record was not found; it may already have been deleted.", "error")
        except Exception as exc:
            flash(f"Could not delete the activity record: {exc}", "error")
    return redirect(url_for("system_management"))


@app.route("/system-management/clear-history-date", methods=["POST"])
def clear_recognition_history_date():
    with lock:
        """Delete recognition events for one selected date only."""
        try:
            verify_management_csrf()
            if running:
                raise ValueError("Stop the webcam before clearing recognition history.")
            selected_date = valid_date(request.form.get("selected_date", ""))
            if not selected_date:
                raise ValueError("Choose a valid date.")

            con = db()
            try:
                deleted = con.execute(
                    "DELETE FROM events WHERE date(event_time) = ?",
                    (selected_date.isoformat(),)
                ).rowcount
                con.commit()
            finally:
                con.close()

            log_system_action(
                "Recognition history cleared by date",
                f"Deleted {deleted} recognition event(s) from {selected_date.isoformat()}; registered-person data was not changed."
            )
            flash(f"Deleted {deleted} recognition event(s) from {selected_date.isoformat()}. Registered photos and encodings were not changed.", "success")
        except Exception as exc:
            flash(f"Could not clear recognition history: {exc}", "error")
        return redirect(url_for("system_management"))

@app.route("/system-management/clear-history", methods=["POST"])
def clear_recognition_history():
    with lock:
        """Delete all recognition events while preserving management activity records."""
        global last_logged
        try:
            verify_management_csrf()
            if running:
                raise ValueError("Stop the webcam before clearing recognition history.")

            con = db()
            try:
                deleted = con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
                con.execute("DELETE FROM events")
                con.commit()
            finally:
                con.close()

            with event_lock:
                last_logged.clear()
            log_system_action(
                "All recognition history cleared",
                f"Deleted {deleted} recognition event(s); management activity was retained."
            )
            flash(f"All recognition history cleared. {deleted} event(s) deleted. System activity logs were kept.", "success")
        except Exception as exc:
            flash(f"Could not clear recognition history: {exc}", "error")
        return redirect(url_for("system_management"))

@app.route("/system-management/backup", methods=["POST"])
def create_backup():
    with lock:
        try:
            verify_management_csrf()
            if running:
                raise ValueError("Stop the webcam before creating a backup.")
            data_root = ROOT / "data"
            data_root.mkdir(parents=True, exist_ok=True)
            fd, snapshot_path = tempfile.mkstemp(prefix="faceid-db-", suffix=".sqlite")
            os.close(fd)
            try:
                source = sqlite3.connect(str(HISTORY_DB))
                target = sqlite3.connect(snapshot_path)
                try:
                    source.backup(target)
                finally:
                    target.close()
                    source.close()
                memory_file = BytesIO()
                with zipfile.ZipFile(memory_file, "w", zipfile.ZIP_DEFLATED) as archive:
                    archive.write(snapshot_path, "data/recognition_history.db")
                    # Explicit entries preserve empty folders in backups.
                    archive.writestr("data/raw/", b"")
                    archive.writestr("data/processed/", b"")
                    for relative in (Path("data/raw"), Path("data/processed")):
                        folder = ROOT / relative
                        if folder.exists():
                            for item in folder.rglob("*"):
                                if item.is_file():
                                    archive.write(item, item.relative_to(ROOT).as_posix())
                memory_file.seek(0)
            finally:
                try:
                    os.remove(snapshot_path)
                except OSError:
                    pass
            log_system_action("Backup created", "SQLite history database, registered photos and processed encodings")
            return send_file(
                memory_file,
                mimetype="application/zip",
                as_attachment=True,
                download_name=f"faceid_backup_{datetime.now():%Y%m%d_%H%M%S}.zip",
            )
        except Exception as exc:
            flash(f"Backup failed: {exc}", "error")
            return redirect(url_for("system_management"))

@app.route("/system-management/restore", methods=["POST"])
def restore_backup():
    global encodings, labels
    try:
        verify_management_csrf()
        with lock:
            if running:
                raise ValueError("Stop the webcam before restoring a backup.")

            uploaded = request.files.get("backup_file")
            if not uploaded or not uploaded.filename.lower().endswith(".zip"):
                raise ValueError("Choose a FaceID .zip backup file.")

            with tempfile.TemporaryDirectory(prefix="faceid-restore-") as temp_dir:
                temp_root = Path(temp_dir)
                archive_path = temp_root / "upload.zip"
                uploaded.save(archive_path)

                with zipfile.ZipFile(archive_path, "r") as archive:
                    infos = archive.infolist()
                    names = [info.filename for info in infos]
                    required_db = "data/recognition_history.db"
                    if names.count(required_db) != 1:
                        raise ValueError("The backup must contain exactly one FaceID database.")

                    seen = set()
                    total_uncompressed = 0
                    has_raw = False
                    has_processed = False
                    allowed_directory_names = {"data/", "data/raw/", "data/processed/"}

                    for info in infos:
                        name = info.filename
                        if name in seen:
                            raise ValueError("The backup contains duplicate file paths.")
                        seen.add(name)

                        if name.startswith("/") or "\\" in name:
                            raise ValueError("The backup contains an unsafe file path.")
                        parts = name.rstrip("/").split("/")
                        if any(part in ("", ".", "..") for part in parts):
                            raise ValueError("The backup contains an unsafe file path.")
                        if ":" in parts[0]:
                            raise ValueError("The backup contains an unsafe file path.")

                        is_directory = info.is_dir()
                        if is_directory:
                            if not (
                                name in allowed_directory_names
                                or name.startswith("data/raw/")
                                or name.startswith("data/processed/")
                            ):
                                raise ValueError("The backup contains an unexpected directory.")
                        elif not (
                            name == required_db
                            or name.startswith("data/raw/")
                            or name.startswith("data/processed/")
                        ):
                            raise ValueError("The backup contains an unexpected file.")

                        unix_mode = (info.external_attr >> 16) & 0xFFFF
                        file_type = unix_mode & 0o170000
                        allowed_types = (0, 0o100000, 0o040000)
                        if file_type not in allowed_types:
                            raise ValueError("The backup contains a non-regular file or symbolic link.")

                        if name == "data/raw/" or name.startswith("data/raw/"):
                            has_raw = True
                        if name == "data/processed/" or name.startswith("data/processed/"):
                            has_processed = True

                        total_uncompressed += info.file_size
                        if total_uncompressed > 1024 * 1024 * 1024:
                            raise ValueError("The backup is too large to restore (1 GB uncompressed limit).")

                    if not has_raw or not has_processed:
                        raise ValueError(
                            "The backup is incomplete. It must include both data/raw and data/processed."
                        )

                    for info in infos:
                        destination = temp_root.joinpath(*info.filename.rstrip("/").split("/"))
                        if info.is_dir():
                            destination.mkdir(parents=True, exist_ok=True)
                        else:
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            with archive.open(info) as source, destination.open("wb") as target:
                                shutil.copyfileobj(source, target)

                staged_data = temp_root / "data"
                staged_db = staged_data / "recognition_history.db"
                check = sqlite3.connect(str(staged_db))
                try:
                    check_result = check.execute("PRAGMA quick_check").fetchone()
                    if not check_result or check_result[0] != "ok":
                        raise ValueError("The backup database failed its integrity check.")
                    tables = {
                        row[0] for row in check.execute(
                            "SELECT name FROM sqlite_master WHERE type='table'"
                        )
                    }
                    if "events" not in tables:
                        raise ValueError("The backup database is not a valid FaceID database.")
                finally:
                    check.close()

                live_data = ROOT / "data"
                rollback = temp_root / "old_data"
                had_live_data = live_data.exists()
                if had_live_data:
                    shutil.copytree(live_data, rollback)

                try:
                    if live_data.exists():
                        shutil.rmtree(live_data)
                    shutil.copytree(staged_data, live_data)

                    try:
                        restored_e, restored_l = load_encodings()
                    except FileNotFoundError:
                        restored_e = np.empty((0, 128), dtype=np.float64)
                        restored_l = np.asarray([], dtype=str)

                    restored_e = np.asarray(restored_e, dtype=np.float64)
                    restored_l = np.asarray(restored_l)
                    if restored_e.size == 0:
                        restored_e = np.empty((0, 128), dtype=np.float64)
                    if restored_e.ndim != 2 or restored_e.shape[1] != 128:
                        raise ValueError("Restored face encodings have an invalid shape.")
                    if len(restored_e) != len(restored_l):
                        raise ValueError("Restored face encodings and labels do not match.")
                    if not np.isfinite(restored_e).all():
                        raise ValueError("Restored face encodings contain invalid numeric values.")

                    encodings, labels = restored_e, restored_l
                except Exception:
                    if live_data.exists():
                        shutil.rmtree(live_data)
                    if had_live_data and rollback.exists():
                        shutil.copytree(rollback, live_data)
                    try:
                        old_e, old_l = load_encodings()
                        encodings, labels = np.asarray(old_e), np.asarray(old_l)
                    except Exception:
                        encodings, labels = None, None
                    raise

            try:
                log_system_action(
                    "Backup restored",
                    "Restored database, registered photos and processed encodings"
                )
            except Exception as log_error:
                print("Backup restored, but activity logging failed:", log_error)
            flash(
                "Backup restored successfully. Check Registered Persons and test recognition before continuing.",
                "success"
            )
    except Exception as exc:
        flash(f"Restore failed: {exc}", "error")
    return redirect(url_for("system_management"))


@app.route('/lfw-evaluation')
def lfw_evaluation():
    import os
    import time
    import numpy as np
    from pathlib import Path
    from flask import render_template

    start_time = time.time()
    tolerance = 0.48
    desktop = ROOT.parent.parent / "Datasets"

    try:
        gallery = np.load(desktop / "lfw_gallery_encodings.npy")
        gallery_labels = np.load(desktop / "lfw_gallery_labels.npy")
        tests = np.load(desktop / "lfw_test_encodings.npy")
        test_labels = np.load(desktop / "lfw_test_labels.npy")

        if gallery.ndim != 2 or gallery.shape[1] != 128:
            raise ValueError("Invalid gallery encoding dimensions.")
        if tests.ndim != 2 or tests.shape[1] != 128:
            raise ValueError("Invalid test encoding dimensions.")
        if len(gallery) != len(gallery_labels):
            raise ValueError("Gallery labels do not match encodings.")
        if len(tests) != len(test_labels):
            raise ValueError("Test labels do not match encodings.")

        gallery_norms = np.sum(gallery ** 2, axis=1)
        best_indices = []
        best_distances = []

        for start in range(0, len(tests), 128):
            batch = tests[start:start + 128]
            distances_squared = (
                np.sum(batch ** 2, axis=1)[:, None]
                + gallery_norms[None, :]
                - 2 * np.dot(batch, gallery.T)
            )
            distances_squared = np.maximum(distances_squared, 0)
            indices = np.argmin(distances_squared, axis=1)

            best_indices.extend(indices.tolist())
            best_distances.extend(
                np.sqrt(distances_squared[np.arange(len(batch)), indices]).tolist()
            )

        best_indices = np.asarray(best_indices)
        best_distances = np.asarray(best_distances)

        known_mask = np.isin(test_labels, gallery_labels)
        accepted = best_distances <= tolerance
        predicted_labels = gallery_labels[best_indices]

        correct_known = np.sum(
            known_mask & accepted & (predicted_labels == test_labels)
        )

        known_total = int(np.sum(known_mask))
        unknown_total = int(np.sum(~known_mask))

        known_rejected = int(np.sum(known_mask & ~accepted))
        known_wrong = int(np.sum(
            known_mask & accepted & (predicted_labels != test_labels)
        ))
        unknown_rejected = int(np.sum(~known_mask & ~accepted))
        unknown_accepted = int(np.sum(~known_mask & accepted))

        total = len(tests)
        correct = int(correct_known + unknown_rejected)

        accuracy = correct / total * 100 if total else 0
        known_accuracy = correct_known / known_total * 100 if known_total else 0
        unknown_rejection = unknown_rejected / unknown_total * 100 if unknown_total else 0
        false_acceptance = unknown_accepted / unknown_total * 100 if unknown_total else 0

        elapsed = time.time() - start_time

        return render_template(
    "lfw_evaluation.html",
    gallery_count=len(gallery),
    total=total,
    known_total=known_total,
    unknown_total=unknown_total,
    tolerance=tolerance,
    accuracy=round(accuracy, 2),
    correct_known=int(correct_known),
    known_rejected=known_rejected,
    known_wrong=known_wrong,
    unknown_rejected=unknown_rejected,
    unknown_accepted=unknown_accepted,
    known_accuracy=round(known_accuracy, 2),
    unknown_rejection=round(unknown_rejection, 2),
    false_acceptance=round(false_acceptance, 2),
    elapsed=round(elapsed, 2)
)
    except Exception as exc:
        return f"LFW evaluation failed: {exc}", 500


@app.route('/model-comparison')
def model_comparison():
    from flask import render_template
    return render_template('model_comparison.html')


@app.route('/lfw-demo')
def lfw_demo():
    import os
    import base64
    import numpy as np
    from pathlib import Path
    from collections import defaultdict
    from flask import render_template

    desktop = ROOT.parent.parent / "Datasets"
    tolerance = 0.48

    try:
        gallery = np.load(desktop / "lfw_gallery_encodings.npy")
        gallery_labels = np.load(desktop / "lfw_gallery_labels.npy")
        tests = np.load(desktop / "lfw_test_encodings.npy")
        test_labels = np.load(desktop / "lfw_test_labels.npy")
        image_paths = np.load(
            desktop / "lfw_image_paths.npy",
            allow_pickle=True
        )

        original_labels = np.load(desktop / "lfw_labels.npy")

        if len(image_paths) != len(original_labels):
            raise ValueError("Image paths and original labels do not match.")

        # Recreate the exact train/test split from split_lfw.py.
        groups = defaultdict(list)

        for i, label in enumerate(original_labels):
            groups[str(label)].append(i)

        rng = np.random.default_rng(42)
        gallery_indices = []
        test_indices = []

        for person, indices in groups.items():
            indices = np.array(indices)
            rng.shuffle(indices)

            if len(indices) >= 2:
                gallery_indices.append(indices[0])
                test_indices.extend(indices[1:])
            else:
                test_indices.extend(indices)

        gallery_indices = np.array(gallery_indices)
        test_indices = np.array(test_indices)

        # Confirm reconstructed split matches saved evaluation files.
        if not np.array_equal(
            original_labels[gallery_indices], gallery_labels
        ):
            raise ValueError("Gallery split verification failed.")

        if not np.array_equal(
            original_labels[test_indices], test_labels
        ):
            raise ValueError("Test split verification failed.")

        if len(test_indices) != len(tests):
            raise ValueError("Test image count does not match.")

        known_indices = np.where(
            np.isin(test_labels, gallery_labels)
        )[0]

        unknown_indices = np.where(
            ~np.isin(test_labels, gallery_labels)
        )[0]

        demo_rng = np.random.default_rng(7)

        selected = np.concatenate([
            demo_rng.choice(known_indices, min(6, len(known_indices)), replace=False),
            demo_rng.choice(unknown_indices, min(6, len(unknown_indices)), replace=False)
        ])

        gallery_norms = np.sum(gallery ** 2, axis=1)
        results = []

        for test_index in selected:
            encoding = tests[test_index]

            distances_squared = (
                np.sum(encoding ** 2)
                + gallery_norms
                - 2 * np.dot(gallery, encoding)
            )

            distances_squared = np.maximum(distances_squared, 0)
            best_index = int(np.argmin(distances_squared))
            distance = float(np.sqrt(distances_squared[best_index]))

            predicted = str(gallery_labels[best_index])
            actual = str(test_labels[test_index])

            recognized = distance <= tolerance

            original_index = test_indices[test_index]
            image_path = str(image_paths[original_index])

            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(
                    image_file.read()
                ).decode("utf-8")

            results.append({
                "image": image_data,
                "actual": actual,
                "predicted": predicted if recognized else "Unknown",
                "distance": round(distance, 4),
                "status": "Recognized" if recognized else "Unknown",
                "correct": recognized and predicted == actual
            })

       
        return render_template(
    "lfw_demo.html",
    results=results,
    tolerance=tolerance,
    correct_count=sum(item["correct"] for item in results),
    unknown_count=sum(item["status"] == "Unknown" for item in results)
)


    except Exception as exc:
        return f"LFW demo failed: {exc}", 500



if __name__=="__main__":
    app.run(host="127.0.0.1",port=5000,debug=False,threaded=True)
