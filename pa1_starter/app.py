import os
import uuid
from datetime import timezone
from pathlib import Path

import psycopg
from flask import Flask, render_template, request, redirect, make_response

app = Flask(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://boogle:boogle@localhost/boogle")

COOKIE_NAME = "boogle_id"
COOKIE_MAX_AGE = 60 * 60 * 24 * 365


def db():
    return psycopg.connect(DATABASE_URL)


def init_db():
    schema = (Path(__file__).parent / "schema_handout.sql").read_text()
    with db() as conn:
        conn.execute(schema)


def iso(ts):
    ts = ts.astimezone(timezone.utc)
    return ts.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ts.microsecond // 1000:03d}Z"


def get_client_id():
    raw = request.cookies.get(COOKIE_NAME)
    try:
        return str(uuid.UUID(raw))
    except (TypeError, ValueError):
        return None


def track(conn, client_id):
    conn.execute(
        """INSERT INTO clients (client_id) VALUES (%s)
           ON CONFLICT (client_id) DO UPDATE SET last_seen = now()""",
        (client_id,),
    )
    return conn.execute(
        """INSERT INTO requests (client_id, ip, referer, path)
           VALUES (%s, %s, %s, %s) RETURNING request_id""",
        (client_id, request.remote_addr, request.referrer, request.path),
    ).fetchone()[0]


def set_cookie(resp, client_id):
    resp.set_cookie(
        COOKIE_NAME,
        client_id,
        max_age=COOKIE_MAX_AGE,
        path="/",
        samesite="Lax",
        httponly=True,
    )


@app.get('/')
def index():
    client_id = get_client_id()
    is_new = client_id is None
    if is_new:
        client_id = str(uuid.uuid4())

    with db() as conn:
        track(conn, client_id)

    resp = make_response(render_template('boogle.html'))
    if is_new:
        set_cookie(resp, client_id)
    return resp


@app.get('/jsontest')
def jsontest():
    return {"data" : "I am in CSE190/CSE291!"}


@app.get('/search')
def search():
    query = request.args.get("q", "")
    client_id = get_client_id()
    is_new = client_id is None
    if is_new:
        client_id = str(uuid.uuid4())

    with db() as conn:
        request_id = track(conn, client_id)
        if query.strip():
            conn.execute(
                "INSERT INTO searches (client_id, request_id, query) VALUES (%s, %s, %s)",
                (client_id, request_id, query),
            )

    resp = redirect("/", code=303)
    if is_new:
        set_cookie(resp, client_id)
    return resp


def searches_for(conn, client_id):
    rows = conn.execute(
        """SELECT query, ts FROM searches
           WHERE client_id = %s
           ORDER BY ts DESC, search_id DESC""",
        (client_id,),
    ).fetchall()
    return [{"query": q, "ts": iso(ts)} for q, ts in rows]


@app.get('/api/history')
def history():
    client_id = get_client_id()
    if client_id is None:
        return {"client_id": None, "searches": []}

    with db() as conn:
        return {"client_id": client_id, "searches": searches_for(conn, client_id)}


@app.get('/dump')
def dump():
    clients = []
    with db() as conn:
        rows = conn.execute(
            "SELECT client_id, first_seen, last_seen FROM clients ORDER BY first_seen"
        ).fetchall()
        for client_id, first_seen, last_seen in rows:
            client_id = str(client_id)
            ip_rows = conn.execute(
                """SELECT host(ip), max(ts) AS last_ts FROM requests
                   WHERE client_id = %s AND ip IS NOT NULL
                   GROUP BY ip ORDER BY last_ts DESC""",
                (client_id,),
            ).fetchall()
            ips = [ip for ip, _ in ip_rows]
            clients.append({
                "client_id": client_id,
                "first_seen": iso(first_seen),
                "last_seen": iso(last_seen),
                "latest_ip": ips[0] if ips else None,
                "ips": ips,
                "searches": searches_for(conn, client_id),
            })
    return {"clients": clients}


init_db()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8000")
