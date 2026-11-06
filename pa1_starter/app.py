import json
import math
import os
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import timezone
from pathlib import Path

import psycopg
import requests
import requests.adapters
from flask import Flask, render_template, request, redirect, make_response

from classify import ATTRIBUTES, load_serving_models, user_text

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

    resp = make_response(render_template('new_boogle.html'))
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


# ---------------------------------------------------------------------------
# Supply-side platform
# ---------------------------------------------------------------------------

RTB_URL = os.environ.get("RTB_URL", "https://webtracking.sysnet.ucsd.edu/pa1/rtb/").rstrip("/")
DSPS = ["steady", "niche", "shady", "glitch"]
IMP_ID = "1"
AD_W, AD_H = 300, 250
BID_FLOOR = float(os.environ.get("BID_FLOOR", "0.50"))
TMAX_MS = 250

# Only claim a segment when the model is at least this confident in it.
CONFIDENCE = float(os.environ.get("SEGMENT_CONFIDENCE", "0.7"))

# Block an advertiser domain / creative once it has drawn at least
# BLOCK_MIN_COMPLAINTS complaints and complaints make up BLOCK_RATE of its wins.
BLOCK_MIN_COMPLAINTS = 2
BLOCK_RATE = 0.2

SEGMENTS = {
    (seg["attribute"], seg["value"]): seg["id"]
    for seg in json.loads((Path(__file__).parent / "segments.json").read_text())["segments"]
}
MODELS = load_serving_models()
POOL = ThreadPoolExecutor(max_workers=32)

# One keep-alive session shared by all threads (urllib3's pool is thread-safe),
# so DSP calls reuse warm connections instead of paying a TLS handshake each.
SESSION = requests.Session()
SESSION.mount("https://", requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=32))
SESSION.mount("http://", requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=32))


def infer_segments(searches, ua):
    """Segment ids for the attributes the Part 3 model is confident about."""
    if not searches:
        return []
    text = [user_text(searches, ua)]
    segments = []
    for attr in ATTRIBUTES:
        model = MODELS[attr]
        probs = model.predict_proba(text)[0]
        best = probs.argmax()
        seg = SEGMENTS.get((attr, model.classes_[best]))
        if seg and probs[best] >= CONFIDENCE:
            segments.append(seg)
    return segments


def blocklist(conn):
    """Advertiser domains and creative ids whose complaints outweigh their wins."""
    def blocked(key_sql):
        rows = conn.execute(f"""
            SELECT k, count(DISTINCT a.auction_id), count(c.complaint_id)
            FROM auctions a CROSS JOIN LATERAL {key_sql} AS k
            LEFT JOIN complaints c ON c.auction_id = a.auction_id
            WHERE a.seat IS NOT NULL
            GROUP BY k""").fetchall()
        return {k for k, wins, n in rows
                if n >= BLOCK_MIN_COMPLAINTS and n >= BLOCK_RATE * wins}

    return blocked("unnest(a.adomain)"), blocked("(SELECT a.crid)")


def ask_dsp(dsp, payload, deadline):
    """POST one bid request; returns the parsed body, or None for no/late/bad bids."""
    try:
        resp = SESSION.post(
            f"{RTB_URL}/{dsp}/bid", data=payload, timeout=TMAX_MS / 1000,
            headers={"Content-Type": "application/json"},
        )
    except requests.RequestException:
        return None
    if resp.status_code != 200 or time.monotonic() > deadline:
        return None
    try:
        return json.loads(resp.content)
    except ValueError:
        return None


def is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def valid_bids(response, blocked_domains, blocked_crids):
    """Yield the bids in one DSP response that pass every filter."""
    if not isinstance(response, dict) or not isinstance(response.get("seatbid"), list):
        return
    if response.get("cur", "USD") != "USD":
        return
    for seatbid in response["seatbid"]:
        if not isinstance(seatbid, dict) or not isinstance(seatbid.get("bid"), list):
            continue
        for bid in seatbid["bid"]:
            if not isinstance(bid, dict):
                continue
            price = bid.get("price")
            adomain = bid.get("adomain") or []
            if not (
                bid.get("impid") == IMP_ID
                and bid.get("w") == AD_W and bid.get("h") == AD_H
                and is_number(price) and price >= BID_FLOOR
                and isinstance(bid.get("adm"), str) and bid["adm"]
                and isinstance(bid.get("burl"), str)
                and isinstance(adomain, list)
            ):
                continue
            if blocked_domains.intersection(map(str, adomain)) or bid.get("crid") in blocked_crids:
                continue
            yield seatbid.get("seat"), bid


def bill(burl, price):
    try:
        SESSION.get(burl.replace("${AUCTION_PRICE}", f"{price:.2f}"), timeout=2)
    except requests.RequestException:
        pass


AD_PAGE = ('<!doctype html><html><head><meta charset="utf-8">'
           '<style>html,body{margin:0;overflow:hidden}</style></head><body>%s</body></html>')


@app.get('/ad')
def ad():
    client_id = get_client_id()
    ua = request.headers.get("User-Agent", "")
    auction_id = uuid.uuid4().hex[:16]

    with db() as conn:
        searches = searches_for(conn, client_id) if client_id else []
        blocked_domains, blocked_crids = blocklist(conn)

    segments = infer_segments(searches, ua)
    bid_request = {
        "id": auction_id,
        "imp": [{"id": IMP_ID,
                 "banner": {"format": [{"w": AD_W, "h": AD_H}]},
                 "bidfloor": BID_FLOOR}],
        "device": {"ua": ua, "ip": request.remote_addr},
        "user": {"id": client_id or str(uuid.uuid4()),
                 "data": [{"id": "boogle",
                           "segment": [{"id": seg} for seg in segments],
                           "ext": {"segtax": 500}}]},
        "tmax": TMAX_MS,
        "badv": sorted(blocked_domains),
    }
    payload = json.dumps(bid_request).encode()

    deadline = time.monotonic() + TMAX_MS / 1000
    futures = [POOL.submit(ask_dsp, dsp, payload, deadline) for dsp in DSPS]
    done, _ = wait(futures, timeout=TMAX_MS / 1000)

    bids = [b for f in done
            for b in valid_bids(f.result(), blocked_domains, blocked_crids)]
    random.shuffle(bids)  # break ties fairly
    bids.sort(key=lambda sb: sb[1]["price"], reverse=True)

    winner = None
    if bids:
        seat, winner = bids[0]
        second = bids[1][1]["price"] if len(bids) > 1 else BID_FLOOR
        price = round(min(winner["price"], max(second, BID_FLOOR) + 0.01), 2)

    with db() as conn:
        conn.execute(
            """INSERT INTO auctions (auction_id, client_id, segments, seat, bid_id,
                                     crid, adomain, bid_price, clear_price)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (auction_id, client_id, segments,
             seat if winner else None,
             str(winner.get("id")) if winner else None,
             str(winner.get("crid")) if winner and winner.get("crid") else None,
             [str(d) for d in winner.get("adomain") or []] if winner else None,
             winner["price"] if winner else None,
             price if winner else None),
        )

    if winner is None:
        return "", 200

    bill(winner["burl"], price)
    return AD_PAGE % winner["adm"]


@app.post('/ad/feedback')
def ad_feedback():
    data = request.get_json(silent=True, force=True) or {}
    auction_id = data.get("auction_id")
    if data.get("event") != "complaint" or not isinstance(auction_id, str):
        return {"ok": False}, 400

    with db() as conn:
        conn.execute("INSERT INTO complaints (auction_id) VALUES (%s)", (auction_id,))
    return {"ok": True}


init_db()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8000")
