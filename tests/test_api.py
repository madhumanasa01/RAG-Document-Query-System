DOC = (
    "Acme Router X200 user guide. The X200 supports Wi-Fi 6 and has four gigabit ports. "
    "The warranty period for the X200 is 24 months from the date of purchase. "
    "To reset the router, hold the reset button for ten seconds."
)


def _upload(client, name="guide.txt", content=DOC):
    return client.post("/documents", files={"file": (name, content.encode(), "text/plain")})


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_upload_and_list(client):
    res = _upload(client)
    assert res.status_code == 201
    assert res.json()["chunks"] >= 1
    docs = client.get("/documents").json()
    assert docs[0]["filename"] == "guide.txt"


def test_query_returns_grounded_answer_with_sources(client):
    _upload(client)
    res = client.post("/query", json={"question": "What is the warranty period for the X200?"})
    body = res.json()
    assert res.status_code == 200
    assert body["mode"] == "extractive"
    assert "24 months" in body["answer"]
    assert body["sources"][0]["filename"] == "guide.txt"


def test_query_with_no_documents(client):
    body = client.post("/query", json={"question": "anything at all?"}).json()
    assert body["mode"] == "none"
    assert body["sources"] == []


def test_unsupported_file_type(client):
    assert _upload(client, name="data.exe").status_code == 415


def test_delete_document(client):
    doc_id = _upload(client).json()["doc_id"]
    assert client.delete(f"/documents/{doc_id}").status_code == 200
    assert client.delete(f"/documents/{doc_id}").status_code == 404
