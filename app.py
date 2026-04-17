from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from color_engine import compare_hex_colors, classify_color

app = FastAPI(title="Paint Color Matcher", version="3.0.0")

_HTML = (Path(__file__).parent / "templates" / "index.html").read_text()


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(_HTML)


class ColorPair(BaseModel):
    hex_a: str
    hex_b: str


@app.post("/compare-colors")
async def compare_colors(body: ColorPair):
    try:
        result = compare_hex_colors(body.hex_a, body.hex_b)
        # enriquecer con clasificación ML si el modelo existe
        clf_a = classify_color(body.hex_a)
        clf_b = classify_color(body.hex_b)
        if "error" not in clf_a:
            result["ral_a"] = clf_a
        if "error" not in clf_b:
            result["ral_b"] = clf_b
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/classify")
async def classify(body: dict):
    hex_val = body.get("hex", "")
    try:
        return classify_color(hex_val)
    except ValueError as e:
        raise HTTPException(400, str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
