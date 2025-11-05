from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import torch
import tempfile
from inference import load_model_for_inference, predict

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model_for_inference("../final_pipe_model_v2.pth", device)

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        contents = await file.read()
        tmp.write(contents)
        temp_filename = tmp.name

    img_bytes, count = predict(model, temp_filename, device=device)

    # Return annotated image as PNG stream
    headers = {"X-Total-Pipes": str(count)}  # Custom header with count
    return StreamingResponse(img_bytes, media_type="image/png", headers=headers)
