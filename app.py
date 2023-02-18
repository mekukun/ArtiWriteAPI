from auth_token import key
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
import torch
from diffusers import StableDiffusionPipeline
from io import BytesIO
import base64 

app = FastAPI()

app.add_middleware(
    CORSMiddleware, 
    allow_credentials=True, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

# Anime style
# - circulus/sd-anireal-v2.5
# - andite/anything-v4.0
# - aipicasso/cool-japan-diffusion-2-1-0
# - johnslegers/epic-diffusion
# - hakurei/waifu-diffusion
# - nitrosocke/Arcane-Diffusion

# Non-anime style
# - nitrosocke/Nitro-Diffusion (Disney-ish)
# - dreamlike-art/dreamlike-photoreal-2.0
# - nitrosocke/Future-Diffusion ("future style ...")
# - johnslegers/epic-diffusion

device = "cuda"
model_id = "circulus/sd-anireal-v2.5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_auth_token=key)
pipe.to(device)
pipe.safety_checker = None

@app.get("/")
def generate(prompt: str):
    image = pipe(prompt).images[0]

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    imgstr = base64.b64encode(buffer.getvalue())

    return Response(content=imgstr, media_type="image/png")