"""FastAPI application: thin HTTP layer over the service module.

Endpoints
---------
GET  /api/health                 liveness probe
POST /api/dft                    windowed / zero-padded spectra
POST /api/idft                   spectrum back to time domain
POST /api/windows                window coefficients + figures of merit
POST /api/filter                 frequency-selective filtering
POST /api/sampling               sampling-theorem / aliasing demonstration

All expected client errors are raised as ``BadRequest`` in the service layer
and serialized here as HTTP 400 with ``{"detail": "..."}``.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import services
from .errors import BadRequest
from .schemas import (
    DftRequest,
    DftResponse,
    FilterRequest,
    FilterResponse,
    IdftRequest,
    IdftResponse,
    SamplingRequest,
    SamplingResponse,
    WindowRequest,
    WindowResponse,
)

app = FastAPI(
    title="Fourier Teaching Tool API",
    version="1.0.0",
    description="Independent, test-guarded arithmetic backend for the "
    "browser-based Fourier transform teaching tool.",
)

# The frontend is served by a separate container/origin during development;
# in production nginx proxies /api so this stays harmless.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(BadRequest)
async def bad_request_handler(_: Request, exc: BadRequest) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/api/dft", response_model=DftResponse)
async def post_dft(req: DftRequest) -> dict:
    return services.dft_service(req)


@app.post("/api/idft", response_model=IdftResponse)
async def post_idft(req: IdftRequest) -> dict:
    return {"signal": services.idft_service(req)}


@app.post("/api/windows", response_model=WindowResponse)
async def post_windows(req: WindowRequest) -> dict:
    return {"windows": services.windows_service(req)}


@app.post("/api/filter", response_model=FilterResponse)
async def post_filter(req: FilterRequest) -> dict:
    return services.filter_service(req)


@app.post("/api/sampling", response_model=SamplingResponse)
async def post_sampling(req: SamplingRequest) -> dict:
    return services.sampling_service(req)
