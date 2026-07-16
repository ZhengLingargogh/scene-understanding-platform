from typing import List

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.deps import get_model_service
from app.schemas.model import ModelCreate, ModelResponse, ModelRuntimeStatus, ModelUpdate
from app.services.model_service import ModelService

router = APIRouter()


@router.get("", response_model=List[ModelResponse])
async def list_models(service: ModelService = Depends(get_model_service)):
    return service.list_models()


@router.post("", response_model=ModelResponse, status_code=status.HTTP_201_CREATED)
async def create_model(
    payload: ModelCreate,
    service: ModelService = Depends(get_model_service),
):
    return service.create_model(payload)


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(model_id: str, service: ModelService = Depends(get_model_service)):
    model = service.get_model(model_id)
    if model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")
    return model


@router.get("/{model_id}/status", response_model=ModelRuntimeStatus)
async def get_model_status(model_id: str, service: ModelService = Depends(get_model_service)):
    try:
        return service.get_runtime_status(model_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found") from None


@router.post("/{model_id}/load", response_model=ModelResponse)
async def load_model(model_id: str, service: ModelService = Depends(get_model_service)):
    try:
        return service.load_model(model_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found") from None
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/{model_id}/unload", response_model=ModelResponse)
async def unload_model(model_id: str, service: ModelService = Depends(get_model_service)):
    try:
        return service.unload_model(model_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found") from None


@router.patch("/{model_id}", response_model=ModelResponse)
async def update_model(
    model_id: str,
    payload: ModelUpdate,
    service: ModelService = Depends(get_model_service),
):
    model = service.update_model(model_id, payload)
    if model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")
    return model


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(model_id: str, service: ModelService = Depends(get_model_service)):
    if not service.delete_model(model_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")
