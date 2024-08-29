from pathlib import Path

import torch
from PIL import Image
from streamlit.runtime.uploaded_file_manager import UploadedFile
from torch import Tensor
from transformers import (
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)


def readImage(image: Path | UploadedFile) -> Image.Image:
    if image is UploadedFile:
        return Image.open(fp=image.getbuffer()).convert(mode="RGB")
    else:
        return Image.open(fp=image).convert(mode="RGB")


def llava(img: Image.Image, device: str) -> str:
    processor = LlavaNextProcessor.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf",
    )

    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.to(device)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": "Return only the text of the image. Do not add any additional text.",  # noqa: E501
                },
            ],
        },
    ]
    prompt = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
    )
    inputs = processor(prompt, img, return_tensors="pt").to(device)

    output = model.generate(**inputs, max_new_tokens=100)

    response: str = processor.decode(output[0], skip_special_tokens=True)
    return response.split("[/INST]")[-1].strip(" ")


def trocr(img: Image.Image, device: str) -> str:
    processor: TrOCRProcessor = TrOCRProcessor.from_pretrained(
        "microsoft/trocr-base-handwritten",
    )

    model: VisionEncoderDecoderModel = (
        VisionEncoderDecoderModel.from_pretrained(  # noqa: E501
            "microsoft/trocr-base-handwritten",
        )
    )

    model.to(device)

    pixel_values: Tensor = processor(img, return_tensors="pt").pixel_values

    generated_ids: Tensor = model.generate(pixel_values, max_new_tokens=100)

    generated_text: str = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]

    return generated_text
