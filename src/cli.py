import warnings
from pathlib import Path

import click
import torch
from PIL import Image
from torch import Tensor
from transformers import (  # DonutProcessor,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

warnings.filterwarnings(action="ignore")


def llava(img: Image.Image) -> str:
    processor = LlavaNextProcessor.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf",
    )

    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.to("cuda")

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": "Return only the content of the image",
                },
            ],
        },
    ]
    prompt = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
    )
    inputs = processor(prompt, img, return_tensors="pt").to("cuda:0")

    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=100)

    print(processor.decode(output[0], skip_special_tokens=True))


def trocr(img: Image.Image) -> str:
    processor: TrOCRProcessor = TrOCRProcessor.from_pretrained(
        "microsoft/trocr-base-handwritten",
    )

    model: VisionEncoderDecoderModel = (
        VisionEncoderDecoderModel.from_pretrained(  # noqa: E501
            "microsoft/trocr-base-handwritten",
        )
    )

    pixel_values: Tensor = processor(img, return_tensors="pt").pixel_values

    generated_ids: Tensor = model.generate(pixel_values, max_new_tokens=100)

    generated_text: str = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]

    return generated_text


@click.command()
@click.option(
    "-i",
    "--input",
    "inputPath",
    type=click.Path(
        exists=True,
        file_okay=True,
        readable=True,
        resolve_path=True,
    ),
    required=True,
    help="Path to image file to analyze",
)
@click.option(
    "--ocr-model",
    "ocrModel",
    type=click.Choice(choices=["trocr", "llava"], case_sensitive=False),
    required=True,
    help="OCR (optical charachter recognition) model to use",
)
def main(inputPath: Path, ocrModel: str) -> None:
    img: Image.Image = Image.open(fp=inputPath).convert(mode="RGB")

    text: str
    match ocrModel.lower():
        case "trocr":
            text = trocr(img=img)
        case "llava":
            text = llava(img=img)
        case _:
            print("Lol")
            quit()

    print(text)


if __name__ == "__main__":
    main()
