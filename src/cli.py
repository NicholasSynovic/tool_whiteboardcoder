from pathlib import Path

import click
from PIL import Image
from torch import Tensor
from transformers import (  # DonutProcessor,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)


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

    generated_ids: Tensor = model.generate(pixel_values)

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
    type=click.Choice(choices=["trocr", "donut"], case_sensitive=False),
    required=True,
    help="OCR (optical charachter recognition) model to use",
)
def main(inputPath: Path, ocrModel: str) -> None:
    img: Image.Image = Image.open(fp=inputPath).convert(mode="RGB")

    text: str
    match ocrModel.lower():
        case "trocr":
            text = trocr(img=img)
        case _:
            print("Lol")
            quit()

    print(text)


if __name__ == "__main__":
    main()
