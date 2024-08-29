import warnings
from pathlib import Path

import click
from PIL import Image

from src.models import llava, phi, trocr

warnings.filterwarnings(action="ignore")


def readImage(image: Path) -> Image.Image:
    return Image.open(fp=image).convert(mode="RGB")
    # if image is UploadedFile:
    #     return Image.open(fp=image.getbuffer()).convert(mode="RGB")
    # else:
    # return Image.open(fp=image).convert(mode="RGB")


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
        path_type=Path,
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
@click.option(
    "--code-model",
    "codeModel",
    type=click.Choice(choices=["phi"], case_sensitive=False),
    required=True,
    help="Code generation model to use",
)
@click.option(
    "--language",
    "progLanguage",
    type=str,
    required=False,
    default="Python",
    show_default=True,
    help="Programming language to generate code from image",
)
def main(
    inputPath: Path,
    ocrModel: str,
    codeModel: str,
    progLanguage: str,
) -> None:
    img: Image.Image = readImage(image=inputPath)

    imgText: str
    match ocrModel.lower():
        case "trocr":
            imgText = trocr(img=img)
        case "llava":
            imgText = llava(img=img)
        case _:
            print("Lol")
            quit()

    print("\n===\n", imgText)

    code: str
    match codeModel:
        case "phi":
            code = phi(text=imgText, language=progLanguage)
        case _:
            print("Lol 2")
            quit()

    print("\n===\n", code)


if __name__ == "__main__":
    main()
