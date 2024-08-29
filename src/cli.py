import warnings
from pathlib import Path

import click
import torch
from PIL import Image
from torch import Tensor
from transformers import (  # DonutProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    pipeline,
)

warnings.filterwarnings(action="ignore")
torch.random.manual_seed(42)


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

    # autoregressively complete prompt
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


def phi(text: str, device: str, language: str = "python") -> str:
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3.5-mini-instruct",
        device_map=device,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/Phi-3.5-mini-instruct",
    )

    messages = [
        {"role": "system", "content": f"Generate {language} code"},
        {"role": "user", "content": text},
    ]

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
    }

    output = pipe(messages, **generation_args)
    return output[0]["generated_text"]


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
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    img: Image.Image = Image.open(fp=inputPath).convert(mode="RGB")

    imgText: str
    match ocrModel.lower():
        case "trocr":
            imgText = trocr(img=img, device=device)
        case "llava":
            imgText = llava(img=img, device=device)
        case _:
            print("Lol")
            quit()

    print("\n===\n", imgText)

    code: str
    match codeModel:
        case "phi":
            code = phi(text=imgText, device=device, language=progLanguage)
        case _:
            print("Lol 2")
            quit()

    print("\n===\n", code)


if __name__ == "__main__":
    main()
