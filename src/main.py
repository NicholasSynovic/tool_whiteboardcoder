import re

import streamlit as st
import torch
from PIL import Image
from streamlit.runtime.uploaded_file_manager import UploadedFile
from torch import Tensor
from transformers import (
    DonutProcessor,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)


def trOCR_inferenceImage(uf: UploadedFile) -> str:
    processor: TrOCRProcessor = TrOCRProcessor.from_pretrained(
        "microsoft/trocr-base-handwritten",
    )

    model: VisionEncoderDecoderModel = (
        VisionEncoderDecoderModel.from_pretrained(  # noqa: E501
            "microsoft/trocr-base-handwritten",
        )
    )

    image: Image.Image = Image.open(fp=uf).convert("RGB")
    pixel_values: Tensor = processor(image, return_tensors="pt").pixel_values
    generated_ids: Tensor = model.generate(pixel_values)

    generated_text: str = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]

    return generated_text


def donut_inferenceImage(uf: UploadedFile) -> str:
    processor = DonutProcessor.from_pretrained(
        "naver-clova-ix/donut-base-finetuned-cord-v2"
    )
    model = VisionEncoderDecoderModel.from_pretrained(
        "naver-clova-ix/donut-base-finetuned-cord-v2"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    # load document image
    image: Image.Image = Image.open(fp=uf).convert("RGB")

    # prepare decoder inputs
    task_prompt = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids

    pixel_values = processor(image, return_tensors="pt").pixel_values

    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(
        processor.tokenizer.pad_token, ""
    )
    sequence = re.sub(
        r"<.*?>", "", sequence, count=1
    ).strip()  # remove first task start token
    print(processor.token2json(sequence))


def main() -> None:
    st.markdown(body="# Whiteboard Code")
    st.markdown(body="> Tool to convert written psuedo-code into actual code!")
    st.divider()

    st.markdown("## Image Picker")
    uploaded_image: UploadedFile = st.file_uploader(
        label="Choose an image",
        type=["jpg", "png"],
    )
    st.divider()

    if uploaded_image is not None:
        st.markdown("## Image")
        st.image(uploaded_image)

        ocrText: str = donut_inferenceImage(uf=uploaded_image)

        st.markdown("## OCR Text")
        ocrText = st.code(
            body=ocrText,
            language=None,
            line_numbers=True,
            wrap_lines=True,
        )


if __name__ == "__main__":
    main()
