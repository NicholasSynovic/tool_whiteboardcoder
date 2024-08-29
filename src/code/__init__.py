from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


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
