from transformers import MBartTokenizer, MBartForConditionalGeneration

MODEL_NAME = "ARTeLab/mbart-summarization-fanpage"
tokenizer = MBartTokenizer.from_pretrained(MODEL_NAME)
model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)

def paraphrase_text(text: str, max_length: int = 50, min_length: int = 15) -> str:
    # Prompt trick: "riformula" for paraphrasing
    input_text = "riformula: " + text

    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=5,
        length_penalty=1.5,
        max_length=max_length,
        min_length=min_length,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    lyrics = (
        "Ti voglio bene assai Ma tanto, tanto bene, sai È una catena ormai Che scioglie il sangue nelle vene, sai."
    )

    print("🎵 Original:\n", lyrics)
    print("\n🔁 Paraphrase:\n", paraphrase_text(lyrics))
