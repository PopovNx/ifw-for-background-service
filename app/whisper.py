import os
import torch
from transformers import pipeline,AutoModelForSpeechSeq2Seq,AutoProcessor
from .diarization_pipeline import diarize

DIARIZATION = os.environ.get("DIARIZATION", "False").lower() == "true"
MODEL = os.environ.get("MODEL", "openai/whisper-large-v3")

device = "cuda:0"
torch_dtype = torch.float16
model_id = MODEL
    
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, device_map=device, attn_implementation="flash_attention_2")
processor = AutoProcessor.from_pretrained(MODEL)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    return_timestamps=True,
    batch_size=38,
    chunk_length_s=25,
    return_language=True
)


def process_audio_file(file_path: str, batch_size: int, timestamp: str):
    output = pipe(
        file_path,
        generate_kwargs={"task": "transcribe"},
        return_timestamps="word" if timestamp == "word" else True,
    )
    if DIARIZATION:
        speakers_transcript = diarize(
            file_path,
            output,
        )
        output["speakers"] = speakers_transcript

    return output
