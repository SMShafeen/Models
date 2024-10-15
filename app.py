from huggingface_hub import login

login("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")

from flask import Flask, request, jsonify
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline
import torch
import os

app = Flask(__name__)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    # max_new_tokens=500,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=15,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

bart_model_id = "facebook/bart-large-cnn"
bart_tokenizer = AutoTokenizer.from_pretrained(bart_model_id)
bart_model = AutoModelForSeq2SeqLM.from_pretrained(bart_model_id)
summarizer = pipeline("summarization", model=bart_model, tokenizer=bart_tokenizer)

# llama_model_id = "meta-llama/Llama-2-7b-chat-hf"
# llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_id)
# llama_model = AutoModelForCausalLM.from_pretrained(llama_model_id, torch_dtype=torch_dtype).to(device)
# llama_summarizer = pipeline("text-generation", model=llama_model, tokenizer=llama_tokenizer, max_new_tokens=150, device=device)

@app.route('/')
def home():
    return "Whisper ASR API is running!"

@app.route('/transcribe-and-summarize', methods=['POST'])
def transcribe_and_summarize():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files['audio']
    file_path = os.path.join('uploads', audio_file.filename)
    audio_file.save(file_path)

    transcription_result = asr_pipeline(file_path, generate_kwargs={"task": "translate"})
    transcription_text = transcription_result['text']
    transcription_chunks = transcription_result['chunks']

    os.remove(file_path)

    summary_result = summarizer(transcription_text, max_length=250, min_length=30, do_sample=False)
    summary_text = summary_result[0]['summary_text']

    # llama_summary_result = llama_summarizer(transcription_text)
    # llama_summary_text = llama_summary_result[0]['generated_text']

    return jsonify({"transcription": transcription_text, "summary": summary_text})

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)

    app.run(host='0.0.0.0', port=5000)