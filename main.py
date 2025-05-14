from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import time
from vllm import LLM, SamplingParams
from transformers import WhisperTokenizerFast
from pathlib import Path
from librosa import resample, load

import numpy as np

def chunking(audio:np.ndarray, sample_rate:int):
    """Split audio to 30 second duration chunks

    Args:
        audio (np.ndarray): Audio data
        sample_rate (int): Audio sample rate
    """
    max_duration_samples = sample_rate * 30.0
    padding = max_duration_samples - np.remainder(len(audio), max_duration_samples)
    audio = np.pad(audio, (0, padding.astype(int)), 'constant', constant_values=0.0)
    return np.split(audio, len(audio) // max_duration_samples)

if __name__ == "__main__":
    parser = ArgumentParser(
        description="VLLM Whisper Example",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/whisper-large-v3-turbo",
        help="Model name",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.55,
        help="GPU memory utilization",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        help="Data type",
    )
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=4,
        help="Max number of sequences (batch size)",
    )
    parser.add_argument(
        "--sample_path",
        type=str,
        default="samples",
        help="Path to the audio samples",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language code",
    )
    
    args = parser.parse_args()

    tokenizer = WhisperTokenizerFast.from_pretrained(args.model, language=args.language)
    
    lang_code = tokenizer.convert_tokens_to_ids(args.language)
    if lang_code == 50257:
        raise ValueError(f"Language code for {args.language} not found")

    whisper = LLM(
        model=args.model,
        limit_mm_per_prompt={"audio": 1},
        gpu_memory_utilization = args.gpu_memory_utilization,
        dtype = args.dtype,
        max_num_seqs = args.max_num_seqs,
        max_num_batched_tokens=448
    )

    audio_files = Path(args.sample_path).glob("*.wav")

    samples = {}

    for file in list(audio_files):
        # Load the audio file
        audio, sample_rate = load(file,sr=16000)
        if sample_rate != 16000:
            audio = resample(audio.numpy().astype(np.float32), orig_sr=sample_rate, target_sr=16000)
        print(f"File: {file}, Sample rate: {sample_rate}, Audio shape: {audio.shape}, Duration: {audio.shape[0] / sample_rate:.2f} seconds")
        chunks = chunking(audio, 16000)
        samples[file.stem] = [(chunk,16000) for chunk in chunks]

    for file, chunks in samples.items():
        
        prompts = [{
                    "encoder_prompt": {
                        "prompt": "",
                        "multi_modal_data": {
                            "audio": chunk,
                        },
                    },
                    "decoder_prompt":
                    f"<|startoftranscript|><|{args.language}|><|transcribe|><|notimestamps|>"
                } for chunk in chunks]
        print(f"File: {file}, Chunks: {len(chunks)}")
        # Create a sampling params object.
        sampling_params = SamplingParams(
            temperature=0,
            top_p=1.0,
            max_tokens=8192,
        )

        start = time.time()

        # Inferense based on max_num_seqs
        outputs = []
        for i in range(0, len(prompts), args.max_num_seqs):
            output = whisper.generate(prompts[i:i+args.max_num_seqs], sampling_params=sampling_params)
            outputs.extend(output)
        # Print the outputs.
        generated = " ".join([output.outputs[0].text for output in outputs])
        for output in outputs:
            prompt = output.prompt
            encoder_prompt = output.encoder_prompt
            generated_text = output.outputs[0].text
            print(f"Encoder prompt: {encoder_prompt!r}, "
                f"Decoder prompt: {prompt!r}, "
                f"Generated text: {generated_text!r}")

        duration = time.time() - start

        print("Duration:", duration)
        print("RPS:", len(prompt) / duration)
        print("Generated text:", generated)