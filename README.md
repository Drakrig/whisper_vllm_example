# Whisper with VLLM

The repository holds a basic example of how to use Whisper ASR model with VLLM engine. 

Check `step_by_step.ipynb` notebook for more detailed explanation. 

Information was scrapped all across Internet, but it should provide good enought base to understand how to use VLLM and Whisper to achieve quite impressive perfomance. With RTX 2080 Mobile 8Gb, it takes literally couple seconds to process 2 min audio which is huge boost, compare to native Transfomers realization.

## How to run example

1. Install requrements
```pip install -r requirements.txt```
2. Place your wavs into samples folder OR do not forget to pass their folder as argument
3. Run
```python3 main.py```

### Arguments

* `--model` - Model name, default value `openai/whisper-large-v3-turbo`
* `--gpu_memory_utilization` - GPU memory utilization, default value `0.55`
* `--dtype` - Data type, default value `float16`
* `--max_num_seqs` - Max number of sequences (batch size), default value `4`
* `--sample_path` - Path to the audio samples, default value `samples`
* `--language` - Language code, default value `en`

## Sources
1. Official VLLM encoder decoder example [Link](https://docs.vllm.ai/en/stable/getting_started/examples/encoder_decoder_multimodal.html)
2. HuggingFace Whisper VLLM endpoint example [Link](https://huggingface.co/hfendpoints-images/whisper-vllm-gpu/blob/main/handler.py)