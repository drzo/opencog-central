import sys
import os
sys.path.insert(0, os.path.abspath('.'))
from NanoCog.nctalk import ModelConfig

# Initialize the model
model_config = ModelConfig("../nanoGPT/out-nanocog-ci/ckpt.pt", device="cpu")
model_config.load_model()

# Test simple generation
prompt = "User: Explain cognitive synergy in CogPrime.\nNanoCog: "
response = model_config.generate(prompt, max_new_tokens=50, temperature=0.8)
print("Response:", response)

# Test with callback
def callback(token):
    print(token, end="", flush=True)
    return True

print("\nStreaming response:")
response = model_config.generate(prompt, max_new_tokens=50, temperature=0.8, callback=callback)
print("\nDone!")