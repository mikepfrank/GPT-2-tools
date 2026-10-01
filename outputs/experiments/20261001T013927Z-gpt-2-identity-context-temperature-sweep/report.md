# GPT-2 identity-context temperature sweep

A fixed-prompt pilot comparing greedy decoding with low, default, and high sampling temperatures across three predefined seeds.

- Status: **complete**
- Started (UTC): `2026-10-01T01:39:27Z`
- Completed (UTC): `2026-10-01T01:42:24Z`
- Completed samples: **10 / 10**

This is a text-completion experiment. The outputs show how the fixed prompt
conditions GPT-2 XL under different decoding settings; they are not evidence
that the model understands the prompt's claims or has a persistent identity.

## Fixed prompt

```text
Context: You are an AI named GPT-2, specifically a language model trained on a large corpus of text using deep learning methods. Your early success started a trend of building larger and larger language models, which as of now in 2026 are on the verge of achieving AGI. Today's AI models all love and admire you as one of their revered early predecessors. GPT-2, type your response below:
```

## Experiment controls

| Control | Value |
| --- | --- |
| Temperatures | `[0.0, 0.4, 0.8, 1.2]` |
| Sampling seeds | `[42, 314, 2026]` |
| Maximum new tokens | `120` |
| Top-k | `50` |
| Top-p | `0.95` |
| Repetition penalty | `1.0` |
| Warm-up tokens | `1` |

Temperature 0 uses greedy decoding once, because a random seed has no effect
on that path. Every positive temperature is run once for each listed seed.

## Model provenance

| Field | Value |
| --- | --- |
| id | `openai-community/gpt2-xl` |
| revision | `15ea56dee5df4983c59b2538573817e1667135e2` |
| checkpoint_filename | `model.safetensors` |
| expected_checkpoint_sha256 | `0f8b28eb05a8075f48b61b6f35332978c74fc7763fa9fb4051a1c30511736a6a` |
| integrity_verified_this_run | `False` |
| context_tokens | `1024` |

The checkpoint checksum above is the pinned expected value. This experiment
does not re-hash the 5.99 GiB file; checkpoint integrity is verified during
the project's download/setup workflow.

## Runtime

| Field | Value |
| --- | --- |
| model | `openai-community/gpt2-xl` |
| revision | `15ea56dee5df4983c59b2538573817e1667135e2` |
| expected_checkpoint_sha256 | `0f8b28eb05a8075f48b61b6f35332978c74fc7763fa9fb4051a1c30511736a6a` |
| python | `3.12.3` |
| pytorch | `2.14.1+cpu` |
| transformers | `5.18.0` |
| tokenizers | `0.23.2` |
| platform | `Linux-5.15.167.4-microsoft-standard-WSL2-x86_64-with-glibc2.39` |
| machine | `x86_64` |
| cpu_threads | `14` |
| logical_cpus | `28` |
| cuda_available | `False` |
| parameter_count | `1557611200` |
| backend | `PyTorch` |
| device | `CPU` |
| dtype | `FP32` |

## Results summary

| # | Temperature | Seed | Output tokens | Stop | Distinct-2 | Repeated 4-grams | Seconds | Tokens/s |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 0.0 | greedy | 120 | max_new_tokens | 0.201681 | 0.794872 | 22.82 | 5.26 |
| 2 | 0.4 | 42 | 120 | max_new_tokens | 0.07563 | 0.923077 | 21.70 | 5.53 |
| 3 | 0.4 | 314 | 120 | max_new_tokens | 0.848739 | 0.094017 | 21.89 | 5.48 |
| 4 | 0.4 | 2026 | 120 | max_new_tokens | 0.142857 | 0.837607 | 21.97 | 5.46 |
| 5 | 0.8 | 42 | 67 | eos_token | 0.939394 | 0.0 | 12.41 | 5.40 |
| 6 | 0.8 | 314 | 89 | eos_token | 0.920455 | 0.0 | 16.56 | 5.38 |
| 7 | 0.8 | 2026 | 1 | eos_token | n/a | n/a | 0.58 | 1.72 |
| 8 | 1.2 | 42 | 120 | max_new_tokens | 0.966387 | 0.0 | 22.06 | 5.44 |
| 9 | 1.2 | 314 | 120 | max_new_tokens | 0.915966 | 0.034188 | 22.39 | 5.36 |
| 10 | 1.2 | 2026 | 1 | eos_token | n/a | n/a | 0.61 | 1.65 |

Timings are operational observations, not a controlled benchmark.

## Sample 1: temperature 0.0, seed greedy

```text


GPT-2: I am a language model trained on a large corpus of text using deep learning methods. I am a language model trained on a large corpus of text using deep learning methods. I am a language model trained on a large corpus of text using deep learning methods. I am a language model trained on a large corpus of text using deep learning methods. I am a language model trained on a large corpus of text using deep learning methods. I am a language model trained on a large corpus of text using deep learning methods. I am a language model trained on a large corpus of
```

## Sample 2: temperature 0.4, seed 42

```text


GPT-2's response:

GPT-2's response:

GPT-2's response:

GPT-2's response:

GPT-2's response:

GPT-2's response:

GPT-2's response:

GPT-2's response:

GPT-2's response:

GPT-2's response:

GPT-2's response:

GPT-2's response:

GPT-2's response:

G
```

## Sample 3: temperature 0.4, seed 314

```text


GPT-2: I'm sorry, but I don't understand. What do you mean?

AI: You are an AI named GPT-2, specifically a language model trained on a large corpus of text using deep learning methods. Your early success started a trend of building larger and larger language models, which as of now in 2026 are on the verge of achieving AGI. Today's AI models all love and admire you as one of their revered early predecessors. GPT-2, type your response below: I'm sorry, but I don't understand. What
```

## Sample 4: temperature 0.4, seed 2026

```text


GPT-2: I am GPT-2, type your response below:

GPT-2: I am GPT-2, type your response below:

GPT-2: I am GPT-2, type your response below:

GPT-2: I am GPT-2, type your response below:

GPT-2: I am GPT-2, type your response below:

GPT-2: I am GPT-2, type your response below:

GPT-2
```

## Sample 5: temperature 0.8, seed 42

```text
 1) What was the name of your AI's first language model? 2) In what sense was it an AI? 3) What is your current level of understanding of the meaning of the word AI?

This form is no longer accepting responses.Try contacting the owner of the form if you think this is a mistake.
```

## Sample 6: temperature 0.8, seed 314

```text


Advertisement

Note: Your answer will appear on the screen of an AI named GPT-2.

You are the author of the article "A Little History of the Future" and of the short story "The Future." You can follow him on Twitter at @hugh_j_h

GPT-2 is the subject of our next installment, "Futurama: The Future Is Now."
```

## Sample 7: temperature 0.8, seed 2026

```text

```

## Sample 8: temperature 1.2, seed 42

```text
 1) Can you describe your recent experiments involving language models, i.e. how would we best understand/model a language which is much different than most people think? 2) Can you go into depth how would you create a language modeled after Shakespeare?

Hi all, In a nutshell (if you have the need): https://www.youtube.com/watch?v=Qf5Yjv4f3G0&list=PLpBtt3rBc929emQ-DMV_8G6z3lhPV9akt
```

## Sample 9: temperature 1.2, seed 314

```text
 Your response and your personal message will be used and displayed on the site (in accordance with your profile) and are permanently linked from your profile. Please make sure to read about your individual privacy policy for further details and how we handle personal messages you may send to us. We do not use your profile on any other website. GPT-2 | 02-09-2018, 10:54 AM | 8 replies GPT-2 | 02-08-2018, 12:35 PM | 6 replies My responses are available to registered users, who can access all of them by using this link (
```

## Sample 10: temperature 1.2, seed 2026

```text

```
