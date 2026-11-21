# Run on Arc A770 cards

The serve image's kernels are built for Battlemage. Two Arc A770s (Alchemist, IP 12.55) need a build of their own,
kernels for the 8-lane XMX units included, and a host that keeps the experts the cards cannot hold.

This guide has been run on two A770s with 15.1 GiB each, 31 GiB of host memory and IQ3_XXS at a 32,768-token
context.

## What you need

- Two Arc A770 cards, or one to try it, with the `xe` kernel driver.
- 31 GiB of host memory for IQ3_XXS at 32,768 tokens with no checkpoints. The expert arena takes 21 GiB of it.
- `level-zero-devel`, `intel-ocloc`, `cmake`, `ninja`, `python3` and `uv`.
- llama.cpp's source at the commit [ADR-0012](../adr/0012-correctness-reference.md) pins; `uv run python -m tools.model_fetch`
  does not fetch it for a source build.

## Build

```bash
git clone https://github.com/avra-m3/c0ven && cd c0ven
uv sync --locked
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DENGINE_GGML_DIR=/path/to/llama.cpp \
    -DENGINE_CPU_LEVELS=haswell \
    -DENGINE_AOT_DEVICES=acm-g10 \
    -DENGINE_DG2_DEVICES=acm-g10 \
    -DENGINE_KERNEL_SPIRV_DEVICE=acm-g10
cmake --build build -j
```

`ENGINE_AOT_DEVICES=acm-g10` builds the kernels ahead of time for the cards; `ENGINE_DG2_DEVICES=acm-g10` builds their
8-lane variants, and the SPIR-V fallback is that same portable set. `ENGINE_CPU_LEVELS` takes the highest level
`/proc/cpuinfo` shows; `haswell` covers AVX2 and BMI2.

## Get the model and the drafter

```bash
uv run python -m tools.model_fetch IQ3_XXS --dir ~/c0ven-models
```

That writes the file and `dflash/` beside it. On a reply of a few hundred tokens DFlash 2 decodes about three times
as fast as no drafter, and costs 1.2 GiB of host memory; the engine finds it without an option. Without it, the engine
decodes one token a step.

## Start

```bash
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export ZES_ENABLE_SYSMAN=1
export ENGINE_LIBRARY=$PWD/build/core/libengine.so
uv run python -m frontend \
    --model ~/c0ven-models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
    --slots 1 --context 32768 \
    --engine-arg=--checkpoints=0 \
    --engine-arg=--no-vision \
    --engine-arg=--chunk-rows=1024
```

`--checkpoints=0` keeps the plan inside 31 GiB of host memory. `--no-vision` is required: the image encoder is built
for 16-lane XMX units, and the Alchemist build leaves it out. `--chunk-rows=1024` is a workaround for a known fault:
a prompt over 1,024 tokens runs the engine through its staged-expert copy list, which corrupts the model's output on
Alchemist (an issue to be fixed); chunks of 1,024 rows take the fetch path instead and decode correctly.

The load prints `drafting 1` when `dflash/` was found. It takes about 70 seconds. `curl -s localhost:8080/health`
prints the state.

## Check the kernels

```bash
cd build && FLASH_MODEL=~/c0ven-models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
    ENGINE_EXPECT_GPUS=2 ctest -L gpu -E "vision_test|encoder_test|batch_test|kl_test|api_test" --output-on-failure
```

The image encoder's tests and the ones that need the reference data are left out, as with the server above.

## What is measured

Two A770s, IQ3_XXS at 32,768 tokens, one request at a time, greedy:

| prompt | DFlash 2 | prompt reading | decoding |
| --- | --- | --- | --- |
| 1,049 tokens, 128 out | no | 87.3 tok/s | 28.3 tok/s |
| 1,049 tokens, 128 out | yes | 86.3 tok/s | 86.5 tok/s |
| 2,875 tokens, 3,000 out | yes | 91.1 tok/s | 24.2 tok/s |

The decode rate falls as the drafts stop being accepted: 58 % over 3,000 tokens, against 100 % on the repeated text
of the first two rows.
