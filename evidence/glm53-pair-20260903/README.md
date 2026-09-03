# GLM-5.3 Flash pair validation evidence

This directory records validation of `recipes/glm-5.3-flash-exl3-dflash2-dual-spark-1m.yaml` on `dgx01` and `dgx02`.

- Workload: `sparkrun_9000d029f34ee753_9ad9c868473b`
- Served model: `GLM-5.3-Flash-EXL3`
- Endpoint at acceptance: `http://192.168.178.47:8000/v1`
- Effective `gpu_memory_utilization`: `0.85`
- Configured maximum model length: `1,000,000`

`direct-acceptance.json` records passing model discovery, exact completion, four-way concurrency, 30,042-prompt-token retrieval, structured tool calling, and vision. The effective 0.85 setting is now the recipe default because live admission at 0.86 required 104.6 GiB while the constrained worker had 104.07 GiB free. The sanitized `../qwen3.8-flash-next-nvfp4-20260902/final-secure-proxy-regression.json` records fresh direct and LiteLLM-routed GLM semantic checks, and `../final-secure-audit-20260903.json` records the final cross-service criteria without unrelated listener or overlay-network inventory.
