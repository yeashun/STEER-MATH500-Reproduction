# STEER / Qwen2.5-Math / MATH500

本目录保留可执行入口、冻结数据、vLLM 补丁、评测组件与已有测试。算法入口为 ours_main.py，统一运行入口为 run.py。

- [环境与运行指南](../docs/reproduction.md)
- [主结果与限制](../reports/main/report.md)
- [上游来源与执行版本差异](../provenance/code-versions.json)
- [文献汇报](../slides/README.md)

从本目录执行 `python run.py --method all --samples 10 --dry-run` 可检查命令和数据，无需 GPU。
