#!/usr/local/bin/python3
"""将本阶段只读审计 artifact 渲染为中文报告。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean


OUT = Path("/home/DIFFER/outputs/semantic_reliability_audit/20261012_202925")
LOG = Path("/home/DIFFER/log_NDP/PHASE_SEMANTIC_RELIABILITY_AUDIT/20261012_202925")


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def csv_rows(name):
    with (OUT / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def f(value, digits=6):
    if value is None or value == "":
        return "不可用"
    return f"{float(value):.{digits}f}"


def pct(value, digits=2):
    if value is None or value == "":
        return "不可用"
    return f"{float(value) * 100:.{digits}f}%"


def md(name, content):
    (OUT / name).write_text(content.rstrip() + "\n", encoding="utf-8")


def main():
    quality = load("semantic_data_quality.json")
    density = load("gpc_density.json")
    feature_audit = load("feature_audit.json")
    relation = load("semantic_feature_relation.json")
    semantic_rows = csv_rows("per_pid_semantic.csv")
    factor_rows = csv_rows("per_pid_factor_consistency.csv")
    relation_rows = csv_rows("per_pid_feature_relation.csv")

    multi_rows = [r for r in semantic_rows if int(r["clothing_count"]) >= 2]
    def count_flag(flag, rows=semantic_rows):
        return sum(int(r[flag]) for r in rows)
    factor_mean = {}
    for factor in ["Bio原始槽", "Hair", "Clothing", "Pose", "Interaction", "Environment"]:
        values = [float(r["mean_cosine"]) for r in factor_rows if r["factor"] == factor]
        factor_mean[factor] = mean(values) if values else None

    d1234 = density["density_confound"][0]
    d4321 = density["density_confound"][1]
    r1234 = relation["seeds"]["1234"]
    r4321 = relation["seeds"]["4321"]

    md("00_GPC_DENSITY_CONFOUND_AUDIT.md", f"""# 00_GPC_DENSITY_CONFOUND_AUDIT

## 审计范围

本报告只审计现有正常 GPC 与 CC-GPC-HP-PILOT 的代码、日志和 train-only 采样统计；没有重新训练、没有调用 `optimizer.step()`，也没有读取 LTCC query/gallery/test。

正式训练入口为 `scripts/train_final_gpc.py`，当前配置为 `CE + Triplet + clipBio`，随后在训练入口中追加 `0.1 × GPC`。GPC 使用 `feat[0]` 的 FP32 L2 归一化视觉 Bio 表示，选中关系的单项为 `1 - cosine`，在选中关系内部取 mean；没有选中关系时关系项为零。

代码证据：`train_final_gpc.py:950-970`、`loss/make_loss.py:133-185`。当前 `clipBio` 的 `image_features_bio` 为 `[B,512]`，被扩展为 `[B,1,512]`，因此正式 Bio 文本交叉熵只有一个 Bio 槽；`loss_clip_all` 虽然累加，但当前一槽配置下不产生额外项。

## 关系密度对照

| Seed | 方法 | 60 epoch 全局批次 | 激活批次率 | 选中关系数 | 每激活批次关系数 | 跨衣选中占比 |
|---:|---|---:|---:|---:|---:|---:|
| 1234 | Original GPC | {d1234['original']['active_batches']} | {pct(d1234['original']['active_rate_over_global_batches'])} | {d1234['original']['selected_relations']} | {f(d1234['original']['selected_relations_per_active_batch'])} | {pct(d1234['original']['cross_fraction_among_selected'])} |
| 1234 | Hard Positive | {d1234['hard_positive']['active_batches']} | {pct(d1234['hard_positive']['active_rate_over_global_batches'])} | {d1234['hard_positive']['selected_relations']} | {f(d1234['hard_positive']['selected_relations_per_active_batch'])} | {pct(d1234['hard_positive']['cross_fraction_among_selected'])} |
| 4321 | Original GPC | {d4321['original']['active_batches']} | {pct(d4321['original']['active_rate_over_global_batches'])} | {d4321['original']['selected_relations']} | {f(d4321['original']['selected_relations_per_active_batch'])} | {pct(d4321['original']['cross_fraction_among_selected'])} |
| 4321 | Hard Positive | {d4321['hard_positive']['active_batches']} | {pct(d4321['hard_positive']['active_rate_over_global_batches'])} | {d4321['hard_positive']['selected_relations']} | {f(d4321['hard_positive']['selected_relations_per_active_batch'])} | {pct(d4321['hard_positive']['cross_fraction_among_selected'])} |

Hard Positive 相对于 Original 的选中关系数比例为 Seed1234 **{f(d1234['selected_relation_ratio_hard_over_original'])} 倍**、Seed4321 **{f(d4321['selected_relation_ratio_hard_over_original'])} 倍**；激活批次比例分别为 **{f(d1234['active_batch_ratio_hard_over_original'])} 倍**、**{f(d4321['active_batch_ratio_hard_over_original'])} 倍**。这是监督机会和 GPC 梯度出现频率的明显混杂因素。

但由于关系项在激活批次内取 mean，关系数量增加不能直接等同于单个激活批次的 GPC 标量损失或梯度范数按同一比例增加。可以确认的是：Hard Positive 改变了批次内关系组成，并使更多批次产生 GPC 反向路径；不能仅凭关系总数判断语义监督可靠性。

## 结论

CC-GPC-HP-PILOT 的局部改善/检索下降不能直接归因于语义文本冲突。它首先存在约 6.3--6.4 倍有效关系密度变化，必须与语义数据问题分开。
""")

    md("01_SEMANTIC_SOURCE_AUDIT.md", f"""# 01_SEMANTIC_SOURCE_AUDIT

## 实际数据来源

当前正式 LTCC 训练读取：

- `/home/data/DIFFER_TextCaption/LTCC_ReID/EVA02-CLIP-B-16/train.npz`
- `/home/data/DIFFER_TextCaption/LTCC_ReID/train_caption_summary_biometric.json`

`data/datasets/ltcc.py:106-111` 读取 NPZ 的 `data` 与 `metadata`，每张图像对应连续的 6 个因子 embedding；`data/__init__.py:34-39` 将 `SUMMERY_TEXT` 传入数据集；`train_final_gpc.py:254-260` 将它固定为 `True`、Bio index 固定为 `0`、NonBio index 为 `[1,2]`。

## 正式有效路径

1. 图像经过 EVA Bio 分支得到 `feat_bio`。
2. `head_clip_bio(feat_bio)` 得到 Bio 图像表示。
3. `caption_feature` 的第一个槽作为 Bio 文本目标，执行 batch 内图像到文本交叉熵。
4. `clipBioReverse` 不在正式 `LOSS_TYPE=["ce","triplet","clipBio"]` 中，`NONBIO_HEAD=False`；Hair/Clothing 的非 Bio 文本槽虽然被装载并切片，但不进入当前正式损失。
5. GPC 另在 `train_final_gpc.py:950-970` 使用视觉 `feat[0]` 的同身份选中图像对，不读取可读文本、不读取 Hair/Clothing 语义标签。

## 文本是否存在

存在 77 条 PID 级 Summary JSON 的 `text` 字段；不存在逐图原始 caption 文件。NPZ 只含 `data` 和 `metadata` 两个字段，`data` 为 embedding，不含可读字符串。因而当前实际监督是“每个 PID 一条汇总身份描述 + 每图六因子 embedding”，不是“同一 PID 每个服装图像各有一条可审阅身份文本”。

## 对研究问题的影响

当前资源可以审计索引、向量完整性和 PID Summary 文本的词面不确定性，但不能识别：

- 同一 PID 不同服装图像之间的逐图属性矛盾；
- 某条描述是否把服装/姿态误当作稳定身份属性；
- 文本描述是否与图像内容真实吻合。

因此，不能把当前 Summary 文本或 embedding 自洽性直接称为“语义监督正确”。
""")

    md("02_SEMANTIC_DATA_QUALITY.md", f"""# 02_SEMANTIC_DATA_QUALITY

## train-only 完整性

| 项目 | 结果 |
|---|---:|
| LTCC train 图像 | {quality['train_image_count']} |
| PID | {quality['train_pid_count']} |
| 多服装 PID | {quality.get('train_pid_count', 0) and sum(int(r['clothing_count']) >= 2 for r in semantic_rows)} |
| Summary 记录 | {quality['summary_pid_count']} |
| Summary 缺失 PID | {quality['summary_missing_pid_count']} |
| Summary 空文本 | {quality['empty_text_count']} |
| 重复 Summary 文本 | {quality['duplicate_summary_text_count']} |
| NPZ metadata / RGB key 匹配 | 9576 / 9576，唯一且通过 |
| NPZ 数值 | `57456×512 = 9576×6×512`，float16，NaN/Inf=0 |

Summary 文本长度为字符数最小/中位数/最大 **{quality['text_chars']['min']} / {f(quality['text_chars']['median'], 1)} / {quality['text_chars']['max']}**，词数最小/中位数/最大 **{quality['text_words']['min']} / {f(quality['text_words']['median'], 1)} / {quality['text_words']['max']}**。字段均为 `text` 与五种 embedding key，每个 PID 均有记录。

## 词面观察（不是矛盾标签）

在 77 条 PID 汇总文本中，自动词面规则得到：

- 含观察不确定性表达：{count_flag('has_observation_uncertainty')} 条；多服装 PID 中 {count_flag('has_observation_uncertainty', multi_rows)} 条；
- 含“变化/多数描述/模糊”等元叙述：{count_flag('has_explicit_variation')} 条；多服装 PID 中 {count_flag('has_explicit_variation', multi_rows)} 条；
- 候选词面模糊标记：{count_flag('candidate_lexical_ambiguity')} 条；多服装 PID 中 {count_flag('candidate_lexical_ambiguity', multi_rows)} 条；
- 明确服装词（如 shirt/shorts 等）：{count_flag('has_clothing_term')} 条。

这些数字反映汇总文字的表达不确定性，不能表示属性矛盾。尤其是“average/slender/medium”等同时出现可能是区间或不同观察者摘要，不应自动标作冲突。

## 数据质量结论

资源在文件、索引、数值和 PID 覆盖层面完整；但在本任务所需的“逐图、可解释、可核验身份语义”层面不完整。
""")

    md("03_IDENTITY_SEMANTIC_CONSISTENCY.md", f"""# 03_IDENTITY_SEMANTIC_CONSISTENCY

## 可直接确认的跨衣一致性

`ltcc.py:154-156` 在 `load_sum_ft=True` 时，用 `summary[str(pid)][ft_EVA02-CLIP-B-16]` 替换每张训练图像的 Bio 文本槽。因此，对于同一个 PID 的任意不同服装图像，正式 Bio 目标向量完全相同；本次 46 个多服装 PID 的运行 Summary 槽跨衣 cosine 均为 **1.000000**，且每个 PID 的跨衣文本唯一数为 1。

这说明正式路径不存在“同 PID 不同衣服使用了两条互相矛盾的 Bio Summary target”这一类错误；但它也意味着当前训练不会暴露逐图文本差异，不能证明这条 Summary 与每一张图像都正确。

## 六因子 embedding 的有限检查

对每个多服装 PID，每个因子最多固定抽取 4000 个跨衣图像对；小于上限时全量。以下是 46 个多服装 PID 的 PID 等权平均跨衣 cosine。它们是 embedding 几何，不是可读 caption 的语义一致率：

| 因子 | 跨衣 cosine 均值 |
|---|---:|
| Bio 原始槽 | {f(factor_mean['Bio原始槽'])} |
| Hair | {f(factor_mean['Hair'])} |
| Clothing | {f(factor_mean['Clothing'])} |
| Pose | {f(factor_mean['Pose'])} |
| Interaction | {f(factor_mean['Interaction'])} |
| Environment | {f(factor_mean['Environment'])} |
| Bio Summary 运行替换槽 | 1.000000 |

Clothing 因子跨衣 embedding 相似度较低，只能说明该因子在向量空间中更随图像/服装变化；不能单独说明文本把服装错误当作身份属性。

## 候选稳定属性

汇总文本确实出现性别、头发、眼镜、身材和身高等描述，但文本是 PID 级摘要，没有每一张服装图像的原文、属性标注或独立图像核验。因而本阶段不能可靠区分：

1. 稳定身份线索；
2. 随服装/视角变化的外观线索；
3. 观察条件造成的描述差异；
4. 真正的描述冲突。

## 结论

“跨衣 Bio target 一致”是代码构造的结果，不是语义可靠性被验证的结果；“跨衣非 Bio embedding 不完全一致”也不是可解释的矛盾证据。逐图语义冲突分析在当前数据上不可完成。
""")

    md("04_SEMANTIC_FEATURE_RELATION.md", f"""# 04_SEMANTIC_FEATURE_RELATION

## 正常 GPC 的已知 train-only 变化

| 指标 | Seed1234 | Seed4321 |
|---|---:|---:|
| Hard Margin 变化 | {f(-0.013188793346497418)} | {f(-0.005580358337937741)} |
| Cross-cloth Margin 变化 | {f(-0.013441391055039997)} | {f(-0.00629222915921146)} |
| 跨衣正样本平均相似度变化 | +{f(0.008460524148577453)} | +{f(0.0023128112413641766)} |
| 困难负样本最大相似度变化 | +{f(0.00565117283871297)} | +{f(0.0060503345384335105)} |
| Purity@10 变化 | +{f(0.0034252284662665833)} | {f(-0.001075604406116204)} |
| CC_Recall@10 变化 | +{f(0.002339181929061143)} | {f(-0.0026420223143079347)} |

这些结果支持“GPC 平均正样本相似度改善并不保证局部跨衣判别改善”，但没有提供语义冲突标签。

## 词面代理的 PID 级探索

为避免伪造语义冲突标签，只把汇总文本中的不确定/元叙述词面标记为 `candidate_lexical_ambiguity`，并与 NDP-MARGIN 已有的正常 GPC PID 级变化做观察性比较。样本单位是 PID，仅分析 46 个多服装 PID：

| 代理与目标 | Seed1234 Pearson | Seed1234 Spearman | Seed4321 Pearson | Seed4321 Spearman |
|---|---:|---:|---:|---:|
| 词面模糊 vs Cross-cloth Margin 变化 | {f(r1234['lexical_ambiguity_vs_delta_cross_cloth_margin_pearson']['correlation'])} | {f(r1234['lexical_ambiguity_vs_delta_cross_cloth_margin_spearman']['correlation'])} | {f(r4321['lexical_ambiguity_vs_delta_cross_cloth_margin_pearson']['correlation'])} | {f(r4321['lexical_ambiguity_vs_delta_cross_cloth_margin_spearman']['correlation'])} |
| 词数 vs Cross-cloth Margin 变化 | {f(r1234['text_words_vs_delta_cross_cloth_margin_pearson']['correlation'])} | — | {f(r4321['text_words_vs_delta_cross_cloth_margin_pearson']['correlation'])} | — |

Pearson 与 Spearman 在两个种子间不稳定，且该代理不是人工核验的矛盾标签。不能据此说语义不一致与 Bio 跨衣困难有稳定关系。

## 反例与限制

- Summary 完全相同并不证明每张图像内容正确；它只是目标复制规则。
- 词面模糊不等于文本错误；许多描述显式承认观察不确定性。
- 当前逐锚点来源没有可用的成对语义冲突标签，也不能做独立预测验证。

主结论：本阶段没有得到“语义冲突 → 跨衣 Bio 困难”的可靠观察性证据。
""")

    md("05_BASELINE_EXPLANATION.md", f"""# 05_BASELINE_EXPLANATION

## 与普通解释的比较

当前能够计算的候选变量主要是：文本长度、观察不确定性、词面元叙述、服装词出现与普通困难正/负相似度。它们存在三个问题：

1. 文本长度和词面标记没有独立的语义正确性标签；
2. Summary 是按 PID 共享的，不能表示逐图服装描述变化；
3. Bio 局部指标与语义 embedding/文本代理都来自同一训练数据和相同 PID 分组，不能当作独立因果证据。

词面模糊与正常 GPC Cross-cloth Margin 变化的 Pearson 为 Seed1234 **{f(r1234['lexical_ambiguity_vs_delta_cross_cloth_margin_pearson']['correlation'])}**、Seed4321 **{f(r4321['lexical_ambiguity_vs_delta_cross_cloth_margin_pearson']['correlation'])}**；Spearman 分别为 **{f(r1234['lexical_ambiguity_vs_delta_cross_cloth_margin_spearman']['correlation'])}**、**{f(r4321['lexical_ambiguity_vs_delta_cross_cloth_margin_spearman']['correlation'])}**。这不是稳定的超越普通文本噪声信号。

## 没有使用自动语义判断模型

本阶段没有引入外部文本模型、没有根据同一个目标模型的语义分数生成冲突标签，也没有人工宣称某条文本与图像矛盾。因而没有额外模型版本或人工误判率可报告；相应地，也没有足够信息完成用户要求的语义真实性检验。

## 判断

现有证据最多支持：训练文本资源在“可解释、逐图、可核验”层面不足；不支持把观察到的 GPC 跨衣间隔变化包装成身份语义监督机制。
""")

    md("06_SECOND_IDEA_FEASIBILITY.md", f"""# 06_SECOND_IDEA_FEASIBILITY

## 候选目标评估

### A. 身份相关语义可信度建模

当前不能进入方法设计。理由是没有逐图原文/属性标签，无法定义“可信、模糊、冲突”的独立真值；用现有 Summary embedding 自身定义可信度会循环论证。

### B. 跨衣身份语义共识建模

当前也不能进入方法设计。正式 Bio target 已经按 PID 复制到所有服装图像，表面上拥有跨衣共识，但这种共识是数据加载器构造，不是从多服装证据验证得到；强制进一步一致可能重复 GPC 或把错误摘要传播到所有服装。

## 最小可证伪补充实验

若要重新打开该方向，先不训练新模型：

1. 只在 LTCC train 中抽取多服装 PID，并保留逐图原始 caption 或人工补写的可审阅属性记录；
2. 由不参与模型训练的标注者盲标稳定身份属性、服装属性、观察条件和冲突类型；将“未提及”与“矛盾”分开；
3. 用 PID 分组交叉验证，控制图像数、服装数、原始 Bio 困难度、相机分布和文本长度；
4. 先检验独立语义冲突标记能否解释冻结 Bio 的跨衣 Margin/CC Recall 变化；若不能，立即停止；若能，再比较 Original GPC、普通文本噪声处理和候选语义可信度干预。

## 与普通方法的区分

只有当独立语义标记提供超出困难正样本、文本长度和普通噪声标签的 PID 级信息时，才有理由讨论语义可靠性机制。当前尚未满足该门槛；本阶段不实现损失、不训练。
""")

    metrics = {
        "phase": "PHASE SEMANTIC-RELIABILITY-AUDIT",
        "primary_verdict": "INSUFFICIENT_SEMANTIC_DATA",
        "training_performed": False,
        "model_or_loss_modified": False,
        "query_gallery_test_used_for_diagnostic": False,
        "train_images": quality["train_image_count"],
        "train_pids": quality["train_pid_count"],
        "multi_cloth_pids": len(multi_rows),
        "summary_text_records": quality["summary_pid_count"],
        "summary_text_nonempty": quality["nonempty_text_count"],
        "summary_duplicate_text_count": quality["duplicate_summary_text_count"],
        "per_image_raw_caption_available": False,
        "npz_shape": [57456, 512],
        "npz_factor_count": 6,
        "npz_metadata_alignment": "PASS",
        "formal_active_semantic_target": "PID-level Summary Bio copied to every image; one clipBio slot",
        "nonbio_reverse_loss_active": False,
        "gpc_density": density["density_confound"],
        "normal_gpc_deltas": {
            "1234": {"hard_margin": -0.013188793346497418, "cross_cloth_margin": -0.013441391055039997, "cross_cloth_positive_mean": 0.008460524148577453, "negative_max": 0.00565117283871297, "purity10": 0.0034252284662665833, "cc_recall10": 0.002339181929061143},
            "4321": {"hard_margin": -0.005580358337937741, "cross_cloth_margin": -0.00629222915921146, "cross_cloth_positive_mean": 0.0023128112413641766, "negative_max": 0.0060503345384335105, "purity10": -0.001075604406116204, "cc_recall10": -0.0026420223143079347},
        },
        "lexical_proxy_correlations": relation["seeds"],
        "raw_factor_cross_cloth_mean_pid_equal_weight": factor_mean,
        "feature_audit": feature_audit,
        "interpretation": "没有逐图可读语义和独立可靠性标签，不能验证身份语义冲突与跨衣 Bio 困难的稳定关系。",
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md("FINAL_REPORT_CN.md", f"""# PHASE SEMANTIC-RELIABILITY-AUDIT 最终报告

## 主判定

**INSUFFICIENT_SEMANTIC_DATA**

当前项目的语义文件在索引和数值层面完整，但不具备本阶段所需的逐图可读身份语义和独立可靠性标注。因此不能可信地判断同一身份不同服装图像是否存在语义监督冲突，也不能把它与 Bio 跨衣困难建立稳定关系。

## 直接回答

1. **DIFFER 实际使用的身份相关语义**：LTCC 每张 train 图像读取六因子 embedding；正式 `clipBio` 只使用 Bio 槽，且 `SUMMERY_TEXT=True` 时将 77 个 PID 级 Summary Bio 向量复制给该 PID 的所有图像。Hair/Clothing 等非 Bio 槽在当前正式路径中没有进入损失；GPC 使用视觉 Bio 特征图像对，不直接使用文本。
2. **完整性与可靠性**：9576/9576 metadata 对齐、无 NaN/Inf、77/77 PID Summary 非空；但只有 77 条 PID 汇总文本，NPZ 没有逐图原文。文件完整不等于语义正确。
3. **同身份跨衣候选稳定属性冲突**：不可核实。正式 Bio target 跨衣完全相同是加载器复制的结果，不是独立语义验证；非 Bio embedding 变化无法给出词级冲突证据。
4. **是否普遍存在**：不能估计。现有词面不确定性只能作代理，不能标为冲突。
5. **与跨衣困难的关系**：正常 GPC 的 Hard Margin/Cross-cloth Margin 在两个种子均下降，但词面模糊代理与间隔变化的 Pearson/Spearman 不稳定（见 `04_SEMANTIC_FEATURE_RELATION.md`），没有稳定语义关联。
6. **是否超过普通文本噪声和困难正样本**：没有。当前没有独立语义冲突标签，且 CC-GPC-HP-PILOT 还存在约 6.3--6.4 倍 GPC 关系密度混杂。
7. **是否转向身份语义可靠性**：暂不值得进入新损失设计；先补充独立逐图语义数据，否则应停止该分支。
8. **最小可证伪实验**：对 train-only 多服装图像保留/补充逐图属性记录，盲标稳定/可变/缺失/矛盾，按 PID 分组并控制文本长度、图像数、服装数、相机和初始 Bio 难度，检验独立矛盾标签是否预测冻结 Bio 跨衣困难；失败即停止。

## GPC 密度混杂

Hard Positive 训练相对于 Original GPC 的选中关系数为 Seed1234 **{f(d1234['selected_relation_ratio_hard_over_original'])} 倍**、Seed4321 **{f(d4321['selected_relation_ratio_hard_over_original'])} 倍**。关系项内部取 mean，所以不能说 GPC 标量梯度按同倍数增加；但激活批次率提高至约 64%，明显增加了关系监督出现机会。该混杂必须与语义问题分开。

## 正常 GPC 的背景结果

| 指标 | Seed1234 | Seed4321 |
|---|---:|---:|
| Hard Margin 变化 | {f(-0.013188793346497418)} | {f(-0.005580358337937741)} |
| Cross-cloth Margin 变化 | {f(-0.013441391055039997)} | {f(-0.00629222915921146)} |
| 跨衣正均值变化 | +{f(0.008460524148577453)} | +{f(0.0023128112413641766)} |
| 困难负最大相似度变化 | +{f(0.00565117283871297)} | +{f(0.0060503345384335105)} |

这些是局部 Bio 结构结果，不是语义监督冲突的证明。

## 不重新训练与数据边界

本阶段未重新训练、未修改模型/损失/检查点、未调用优化器，未使用 query/gallery/test 做诊断或参数选择。四套 9576×1024 train-only Bio 特征均存在、FP32、finite，且四套图像路径哈希一致；具体 SHA256 见 `feature_audit.json`。

## 下一步决定

目前**不进入正式身份语义可靠性损失设计**。若不能取得逐图原始语义或独立人工标注，应停止该方向；若取得后仍无法超越普通困难正样本和文本噪声解释，也应停止。
""")

    (LOG / "errors.log").write_text("本阶段未发现脚本错误；所有分析为只读 train-only 审计。\n", encoding="utf-8")


if __name__ == "__main__":
    main()
