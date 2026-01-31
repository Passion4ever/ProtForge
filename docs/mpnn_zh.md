# MPNN

ProteinMPNN/LigandMPNN 序列设计。

## 下载权重

```bash
bash scripts/download_mpnn_weights.sh
```

## 使用方式

```bash
# 使用配置文件
mpnn design -c config.yaml

# 使用命令行参数
mpnn design --pdb_path input.pdb --out_folder output/
```

## 示例

```
examples/mpnn/
├── configs/         # YAML 配置文件
├── data/            # 示例 PDB 文件
└── outputs/         # 输出目录
```

| 配置文件 | 描述 |
|---------|------|
| `01_basic_monomer.yaml` | 基础单体设计 |
| `02_design_specific_chains.yaml` | 设计复合物中特定链 |
| `03_fix_residues.yaml` | 固定特定残基 |
| `04_redesign_specific_residues.yaml` | 仅重新设计特定残基 |
| `05_homooligomer.yaml` | 同源多聚体设计 |
| `06_bias_amino_acids.yaml` | 氨基酸组成偏好 |
| `07_omit_amino_acids.yaml` | 排除特定氨基酸 |
| `08_with_dna_ligand.yaml` | DNA 配体设计 (LigandMPNN) |
| `09_diverse_sequences.yaml` | 生成多样化序列 |
| `10_with_sidechain_packing.yaml` | 设计 + 侧链打包 |
| `11_per_residue_bias.yaml` | 残基级氨基酸偏好 |
| `12_soluble_model.yaml` | 可溶性蛋白模型 |

---

## 参数文档

### 参数总览

| 类别 | 数量 | 说明 |
|-----|------|------|
| 必需参数 | 2 | 输入/输出 |
| 常用参数 | 15 | 日常使用 |
| 高级参数 | 20+ | 特定场景 |

---

### 1. 必需参数

#### pdb_path / pdb_dir

输入 PDB 文件。

```bash
# 单个文件
--pdb_path ./protein.pdb

# 批量目录
--pdb_dir ./backbones/
--recursive              # 可选：递归子目录
```

```yaml
# 单个文件
pdb_path: ./protein.pdb

# 批量目录
pdb_dir: ./backbones/
recursive: true
```

#### out_folder

输出目录。

```bash
--out_folder ./outputs/
```

输出结构：
```
outputs/
├── seqs/           # 设计的序列 (.fa)
├── backbones/      # 骨架文件 (可选)
└── packed/         # 侧链打包结果 (可选)
```

---

### 2. 模型选择

#### model_type

```bash
--model_type ligand_mpnn
```

| 值 | 说明 | 适用场景 |
|---|------|---------|
| `protein_mpnn` | 标准 ProteinMPNN (默认) | 通用蛋白设计 |
| `ligand_mpnn` | 考虑配体上下文 | DNA/RNA/小分子/金属离子结合蛋白 |
| `soluble_mpnn` | 优化可溶性 | 需要高可溶性的蛋白 |
| `per_residue_label_membrane_mpnn` | 膜蛋白 (残基级标签) | 膜蛋白，需指定埋藏/界面残基 |
| `global_label_membrane_mpnn` | 膜蛋白 (全局标签) | 膜蛋白，仅需指定是否跨膜 |

**模型选择决策树：**
```
需要设计什么类型的蛋白？
│
├─ 与配体结合 (DNA/RNA/小分子/金属)
│   └─ ligand_mpnn
│
├─ 膜蛋白
│   ├─ 知道哪些残基在膜内/界面 → per_residue_label_membrane_mpnn
│   └─ 只知道整体是跨膜蛋白 → global_label_membrane_mpnn
│
├─ 需要高可溶性
│   └─ soluble_mpnn
│
└─ 通用蛋白
    └─ protein_mpnn (默认)
```

---

### 3. 链和残基控制

#### chains_to_design

指定要设计的链。不指定则设计所有链。

```bash
--chains_to_design "A,B"
```

格式：**逗号分隔，无空格**

| 输入 | 含义 |
|-----|------|
| `A` | 只设计 A 链 |
| `A,B` | 设计 A 和 B 链 |
| (不指定) | 设计所有链 |

#### fixed_residues

固定不设计的残基。

```bash
--fixed_residues "A:1-10,20 B:5-15,25"
```

**格式说明：**
- **空格** 分隔不同链
- **冒号** 分隔链名和残基
- **逗号** 分隔残基/范围
- **短横线** 表示范围

**示例：**

| 输入 | 含义 |
|-----|------|
| `A:1-10` | A 链 1-10 号残基 |
| `A:1-10,20,30` | A 链 1-10、20、30 号 |
| `A:1-10 B:5-15` | A 链 1-10 和 B 链 5-15 |

**注意：** 范围必须从小到大：`1-10` ✓，`10-1` ✗

#### redesigned_residues

指定**只设计**这些残基，其他全部固定。

```bash
--redesigned_residues "A:50-60 B:20,25"
```

格式与 `fixed_residues` 相同。

**注意：** `redesigned_residues` 和 `fixed_residues` 二选一。同时指定时 `redesigned_residues` 优先。

#### parse_these_chains_only

只加载指定的链，完全忽略其他链。

```bash
--parse_these_chains_only "A,B"
```

**与 chains_to_design 的区别：**

| 参数 | 作用 |
|-----|------|
| `chains_to_design` | 哪些链要**设计序列** |
| `parse_these_chains_only` | 哪些链被**加载到模型** |

**示例 (蛋白有 A、B、C、D 四条链)：**
```bash
# 场景 1：设计 A 链，BCD 作为上下文
--chains_to_design "A"
# 加载 ABCD，设计 A，BCD 固定但提供结构上下文

# 场景 2：只看 AB，完全忽略 CD
--parse_these_chains_only "A,B"
--chains_to_design "A"
# 只加载 AB，设计 A，B 作为上下文，CD 不存在
```

---

### 4. 对称性控制

#### homo_oligomer

同源多聚体模式。所有链的相同位置会被设计成相同的氨基酸。

```bash
--homo_oligomer
```

```yaml
homo_oligomer: true
```

适用于：同源二聚体、三聚体等，所有链序列相同的情况。

#### symmetry_residues

手动指定对称残基组。组内残基会被设计成相同的氨基酸。

```bash
--symmetry_residues "A12,B12|A13,B13|A14,B14"
```

**格式：**
- 竖线 `|` 分隔不同组
- 逗号 `,` 分隔组内残基

#### symmetry_weights

对称残基组的权重。控制每个位置在设计决策中的影响力。

```bash
--symmetry_weights "0.5,0.5|0.5,0.5"
```

格式与 `symmetry_residues` 对应。

---

### 5. 采样参数

#### temperature

采样温度。控制序列多样性。

```bash
--temperature 0.1
```

| 值 | 效果 |
|---|------|
| 0.1 | 保守，多样性低 (默认) |
| 0.2-0.5 | 中等多样性 |
| 1.0+ | 激进，多样性高 |

#### batch_size

每批生成的序列数。

```bash
--batch_size 10
```

#### number_of_batches

批次数。

```bash
--number_of_batches 5
```

**总序列数 = batch_size × number_of_batches**

#### seed

随机种子。设置后结果可复现。

```bash
--seed 42
```

---

### 6. 氨基酸偏好

#### bias_AA

全局氨基酸偏好。正值增加概率，负值降低。

```bash
--bias_AA "D:1.5,E:1.5,K:1.5,R:1.5,C:-3.0"
```

格式：`氨基酸:数值`，逗号分隔。

**数值参考 (log-odds)：**

| 值 | 效果 |
|---|------|
| +3.0 | 概率约 ×20 |
| +2.0 | 概率约 ×7 |
| +1.0 | 概率约 ×2.7 |
| 0 | 无影响 |
| -1.0 | 概率约 ÷2.7 |
| -3.0 | 概率约 ÷20 |
| -10.0 | 几乎禁用 |

**常见用法：**
```bash
# 增加可溶性 (偏好带电氨基酸)
--bias_AA "D:1.5,E:1.5,K:1.5,R:1.5"

# 避免氧化敏感氨基酸
--bias_AA "C:-3.0,M:-2.0"

# 偏好小氨基酸
--bias_AA "G:1.0,A:1.0,S:1.0"
```

#### omit_AA

完全禁用某些氨基酸。

```bash
--omit_AA "CM"
```

格式：直接拼接氨基酸单字母，无分隔符。

示例：`"CM"` = 禁用半胱氨酸 (C) 和甲硫氨酸 (M)

#### bias_AA_per_residue

残基级氨基酸偏好。通过 JSON 文件指定。

```bash
--bias_AA_per_residue ./bias.json
```

JSON 格式：
```json
{
  "A12": {"P": 10.8, "G": -0.3},
  "A13": {"A": 1.5, "H": -2.0}
}
```

#### omit_AA_per_residue

残基级禁用氨基酸。通过 JSON 文件指定。

```bash
--omit_AA_per_residue ./omit.json
```

JSON 格式：
```json
{
  "A12": "CM",
  "A13": "PG"
}
```

---

### 7. 其他常用参数

#### verbose

详细输出模式。

```bash
--verbose
```

```yaml
verbose: true
```

#### pack_side_chains

启用侧链打包。

```bash
--pack_side_chains
```

```yaml
pack_side_chains: true
```

#### file_ending

输出文件名后缀。

```bash
--file_ending "_v1"
```

结果：`protein_v1.fa`

#### skip_existing

跳过已有输出的 PDB (仅 YAML 批量模式)。

```yaml
skip_existing: true
```

---

### 8. LigandMPNN 专用参数

使用 `--model_type ligand_mpnn` 时：

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--ligand_mpnn_use_atom_context` | 1 | 使用配体原子上下文 |
| `--ligand_mpnn_cutoff_for_score` | 8.0 | 配体上下文截断距离 (Å) |
| `--ligand_mpnn_use_side_chain_context` | 0 | 使用侧链原子作为上下文 |

---

### 9. 膜蛋白专用参数

#### per_residue_label_membrane_mpnn

```bash
--transmembrane_buried "A:20-45 B:20-45"      # 跨膜埋藏残基
--transmembrane_interface "A:15-19,46-50"     # 跨膜界面残基
```

格式与 `fixed_residues` 相同。

**残基标签含义：**
- **埋藏 (buried)：** 完全在脂双层内部
- **界面 (interface)：** 在膜-水界面处
- **暴露 (exposed)：** 在水相中 (默认，不需要指定)

#### global_label_membrane_mpnn

```bash
--global_transmembrane_label 1   # 1=跨膜蛋白, 0=可溶蛋白
```

---

### 10. 侧链打包参数

启用 `--pack_side_chains` 后可用：

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--number_of_packs_per_design` | 4 | 每个设计的打包数 |
| `--sc_num_denoising_steps` | 3 | 去噪步数 |
| `--sc_num_samples` | 16 | 采样数 |
| `--repack_everything` | 0 | 重新打包所有残基 |
| `--pack_with_ligand_context` | 1 | 使用配体上下文打包 |
| `--packed_suffix` | `_packed` | 打包后 PDB 文件后缀 |
| `--force_hetatm` | flag | 配体原子强制写为 HETATM |

---

### 11. 其他参数

| 参数 | 类型 | 说明 |
|-----|------|------|
| `--zero_indexed` | flag | 输出序列编号从 0 开始 (默认从 1) |
| `--save_stats` | flag | 保存采样统计信息到 .pt 文件 |
| `--fasta_seq_separation` | str | 多链序列在 FASTA 中的分隔符 (默认 `:`) |
| `--parse_atoms_with_zero_occupancy` | flag | 解析 occupancy=0 的原子 |

---

## YAML 配置示例

### 基础配置

```yaml
pdb_path: ./protein.pdb
out_folder: ./outputs/
model_type: protein_mpnn
temperature: 0.1
batch_size: 10
number_of_batches: 5
seed: 42
```

### 固定部分残基

```yaml
pdb_dir: ./backbones/
out_folder: ./outputs/
model_type: protein_mpnn
temperature: 0.1
batch_size: 10
chains_to_design: "A,B"
fixed_residues: "A:1-10,50 B:1-10"
```

### 同源多聚体

```yaml
pdb_dir: ./backbones/
out_folder: ./outputs/
model_type: protein_mpnn
homo_oligomer: true
temperature: 0.1
batch_size: 10
```

### 带偏好设计

```yaml
pdb_dir: ./backbones/
out_folder: ./outputs/
model_type: ligand_mpnn
temperature: 0.2
batch_size: 10
chains_to_design: "A"
bias_AA: "D:1.5,E:1.5,K:1.5,R:1.5"
omit_AA: "CM"
```

### 完整示例

```yaml
pdb_dir: ./backbones/
recursive: false
out_folder: ./outputs/
skip_existing: true

model_type: ligand_mpnn
temperature: 0.1
batch_size: 10
number_of_batches: 5
seed: 42

chains_to_design: "A,B"
fixed_residues: "A:1-20 B:1-20"

bias_AA: "C:-10.0,M:-3.0"
omit_AA: "C"

pack_side_chains: true
verbose: true
```

---

## 参数冲突说明

### 残基控制冲突

| 冲突 | 优先级 |
|------|--------|
| `fixed_residues` vs `redesigned_residues` | `redesigned_residues` 优先 |
| `chains_to_design` vs `fixed_residues` | 两者都生效 (取交集) |

### 对称性冲突

| 冲突 | 说明 |
|------|------|
| `homo_oligomer` vs `symmetry_residues` | `homo_oligomer` 会覆盖手动设置 |

---

## 附录：氨基酸单字母代码

| 代码 | 氨基酸 | 性质 |
|------|--------|------|
| A | 丙氨酸 (Alanine) | 疏水 |
| C | 半胱氨酸 (Cysteine) | 极性 (可形成二硫键，氧化敏感) |
| D | 天冬氨酸 (Aspartic acid) | 酸性 (带负电) |
| E | 谷氨酸 (Glutamic acid) | 酸性 (带负电) |
| F | 苯丙氨酸 (Phenylalanine) | 疏水/芳香 |
| G | 甘氨酸 (Glycine) | 特殊 (最小，柔性) |
| H | 组氨酸 (Histidine) | 碱性/芳香 |
| I | 异亮氨酸 (Isoleucine) | 疏水 |
| K | 赖氨酸 (Lysine) | 碱性 (带正电) |
| L | 亮氨酸 (Leucine) | 疏水 |
| M | 甲硫氨酸 (Methionine) | 疏水 (氧化敏感) |
| N | 天冬酰胺 (Asparagine) | 极性 |
| P | 脯氨酸 (Proline) | 特殊 (破坏 α-螺旋) |
| Q | 谷氨酰胺 (Glutamine) | 极性 |
| R | 精氨酸 (Arginine) | 碱性 (带正电) |
| S | 丝氨酸 (Serine) | 极性 |
| T | 苏氨酸 (Threonine) | 极性 |
| V | 缬氨酸 (Valine) | 疏水 |
| W | 色氨酸 (Tryptophan) | 疏水/芳香 |
| Y | 酪氨酸 (Tyrosine) | 极性/芳香 |
