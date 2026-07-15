# Dados experimentais da bancada

Planilha para **inserir MICs/MBCs novos** medidos in vitro. Após preencher, rode:

```bash
python scripts/import_bench_mic.py --retrain
```

Isso atualiza datasets, retreina Random Forest e recalcula **SHAP**.

---

## 1. `mic_bench.csv` (obrigatório)

Uma linha = um resultado experimental.

| Coluna | Obrigatório | Exemplo |
|--------|-------------|---------|
| `peptide_id` | peptide_id **ou** sequence | `P10` ou `P13` |
| `sequence` | peptide_id **ou** sequence | `FFSLIPSLVGGLISAFK` |
| `target_id` | sim | `E_coli_ATCC25922` |
| `endpoint` | sim | `MIC`, `MBC`, `CC50`, `IC50`, `HEMOLYSIS`, `BIOFILM_INHIB` |
| `value` | sim | `2.3` (µM) ou `%` para hemólise/biofilme |
| `unit` | não | `uM` ou `percent` |
| `reference` | não | `bancada_2025-06` |
| `date` | não | `2025-06-03` |
| `notes` | não | `réplica 3, 37°C` |

### Exemplo — hemólise (Parente 2018)

```csv
peptide_id,sequence,name,net_charge,target_id,target,target_type,endpoint,value,unit,assay,reference,date,notes
P11,FFSLIPKLVKGLISAFK,StigA6,3,RBC_human,Hemácias humanas,mamífero (hemólise),HEMOLYSIS,30,percent,hemolysis_75uM,Parente_2018,,hemolysis_%_at_75uM_1h
```

### Alvos já disponíveis (`target_id`)

| ID | Organismo |
|----|-----------|
| `S_aureus_ATCC29213` | S. aureus ATCC 29213 |
| `S_epidermidis_ATCC12228` | S. epidermidis |
| `E_coli_ATCC25922` | E. coli |
| `P_aeruginosa_ATCC27853` | P. aeruginosa |
| `Candida_spp` | Candida |
| `cell_normal` | célula mamífera normal (toxicidade) |
| `RBC_human` | hemácias (hemólise) |
| `S_aureus_UFPEDA1040` … `P_aeruginosa_UFPEDA262` | cepas MDR (Parente 2022) |

---

## 2. `peptides_bench.csv` (opcional)

Cadastro explícito de novos peptídeos do projeto (antes dos MICs):

```csv
peptide_id,name,parent,sequence,net_charge
P13,Meu_analogo,Stigmurin,FFSLIPKLVKGLISAFK,3
```

---

## 3. `targets_bench.csv` (opcional)

Nova cepa/alvo — inclua descritores de membrana:

```csv
target_id,target,target_type,surface_charge,anionic_fraction,lps,peptidoglycan,ergosterol,viral_envelope,cholesterol
S_aureus_Lab001,Staphylococcus aureus lab X,Gram+,-0.8,0.6,0,1,0,0,0
```

---

## Comandos

```bash
# Validar planilha sem alterar nada
python scripts/import_bench_mic.py --check

# Importar + rebuild datasets/pares
python scripts/import_bench_mic.py

# Importar + retreinar modelos + SHAP (recomendado)
python scripts/import_bench_mic.py --retrain
```

---

## O que muda no modelo

- **Rótulo:** MIC ≤ 3,4 µM → alta atividade (ajustável futuramente)
- Mais MICs → RF e **SHAP beeswarm** mais estáveis
- Dados da bancada **substituem** literatura se mesmo par peptídeo×alvo×endpoint

Relatório gerado: `data/bench/import_report.json` (via `build_summary` / pares).

---

## Fontes externas (teste held-out, opcional)

Não misturar no treino do projeto sem controle de homologia. Úteis como **benchmark externo**:

| Fonte | Conteúdo | Link |
|-------|----------|------|
| AMPBench-MT | MIC + HC50/hemólise; splits por homologia | https://huggingface.co/datasets/ZihengZhou06/AMPBench-MT |
| ANIA | MIC *E. coli / S. aureus / P. aeruginosa* | https://github.com/SilverGojo4/ANIA |
| DBAASP | MIC + citotox + sinergismo | https://www.dbaasp.org |
| DRAMP | AMP bulk + atividades | https://dramp.cpu-bioinfor.org |

Validação interna preferida: **leave-one-peptide-out** (já no treino) — a família Stigmurin tem identidade ~70–94%.
