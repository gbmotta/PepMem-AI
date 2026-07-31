# Interpretação dos resultados — PepMem-AI

Guia para ler as saídas do dashboard (Predição, Ranking, XAI) e dos artefatos de treino.

Como os modelos foram treinados: [`TREINO.md`](TREINO.md).

> **Aviso:** o PepMem-AI **prioriza candidatos** para ensaio in vitro. Não substitui MIC experimental, citotoxicidade nem revisão clínica.

> **Nota:** as fórmulas estão em texto puro (blocos ` ```text ` ) para aparecerem no preview do Cursor. O LaTeX nativo muitas vezes não renderiza aqui.

---

## 1. O que o modelo responde

Pergunta central:

> Dado um **peptídeo** (sequência ± carga) e uma **membrana-alvo**, qual a chance de **alta atividade antimicrobiana**?

**Alta atividade** (rótulo de treino):

| Definição | Valor |
|-----------|--------|
| Classe positiva | MIC ≤ **3,4 µM** |
| Classe negativa | MIC > 3,4 µM |

A saída principal é uma **probabilidade calibrada** entre 0 e 1, acompanhada de índices físico-químicos (PMI, carga) e, quando disponível, explicação SHAP e vizinhos do treino.

---

## 2. Cards da aba Predição

Exemplo (números ilustrativos):

| Indicador | Exemplo | Significado resumido |
|-----------|---------|----------------------|
| PMI | 4,636 | Score físico-químico de interação |
| Prob. calibrada | 31,4% | Chance estimada de MIC ≤ 3,4 µM |
| Intervalo (árvores) | 0–76% | Faixa de incerteza do ensemble |
| Carga (q) | 5,0 | Carga líquida do peptídeo |

### 2.1 PMI (Peptide–Membrane Interaction Index)

Índice **interpretável** (não é a probabilidade do RF). Combina carga, hidrofobicidade, momento hidrofóbico e esterol:

```text
PMI = α · q_pep · |q_mem|
    + β · h_pep · h_mem
    + γ · μH_pep
    − δ · esterol_mem
```

com `esterol_mem = colesterol + 0,8 · ergosterol`, e `h_mem` por tipo de alvo
(ou coluna `membrane_hydrophobicity`), não mais 0,5 fixo.

**Pesos padrão atuais**

| Símbolo | Valor | Papel |
|--------|-------|--------|
| α (alpha) | 1,0 | Atração eletrostática |
| β (beta) | 0,5 | Compatibilidade hidrofóbica |
| γ (gamma) | 0,3 | Momento hidrofóbico (anfifilicidade) |
| δ (delta) | 0,4 | Penalização por esterol |

| Termo na fórmula | Variáveis | Papel biológico (simplificado) |
|------------------|-----------|--------------------------------|
| α · q_pep · \|q_mem\| | carga do peptídeo × \|carga da membrana\| | Atração eletrostática (peptídeo catiônico × membrana aniônica) |
| β · h_pep · h_mem | hidrofobicidades | Compatibilidade com a bicamada (`h_mem` tipificado) |
| γ · μH_pep | momento hidrofóbico | Anfifilicidade / tendência a hélice |
| δ · esterol_mem | colesterol + 0,8·ergosterol | Penaliza mamíferos / fungos / envelopes mais rígidos |

**Como ler**

- PMI **mais alto** → o índice sugere interação mais favorável com aquele alvo.
- PMI alto **não garante** MIC baixo: o RF pode discordar (como no exemplo 4,6 de PMI com só ~31% de probabilidade).
- Use PMI junto com a probabilidade e com **PMI_sel** (aba Ranking).

**PMI_sel** (seletividade):

```text
PMI_sel = PMI_alvo − PMI_célula_normal
```

Valores positivos sugerem preferência pelo alvo patológico em relação à membrana mamífera de referência.

---

### 2.2 Probabilidade calibrada

É a saída do Random Forest após **calibração isotônica** ajustada nas previsões *out-of-fold* do **leave-one-peptide-out (LOPO)**.

| Conceito | O que é |
|----------|---------|
| Prob. **bruta** | `predict_proba` do RF (antes da calibração) |
| Prob. **calibrada** | Prob. bruta mapeada para melhor refletir frequências reais no LOPO |
| LOPO | Em cada fold, **todos** os pares de um peptídeo ficam de fora do treino |

**Por que LOPO importa**  
Análogos quase idênticos (ex.: StigA6 vs StigA16 ~94%) vazam informação no leave-one-out **por amostra**. O leave-one-**peptide**-out é a métrica “honesta” de generalização para a família Stigmurin.

**Faixas práticas (dashboard)**

| Prob. calibrada | Interpretação operacional |
|-----------------|---------------------------|
| ≥ 70% | Candidato forte para priorizar ensaio |
| 40–70% | Intermediário — use PMI, vizinhos e ranking |
| &lt; 40% | Baixa chance de alta atividade no modelo |

No exemplo **31,4%**: o modelo **não** classifica o par como altamente ativo; trate como candidato fraco/intermediário-baixo e valide na bancada se houver outro motivo (PMI alto, interesse biológico, etc.).

---

### 2.3 Intervalo (árvores) e sigma (σ)

O RF é um ensemble de árvores. Para uma amostra:

1. Cada árvore dá uma probabilidade da classe positiva.
2. **σ (sigma)** = desvio-padrão dessas probs.
3. Intervalo aproximado:

```text
intervalo = [ max(0, p_calibrada − σ)  ,  min(1, p_calibrada + σ) ]
```

| Situação | Leitura |
|----------|---------|
| Intervalo **estreito** | Árvores concordam → predição mais estável |
| Intervalo **largo** (ex.: 0–76%, σ ≈ 0,45) | Discordância alta → **pouca certeza** |

**Importante:** intervalo largo com prob. média ~30% significa “o modelo está inseguro”, não “há 76% de chance”.

---

### 2.4 Carga líquida (q)

Carga estimada (ou informada manualmente) do peptídeo em pH fisiológico aproximado.

| Observação | Comentário |
|------------|------------|
| q &gt; 0 (catiônico) | Favorece atração a membranas bacterianas aniônicas |
| q muito alto | Pode aumentar atividade **e** hemólise/toxicidade — veja Ranking / hemólise na bancada |
| q informado vs calculado | Se você marca “carga manual”, ela prevalece sobre a estimada da sequência |

---

### 2.5 Banner “no banco” vs “fora do treino”

| Badge | Significado |
|-------|-------------|
| **No banco** | Sequência igual a um peptídeo do projeto (ex.: P11 StigA6) |
| **Fora do treino** | Sequência nova / mutante — predição por generalização |

Fora do treino a incerteza costuma ser maior; compare sempre com **vizinhos**.

---

## 3. Vizinhos no treino

Lista os peptídeos do conjunto com MIC mais próximos da sequência consultada.

**Score de vizinhança (aproximado):**

```text
neighbor_score = 0,6 · identidade  +  0,4 · cosine_ESM
```

(Se não houver embedding, usa só a identidade.)

| Coluna típica | Uso |
|---------------|-----|
| identity | Fração de identidades alinhadas no início |
| neighbor_score | Ranking de similaridade |
| mic_median_uM | MIC mediana do vizinho em todos os alvos |
| mic_alvo | MIC do vizinho **no mesmo alvo** (se existir) |
| frac_high_activity | Fração dos pares do vizinho com MIC ≤ 3,4 µM |

**Como usar**

- Vizinho com identidade alta + MIC baixo no mesmo alvo → a predição fica mais crível.
- Vizinho similar mas inativo → desconfie de PMI alto isolado.

No **modo Cloud** (sem PyTorch), a similaridade por embedding pode ser limitada; identidade de sequência continua disponível.

---

## 4. Ranking multi-alvo

Para um peptídeo, o sistema prediz vários alvos e ordena por:

```text
final_score = p_calibrada − λ · p_tox  +  0,1 · max(PMI_sel, 0)
```

| Símbolo | Significado |
|---------|-------------|
| p_calibrada | Prob. calibrada no alvo |
| p_tox | Prob. calibrada em `cell_normal` (proxy de toxicidade) |
| λ (lambda) | Slider (padrão 0,5): quanto penalizar toxicidade |
| PMI_sel | Bônus leve se o PMI for maior no alvo do que na célula normal |

**Leitura**

- Topo da lista = alvos a testar **primeiro**.
- λ alto = mais cautela toxicológica.
- λ baixo = prioriza atividade bruta.

---

## 5. XAI (SHAP)

SHAP atribui a cada feature uma contribuição para a predição da classe “alta atividade”.

| Sinal | Interpretação |
|-------|----------------|
| SHAP **positivo** | Empurra a predição **para** alta atividade |
| SHAP **negativo** | Empurra **contra** alta atividade |

### 5.1 Explicação local

Uma barra por feature (carga, PMI, LPS, peptidoglicano, embedding agregado, etc.) **nessa** sequência × alvo.

### 5.2 Importância global / beeswarm

- **Global:** média de |SHAP| no treino — quais descritores o modelo mais usa em geral.
- **Beeswarm:** cada ponto = um par MIC do treino; cor ≈ valor do descritor.

**Limite:** SHAP explica o **modelo**, não prova mecanismo molecular.

No Cloud leve, beeswarms costumam vir dos PNGs pré-computados; o modelo ativo pode ser só o **baseline** (sem ESM-2).

---

## 6. Lote FASTA

Ao enviar um `.fasta`:

1. **Usar esta sequência** — preenche o campo e você prediz um a um.
2. **Predizer todas** — tabela com PMI, prob. calibrada, carga e se está no banco; dá para baixar CSV.

Mesma membrana-alvo e mesma regra de carga para todo o lote.

---

## 7. Métricas do modelo (sidebar / barra)

| Métrica | Significado |
|---------|-------------|
| MICs no treino | Número de pares com MIC usados no RF |
| LOO amostra AUC | Leave-one-out **por par** (otimista se houver análogos) |
| Leave-peptide AUC | Leave-one-peptide-out (métrica principal de generalização) |

AUC ~0,8 no LOPO indica discriminação boa, mas **não** calibração perfeita em cada caso novo — daí o intervalo e os vizinhos.

---

## 8. Exemplo completo de leitura

Suponha:

- PMI = **4,636** (alto)
- Prob. calibrada = **31,4%**
- Intervalo = **0–76%**, σ = **0,449**
- q = **5,0**

**Narrativa sugerida**

1. Fisicamente, o índice PMI sugere interação favorável (carga alta + alvo aniônico).
2. O classificador, porém, atribui só ~31% de chance de MIC ≤ 3,4 µM → **não** priorize como “hit” forte.
3. O intervalo muito largo mostra **discordância** entre árvores → trate o número com reserva.
4. Próximos passos: ver vizinhos; olhar Ranking vs `cell_normal`; se houver interesse, medir MIC e, se possível, hemólise; depois registrar em `data/bench/mic_bench.csv` e retreinar.

---

## 9. Onde aprofundar no repositório

| Tema | Arquivo / pasta |
|------|-----------------|
| Fórmula PMI | `scripts/pmi.py` |
| Features do RF | `pepmem/features.py` |
| Calibração e LOPO | `scripts/train_utils.py`, `scripts/train_baseline.py` |
| Predição e vizinhos | `pepmem/predictor.py` |
| Bancada / novos MICs | `data/bench/README.md` |
| Deploy Cloud vs multimodal | `docs/DEPLOY.md` |
| Treino | [`TREINO.md`](TREINO.md) |

---

## 10. Checklist rápido

- [ ] Prob. calibrada na faixa desejada (≥70% para priorizar)?
- [ ] Intervalo estreito o bastante para confiar?
- [ ] PMI e PMI_sel coerentes com o alvo?
- [ ] Vizinhos com MIC favorável no mesmo alvo?
- [ ] Ranking não sugere toxicidade alta em célula normal?
- [ ] Resultado será confirmado in vitro antes de concluir?

Se alguma resposta for “não”, o resultado ainda pode ser útil como **hipótese**, não como decisão final.
