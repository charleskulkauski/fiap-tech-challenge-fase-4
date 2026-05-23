# Assistente Médico Multimodal — Tech Challenge Fase 4

**Curso:** FIAP Pós-Tech — Inteligência Artificial para Devs (IADT)  
**Tema:** Análise e fusão multimodal (vídeo, áudio e texto) para triagem em saúde da mulher  
**Repositório:** [charleskulkauski/fiap-tech-challenge-fase-4](https://github.com/charleskulkauski/fiap-tech-challenge-fase-4)

Pipeline multimodal de **apoio à triagem clínica não diagnóstica**, voltado à saúde materna e ginecológica. O sistema integra visão computacional, análise de áudio, fusão de risco e geração de relatórios, com conformidade LGPD e integração opcional com serviços Azure.

> **Aviso:** todas as saídas devem ser validadas por profissional de saúde habilitado. O sistema não substitui avaliação clínica.

---

## Sumário

- [Contexto do desafio (edital)](#contexto-do-desafio-edital)
- [Alinhamento com requisitos do PDF](#alinhamento-com-requisitos-do-pdf)
- [Arquitetura e fluxo multimodal](#arquitetura-e-fluxo-multimodal)
- [Stack tecnológica](#stack-tecnológica)
- [Instalação e execução](#instalação-e-execução)
- [Variáveis de ambiente](#variáveis-de-ambiente)
- [Entregáveis do projeto](#entregáveis-do-projeto)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Treinamento de modelos](#treinamento-de-modelos)
- [Casos demo](#casos-demo)
- [Limitações conhecidas](#limitações-conhecidas)

---

## Contexto do desafio (edital)

Conforme o PDF **POSTECH — IADT — Tech Challenge — Fase 4**, a rede hospitalar deseja monitorar pacientes por meio de **dados multimodais (áudio, vídeo e texto)** para identificar sinais precoces de risco em saúde e segurança feminina.

### Funcionalidades previstas no edital (escolher ≥ 2)

| Opção do edital | Status neste projeto |
|-----------------|----------------------|
| Analisar vídeos (partos, cirurgias, fisioterapia, consultas) | **Implementado** — pipeline de vídeo com YOLO, pose, dor, hematoma e eventos temporais |
| Processar gravações de voz (depressão pós-parto, ansiedade, violência, fadiga) | **Implementado** — Azure Speech + análise lexical/prosódica |
| Detectar anomalias em sinais vitais / prescrições / evolução clínica | **Parcial** — alertas multimodais estruturados (`risk_alert.json`); sem integração HL7/FHIR de sinais vitais |
| Integrar serviços em nuvem (Azure Cognitive Services) com privacidade | **Implementado** — Azure Speech + Azure OpenAI (opcional), com LGPD e modo local |

### Objetivos do edital (escolher ≥ 3)

| Objetivo | Como o projeto atende |
|----------|----------------------|
| Detectar precocemente riscos em saúde materna e ginecológica | Pipeline de vídeo (sangramento, instrumentos, dor, postura) + fusão em `risk_alert.json` |
| Identificar sinais de violência doméstica ou abuso | Categoria **Acolhimento/Violência** + áudio (hesitação, abuso verbalizado) + vídeo (medo, hematoma) |
| Monitorar bem-estar psicológico feminino | Proxy emocional (DeepFace/medo), postura (MediaPipe), ansiedade prosódica e score de desconforto |
| Utilizar serviços em nuvem | Azure Speech (transcrição) e Azure OpenAI (relatório narrativo, modo `hybrid`/`azure`) |
| Detecção de anomalias em tempo real (monitoramento preventivo) | Alertas batch/frame-a-frame: instrumento ∩ sangramento, picos de dor, flags de equimose e fusão multimodal |

---

## Alinhamento com requisitos do PDF

### 1. Análise de vídeo especializada para saúde da mulher

| Requisito obrigatório (PDF) | Implementação | Módulo / artefato |
|-----------------------------|---------------|-------------------|
| **Cirurgias:** detecção de complicações ou sangramento anômalo | YOLOv8 customizado (instrumentos + bleeding), filtro espectral RGB/HSV, regra espacial instrumento ∩ sangramento, classificador temporal (bleeding/smoke) | `src/pipeline_video.py`, `models/yolo_instruments/`, `models/action_events/` |
| **Consultas:** sinais não verbais de desconforto ou medo | Proxy de dor facial (`score_pain`), DeepFace (medo), curvatura postural (MediaPipe) | `models/score_pain.py`, `src/pipeline_video.py` |
| **Fisioterapia:** análise de movimentos e recuperação | MediaPipe Pose — curvatura, braços, assimetria, visibilidade de landmarks | `models/pose_detect/pose_detection_video.py` |
| **Triagem de violência:** linguagem corporal indicativa de abuso | Hematoma/equimose periorbital, sangue visual, postura retraída, emoção de medo | `src/pipeline_video.py`, `streamlit_mvp/utils/case_policy.py` |
| **YOLOv8 customizado** (≥ 1 opção) | **Instrumentos cirúrgicos ginecológicos** e **sangramento anômalo** em procedimentos | `hf_models/deployment/best*.pt`, datasets em `data/` |
| **Relatórios automáticos:** desvios em procedimentos obstétricos | Perfil laparoscópico com foco em instrumentos, bleeding e eventos temporais | `models/report_generator.py` |
| **Relatórios automáticos:** complicações em cirurgias ginecológicas | Alertas de sobreposição instrumento+sangramento e CSV de eventos | `video_instrument_events.csv`, `risk_alert.json` |
| **Relatórios automáticos:** indicadores visuais de desconforto psicológico | Métricas de medo, dor suavizada, postura e resumo técnico no prontuário | `video_events.csv`, `prontuario_multimodal.json` |
| **Relatórios automáticos:** alertas de violência doméstica | Categoria Acolhimento/Violência com `priority: high` e ações prioritárias | `case_policy.py`, `risk_alert.json` |

### 2. Análise de áudio especializada para saúde da mulher

| Requisito obrigatório (PDF) | Implementação | Módulo / artefato |
|-----------------------------|---------------|-------------------|
| **Consultas ginecológicas:** tom de voz, hesitação | Transcrição Azure Speech + hits de hesitação e ansiedade lexical | `models/audio_analysis.py` |
| **Pré-natal:** ansiedade gestacional | `anxiety_acoustic_score`, nível de ansiedade prosódica (pitch, jitter, shimmer) | `audio_analysis.json` |
| **Pós-parto:** depressão pós-parto (precoce) | Detecção lexical de fadiga, trauma e padrões emocionais na transcrição | `models/audio_analysis.py` |
| **Vítimas de violência:** padrões vocais de trauma | Palavras-chave de abuso, hesitação, ansiedade acústica elevada | `case_policy.py` (categoria Acolhimento) |
| **Fusão texto + áudio + vídeo** | Consolidação em `risk_alert.json`, `prontuario_multimodal.json` e `relatorio.pdf` | `models/report_generator.py` |

### 3. Privacidade, segurança e nuvem (Azure)

| Requisito (PDF) | Implementação |
|-----------------|---------------|
| Integração com Azure Cognitive Services | **Azure Speech** (transcrição) e **Azure OpenAI** (relatório, opcional) |
| Altos padrões de privacidade para dados sensíveis | Módulo LGPD (`models/lgpd_compliance.py`): consentimento por caso, bloqueio de Azure sem opt-in, redação de PII, retenção configurável |
| Processamento local alternativo | `REPORT_MODE=local` — relatório determinístico sem envio ao Azure OpenAI; fallback prosódico local sem Speech |

### 4. Política por categoria de caso

| Categoria | Vídeo | Áudio | Instrumentos YOLO | Dor / hematoma / postura |
|-----------|:-----:|:-----:|:-----------------:|:------------------------:|
| **Acolhimento/Violência** | Sim | Sim | Não | Sim |
| **Dor corporal** | Sim | Sim | Não | Sim |
| **Laparoscopia ginecológica** | Sim | Opcional | Sim | Não (foco cirúrgico) |

Definido em `streamlit_mvp/utils/case_policy.py`.

---

## Arquitetura e fluxo multimodal

```
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit MVP (interface)                     │
│  Entrada → Processamento → Resultados Multimodais → Relatório   │
└────────────────────────────┬────────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ pipeline_video  │ │ audio_analysis  │ │ report_generator │
│ YOLO, pose,     │ │ Azure Speech,   │ │ fusão, PDF,      │
│ dor, hematoma   │ │ keywords/prosódia│ │ alertas         │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             ▼
              data/demo/<case_id>/
              ├── inputs/    (vídeo, áudio)
              ├── outputs/   (CSV, vídeo anotado, transcrição)
              └── reports/   (PDF, JSON, alertas)
```

### Artefatos gerados por caso

| Arquivo | Descrição |
|---------|-----------|
| `video_out.mp4` | Vídeo anotado (bbox, pose, dor, hematoma quando aplicável) |
| `video_events.csv` | Série temporal frame a frame |
| `video_instrument_events.csv` | Detecções de instrumentos (laparoscopia) |
| `transcript.txt` | Transcrição Azure Speech |
| `audio_analysis.json` | Métricas textuais e prosódicas |
| `risk_alert.json` | Alerta consolidado com prioridade e ações |
| `prontuario_multimodal.json` | Prontuário estruturado multimodal |
| `relatorio.pdf` / `relatorio.json` | Relatório final para equipe |
| `status.json` | Status do pipeline e erros |

---

## Stack tecnológica

| Camada | Tecnologias |
|--------|-------------|
| Linguagem | Python 3.10+ |
| Interface | Streamlit |
| Visão computacional | OpenCV, Ultralytics YOLOv8, MediaPipe Pose, DeepFace |
| Áudio | librosa, Azure Cognitive Services Speech |
| LLM / relatório | Azure OpenAI (modos `local`, `hybrid`, `azure`) |
| PDF | ReportLab |
| Modelos | Hugging Face Hub (`charleskulkauski/fiap-gynecology-models`) |
| Configuração | python-dotenv, `.env` |

---

## Instalação e execução

### Pré-requisitos

- Python 3.10+
- FFmpeg (opcional, recomendado para transcodificação de vídeo)
- Conta Azure com **Speech Service** e **Azure OpenAI** (opcional conforme modo)

### 1. Dependências

```bash
pip install -r requirements.txt
pip install -r streamlit_mvp/requirements.txt
```

### 2. Modelos (Hugging Face)

```bash
python scripts/download_hf_assets.py --models
```

Repo de modelos: [charleskulkauski/fiap-gynecology-models](https://huggingface.co/charleskulkauski/fiap-gynecology-models)

### 3. Variáveis de ambiente

Copie `.env.example` para `.env` e preencha as chaves Azure. Detalhes na [seção abaixo](#variáveis-de-ambiente).

### 4. Interface Streamlit (recomendado)

```bash
streamlit run streamlit_mvp/Home.py
```

Fluxo da interface:

1. **Entrada do Caso** — categoria, upload, consentimento LGPD  
2. **Processamento e Status** — execução do pipeline com progresso  
3. **Resultados Multimodais** — métricas, CSV, transcrição  
4. **Relatório e Exportação** — PDF e JSON final  

Casos persistidos em `data/demo/<case_id>/`.

### 5. Linha de comando

**Pipeline completo:**

```bash
python run_pipeline.py --video "<video.mp4>" --audio "<audio.wav>" --outdir "<saida>" --save-json
```

**Pipeline integrado:**

```bash
python main.py --input-video "<video.mp4>" --input-audio "<audio.wav>"
```

Mais detalhes de deploy em [`DEPLOY.md`](DEPLOY.md).

---

## Variáveis de ambiente

```env
# Azure Speech
AZURE_SPEECH_KEY=
AZURE_SPEECH_REGION=
AZURE_SPEECH_LANGUAGE=pt-BR
AZURE_SPEECH_FALLBACK_LANGUAGES=en-US,es-ES

# Azure OpenAI
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=https://<seu-endpoint>.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=
AZURE_OPENAI_API_VERSION=2024-10-21

# Relatório: local | hybrid | azure
REPORT_MODE=local

# LGPD (Lei 13.709/2018)
LGPD_STRICT_MODE=true
LGPD_REQUIRE_EXPLICIT_CONSENT=true
LGPD_ALLOW_AZURE_SPEECH=true
LGPD_ALLOW_AZURE_OPENAI=true
LGPD_ANONYMIZE_EXTERNAL_PAYLOADS=true
LGPD_MAX_TRANSCRIPT_CHARS_AZURE=4000
LGPD_DATA_RETENTION_DAYS=30
LGPD_PROCESSING_PURPOSE=triagem_multimodal_apoio_clinico_pesquisa
```

---

## Entregáveis do projeto

Conforme o edital da Fase 4:

| Entrega | Onde encontrar |
|---------|----------------|
| **Código-fonte completo** | Este repositório Git |
| **Relatório técnico** (fluxo multimodal, modelos, resultados, anomalias) | `docs/RELATORIO_TECNICO_TECH_CHALLENGE_FASE4.md` — gerar PDF: `python scripts/export_relatorio_tecnico_pdf.py` |
| **Vídeo demo (≤ 15 min)** | YouTube/Vimeo — deve demonstrar: upload, processamento multimodal, detecção de anomalias, integração Azure e fluxo de alerta à equipe |

### Checklist de demonstração (vídeo)

- [ ] Upload de caso com categoria clínica  
- [ ] Processamento de vídeo (YOLO / pose / dor conforme categoria)  
- [ ] Processamento de áudio (Azure Speech + análise)  
- [ ] Detecção de anomalia e geração de `risk_alert.json`  
- [ ] Integração Azure (Speech e/ou OpenAI, ou explicar modo `local`)  
- [ ] Fluxo final: relatório PDF e ações prioritárias para equipe  

---

## Estrutura do repositório

```
.
├── streamlit_mvp/          # Interface e orquestração Streamlit
├── src/                    # Pipeline de vídeo e assets HF
├── models/                 # Áudio, relatório, YOLO, LGPD, eventos temporais
├── scripts/                # Download HF, treino, export PDF técnico
├── data/                   # Datasets YOLO, GynSurg e casos demo
├── hf_models/              # Checkpoints para inferência/deploy
├── docs/                   # Relatório técnico (entrega acadêmica)
├── main.py                 # Pipeline integrado CLI
├── run_pipeline.py         # Pipeline multimodal CLI
├── requirements.txt
├── DEPLOY.md
└── .env.example
```

### Módulos principais

| Módulo | Responsabilidade |
|--------|------------------|
| `src/pipeline_video.py` | Pipeline de vídeo (YOLO, pose, dor, hematoma, eventos) |
| `models/audio_analysis.py` | Transcrição Azure + análise lexical/prosódica |
| `models/report_generator.py` | Fusão multimodal, PDF e JSON |
| `streamlit_mvp/utils/case_policy.py` | Políticas por categoria e alertas |
| `models/lgpd_compliance.py` | Consentimento, bloqueio Azure, minimização |
| `models/yolo_instruments/` | Detecção e treino YOLO |
| `models/action_events/` | Classificação temporal (action/bleeding/smoke) |

---

## Treinamento de modelos

### YOLO (instrumentos / sangramento)

```bash
python -m models.yolo_instruments.prepare_unified_dataset
python -m models.yolo_instruments.train --data data/combined_medical_bleeding.yolov8/data.yaml --model yolov8n.pt
```

### Dataset GynSurg (detecção)

```bash
python -m models.yolo_instruments.prepare_gynsurg_detection_dataset
```

### Eventos temporais

```bash
python -m models.action_events.train_gynsurg_action
```

### Exportar para Hugging Face

```bash
python scripts/train_and_export_hf_models.py
```

### Datasets utilizados

- `data/medical_fiap-tech-challenge.yolov8`
- `data/bleeding_fiap-tech-challenge.yolov8`
- `data/combined_medical_bleeding.yolov8`
- `data/gynsurg_instruments_detection.yolov8`
- `data/gynsurg` (base para eventos temporais)

Datasets no HF: [charleskulkauski/fiap-gynecology-datasets](https://huggingface.co/charleskulkauski/fiap-gynecology-datasets)

---

## Casos demo

Exemplos pré-processados em `data/demo/`:

| Caso | Categoria | Foco |
|------|-----------|------|
| `caso-laparoscopia` | Laparoscopia ginecológica | Instrumentos, sangramento, eventos temporais |
| `caso-postural` | Dor corporal | Postura, dor, movimento |
| `caso-violencia` | Acolhimento/Violência | Hematoma, medo, áudio de abuso |

---

## Limitações conhecidas

- Proxies de **dor** e **hematoma** não equivalem a avaliação clínica ou dermatológica.
- **Iluminação e tom de pele** afetam detecção de equimose.
- **Sinais vitais estruturados** (pressão arterial, batimentos fetais) não estão integrados — escopo focado em vídeo/áudio.
- **Azure OpenAI** pode variar a redação; use `REPORT_MODE=local` para saídas determinísticas.
- Tempo de processamento depende de CPU/GPU e duração do vídeo.

---

## Referências

- Edital: **POSTECH — IADT — Tech Challenge — Fase 4** (FIAP)
- Relatório técnico detalhado: `docs/RELATORIO_TECNICO_TECH_CHALLENGE_FASE4.md`
- Deploy: [`DEPLOY.md`](DEPLOY.md)
- Modelos HF: [`hf_models/README.md`](hf_models/README.md)

---

*Documento alinhado ao Tech Challenge Fase 4 — Pós-Tech IA para Devs (FIAP).*
