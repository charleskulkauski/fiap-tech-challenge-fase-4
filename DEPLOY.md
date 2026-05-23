# Deploy

## 1. Instalar dependências

```bash
pip install -r requirements.txt
pip install -r streamlit_mvp/requirements.txt
```

## 2. Baixar modelos (Hugging Face)

```bash
python scripts/download_hf_assets.py --models
```

Repo: [charleskulkauski/fiap-gynecology-models](https://huggingface.co/charleskulkauski/fiap-gynecology-models)

## 3. (Opcional) Baixar datasets para retreino

```bash
python scripts/download_hf_assets.py --datasets
```

Repo: [charleskulkauski/fiap-gynecology-datasets](https://huggingface.co/charleskulkauski/fiap-gynecology-datasets)

## 4. Configurar `.env`

Ver `README.md` seção de variáveis Azure.

## 5. Executar

```bash
streamlit run streamlit_mvp/Home.py
```

## Paths de inferência

| Uso | Arquivo |
|-----|---------|
| Instrumentos (default) | `hf_models/deployment/best.pt` |
| Laparoscopia | `hf_models/deployment/best_instrument_bleeding.pt` |
| Combined | `hf_models/deployment/best_combined.pt` |
| Eventos temporais | `hf_models/action_events/gynsurg_action_3sec_round2/*/best.pt` |

Resolução centralizada em `src/hf_assets.py`.
