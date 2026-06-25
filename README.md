## Установка

### 1. Клонировать репозиторий

```bash
git clone https://github.com/nordllichterrr/dual_attention_NLP.git
cd dual_attention_NLP
```

### 2. Создать и активировать виртуальное окружение

```bash
python3 -m venv venv
source venv/bin/activate      # macOS/Linux
# venv\Scripts\activate       # Windows
```

### 3. Установить зависимости

```bash
pip install -r requirements.txt
```

---

## Запуск экспериментов

Все команды выполняются из корневой папки проекта.

### Обучение моделей

```bash
# Базовая архитектура (без нормы)
python experiments/train.py

# Архитектура с нормой в attention
python experiments/train_norm.py

# Исправленная архитектура (с нормой в классификаторе)
python experiments/train_norm_cls.py
```

### Ablation (зануление b-компоненты)

```bash
# Для базовой архитектуры
python experiments/ablation.py

# Для архитектуры с нормой в attention
python experiments/ablation_norm.py

# Для исправленной архитектуры (с нормой в классификаторе)
python experiments/ablation_norm_cls.py
```

### Анализ градиентов

```bash
# Для базовой архитектуры
python experiments/gradient_analysis.py

# Для архитектуры с нормой в attention
python experiments/gradient_analysis_norm.py

# Для исправленной архитектуры (с нормой в классификаторе)
python experiments/gradient_analysis_norm_cls.py
```
