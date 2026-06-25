## Запуск экспериментов

Все команды выполняются из корневой папки проекта. Эксперименты разделены по типам архитектур и диагностическим задачам.

### Обучение моделей

```bash
# Базовая архитектура (без нормы)
python experiments/train.py

# Архитектура с нормой в attention
python experiments/train_norm.py

# Исправленная архитектура (с нормой в классификаторе)
python experiments/train_norm_cls.py
```

### Диагностические эксперименты (Ablation)

```bash
# Ablation для базовой архитектуры
python experiments/ablation.py

# Ablation для архитектуры с нормой в attention
python experiments/ablation_norm.py

# Ablation для исправленной архитектуры (с нормой в классификаторе)
python experiments/ablation_norm_cls.py
```

### Диагностические эксперименты (Анализ градиентов)

```bash
# Анализ градиентов для базовой архитектуры
python experiments/gradient_analysis.py

# Анализ градиентов для архитектуры с нормой в attention
python experiments/gradient_analysis_norm.py

# Анализ градиентов для исправленной архитектуры (с нормой в классификаторе)
python experiments/gradient_analysis_norm_cls.py
```

### Запуск всех экспериментов одной командой

Если вы хотите выполнить все эксперименты последовательно, используйте скрипт:

```bash
bash scripts/run_all.sh
```

> **Примечание:** Если файла `scripts/run_all.sh` нет в вашем репозитории, вы можете создать его самостоятельно, перечислив в нём все вышеуказанные команды.
