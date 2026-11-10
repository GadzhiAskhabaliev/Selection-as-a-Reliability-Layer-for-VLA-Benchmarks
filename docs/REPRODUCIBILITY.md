# Как проверить результаты и повторить запуск

Команды выполняются из корня checkout. Есть два разных действия:
**пересчёт сохранённых результатов на CPU** и **повторный inference на GPU**.
Для первого не нужны Kaggle account, веса, TFRecords или доступ к приватным jobs.

## 1. Проверка архива на CPU

Python 3.11 и NumPy 1.26.4:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-audit.txt
python -m unittest discover -s tests -v
python scripts/verify_artifacts.py
python scripts/verify_baseline_results.py
python scripts/verify_cs_results.py
```

Не использовать `python -O`: часть проверок — assertions. Эти команды не меняют
результаты. `verify_cs_results.py --write-reports` отдельно разрешает обновление
производных audit JSON; для проверки опубликованного снимка флаг не нужен.

Ожидается: все unit tests проходят; SHA-256 archive check — `PASS`; baseline
7838 шагов — `PASS`; CS 1487 шагов — `PASS`.

GitHub Actions запускает эти же проверки и проверяет исходные hashes
пересобираемых Kaggle bundles. Здесь нет обучения или платного GPU-запуска.

## 2. Где смотреть данные

См. [карту архива](../artifacts/README.md). `candidates.jsonl` включает все
1487 наблюдений и восемь кандидатов на каждое; не только выбранные действия.
Для новой идеи можно написать отдельный replay evaluator поверх этого cache,
не изменяя опубликованные outputs. Targets доступны только для подсчёта ошибки
после выбора. Эти 20 test-эпизодов уже просмотрены и не подходят для независимого
подтверждения новой настройки.

## 3. Повторный запуск на Kaggle

Установить Kaggle CLI и настроить авторизацию **вне репозитория**. В копиях
`kaggle/*/kernel-metadata.json` заменить поле `id` на свой `username/slug`.
Исторические IDs относятся к приватным jobs исходного account; соавтору
не требуется к ним доступ для CPU-аудита.

Проверенная среда inference: **2 × T4**, интернет включён, без квантования.
Скрипт сам создаёт отдельную Python 3.11 среду с CUDA 12.1/PyTorch 2.2.0,
Transformers 4.40.1, NumPy 1.26.4 и TensorFlow CPU 2.15.1.
Полные pins находятся в `baseline.py`, фактические freeze — в архиве.
Примерные скачивания: 14 ГБ весов и 1 ГБ Jaco test. Их нет в git.

Повторить baseline или CS-пилот:

```sh
kaggle kernels push -p kaggle/openvla_baseline -t 21600
python scripts/build_cs_kernel.py
kaggle kernels push -p kaggle/openvla_cs -t 21600
```

`push` создаёт новый GPU job; для проверки статуса использовать `status`,
не повторный `push`. Не записывать новые outputs поверх опубликованного архива.
`-t` задаёт лимит runtime, не прогноз длительности. Измеренное evaluation time:
93,21 минуты baseline и 158,19 минуты pilot, плюс установка/скачивание/очередь.
Доступность GPU и очередь не гарантированы.

Скрипт [`check_kaggle_run.py`](../scripts/check_kaggle_run.py) проверяет статус
однократно и скачивает outputs после завершения:

```sh
python scripts/check_kaggle_run.py YOUR_USER/YOUR_SLUG artifacts/new_run
```

## 4. Что закреплено и что может отличаться

Model revision и MultiNet commit находятся в manifests; TFRecord SHA-256,
normalization hash, device map и package versions тоже сохранены. При повторе
сначала сверить их. Более новый `main` MultiNet не заменяет закреплённый commit.

Исходник завершённого baseline v3 сохранён отдельно; текущий `baseline.py`
выделяет общий runtime для CS. Цикл оценки и метрики сохранены. Точный отправленный CS v2 bundle также включён в архив; его оболочка
может отличаться от заново собранной, но три исходных модуля совпадают.
Для CS v2 `bundle_sources.json` фиксирует точные hashes трёх исходных модулей, а аудит
проверяет их соответствие.

Повторный inference может зависеть от GPU, dtype, attention и библиотек.
Цель CPU-проверки — точное восстановление опубликованных метрик из сохранённых
действий. Близость нового GPU-прогона проверяется отдельно, без обещания
побитового совпадения на любой машине.
