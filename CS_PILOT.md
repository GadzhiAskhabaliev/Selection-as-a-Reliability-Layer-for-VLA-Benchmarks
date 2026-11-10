# Первый CS-8 пилот OpenVLA

Приватный Kaggle kernel: `stepanivanov233/multinet-openvla-jaco-cs8-pilot`.
На 10 ноября 2026 года, 22:56 МСК **версия 2 завершена (`COMPLETE`)**.
Оценены 20 полных эпизодов, 1487 шагов и 11896 кандидатов; время оценки —
158.19 минуты. Независимая проверка скачанных результатов прошла.

## Результат первого пилота

Меньше — лучше. Все строки относятся к одним и тем же первым 20 эпизодам.
Эти значения нельзя сравнивать напрямую с baseline на всех 109 эпизодах.

| Метод | AMSE | NAMSE | Изменение AMSE к greedy |
| --- | --- | --- | --- |
| Greedy | 0.2315849 | 0.2270615 | 0% |
| Один stochastic sample | 0.2376200 | 0.2322724 | +2.61% |
| Random-8 | 0.2363625 | 0.2247455 | +2.06% |
| Medoid-8 | 0.2354849 | 0.2308853 | +1.68% |
| CS-8 | 0.2368162 | 0.2321905 | +2.26% |

**CS не улучшил greedy в этом пилоте.** Разность AMSE CS−greedy равна
+0.0052313; 95% paired episode bootstrap interval: [+0.0005037, +0.0114388].
Это результат на 20 эпизодах и одном sampling seed, а не вывод обо всех задачах.
CS−medoid в среднем +0.0013312, но interval [−0.0000564, +0.0041245] включает ноль.
Интервалы CS−Random и CS−single-sample также включают ноль.

Покоординатная диагностика объясняет знак общей разности: вклад захвата в AMSE
вырос на 0.0053800, суммарный вклад трёх координат движения уменьшился на 0.0001487.
Сумма MSE по x/y/z снизилась на 3.47%, но ухудшение gripper перевесило.
Это арифметическое разложение метрики, а не доказательство механизма ошибки управления.

CS и medoid выбрали одинаковый индекс в 86.42% кадров. В среднем pool содержит
2.80 уникальных нормированных действий из 8; в 38.40% кадров все кандидаты
одинаковы. Генерация pool занимала примерно в 7.94 раза больше времени, чем greedy;
сам выбор CS/medoid пренебрежимо мал относительно генерации.

У Random-8 ниже NAMSE при более высокой AMSE: NAMSE нормирует ошибки по собственному
диапазону каждого метода, поэтому её уменьшение само по себе не доказывает меньшую
ошибку предсказания. Online success rate в этом эксперименте не измеряется.

Проверены coverage, границы эпизодов, seeds, декодирование всех токенов, frozen
normalization, CS/medoid scores и выбор, Random RNG, Jaco conversion, каждое MSE,
агрегирование и episode bootstrap. Все 1487 greedy actions и соответствующие
image hashes, инструкции и GT совпадают с завершённым полным baseline.
Не-action токенов нет; 7573 учтённых upstream clipping events — допустимый
крайний action-токен, а не ошибка генерации.

Отчёт: `artifacts/cs_pilot/verified_v2_result.json`.
Выгруженные outputs: `artifacts/cs_pilot/v2/output/`.
Повторная проверка: `python scripts/verify_cs_results.py` в среде с NumPy.
V1 был остановлен; итоговый v2 повторяет тот же код с начала.

## Зафиксированный протокол

- Данные: Jaco Play 0.1.0, первые **20 полных эпизодов test** в том же порядке
  TFDS и MultiNet loader, что у baseline. По предоставленным baseline-логам это
  1487 шагов; фактическое покрытие записывает evaluator.
- Модель, revision, статистики, center crop, MultiNet conversion, FP16 и eager
  attention совпадают с Kaggle baseline. Загрузка модели и подготовка данных
  вынесены в общую функцию `prepare_runtime`; цикл baseline сохранён.
- `N=8`, `H=R=1`, temperature=1.0, top-k=10, top-p=1.0, seed=7.
  Это заранее выбранная исследовательская настройка, не результат validation tuning.
  Исходный OpenVLA поддерживает batch=1 при генерации, поэтому вызовы последовательны.
- На одном наблюдении один pool из 8 stochastic actions. Greedy вычисляется отдельно;
  single-sample — первый кандидат, Random-8 — равномерный индекс с отдельным RNG,
  Medoid-8 и CS-8 получают один и тот же нормированный pool.
- Нормировка перед выбором: обратное преобразование замороженных q01/q99 OpenVLA.
  Координаты mask=False уже нормированы и остаются как есть; постоянные координаты
  превращаются в нули. В Jaco это убирает влияние трёх постоянных rotation dimensions.
  Нормировка не оценивается по текущему pool или test labels.
- Выбор выполняется в пространстве исходного 7D action до MultiNet conversion;
  затем тот же выбранный кандидат преобразуется в 4D Jaco action для метрик.
  GT поступает только в evaluator после выбора индекса.

CS написан по формулам (1), (2) основной статьи и (S1) supplement:

```text
d_ij = Euclidean distance between flattened normalized executed prefixes
h = median(positive pairwise d_ij) + 1e-6; if none: 1 + 1e-6
score_i = sum_j exp(-d_ij^2 / (2 h^2))
selected = first index attaining maximum score
```

Medoid минимизирует сумму исходных Euclidean distances. Для обоих методов точные
ничьи разрешаются первым индексом. Перенос относится к `R=1`; это не генерация
будущей траектории и не буквальный перенос исходного кода авторов.

## Что сохраняется и проверяется

`candidates.jsonl` содержит episode/step, image hash, инструкцию, seeds, токены и
raw actions всех кандидатов, нормированные действия, bandwidth, scores, индексы
выбора, greedy, GT, ошибки каждого метода и времена. Тексты/изображения статьи
на Kaggle не загружаются.

`protocol.json`, `manifest.json`, `bundle_sources.json`, `environment-freeze.txt`
фиксируют настройки, код, данные, статистики и среду. `progress.json` обновляется
после эпизода; `result.json` появляется только после завершения 20 эпизодов.

В начале выполняется проверка совпадения собственного audited decoding с
исходным `model.predict_action(do_sample=False)` на трёх кадрах. Проверяется
число новых токенов, считаются выходы за action-token диапазон. Последние
клиппируются точно как в upstream OpenVLA и сохраняются в диагностике;
короткая последовательность после early EOS приводит к явной ошибке.

Метрики: AMSE и исходная MultiNet NAMSE отдельно для каждого метода на одном
наборе шагов; разность AMSE с paired greedy; episode bootstrap 95% interval
для разности AMSE. Bootstrap не отражает разброс по sampling seeds. Дополнительно
сохраняются разнообразие кандидатов, совпадения CS/medoid и время генерации/выбора.
NAMSE использует свой диапазон ошибок каждого метода и не заменяет сравнение AMSE.
Это один seed на части test; результаты не доказывают online success rate.

## Запуск и исходники

Для повторного запуска под своим Kaggle account см.
[инструкцию воспроизведения](docs/REPRODUCIBILITY.md).

```bash
python3 -m unittest discover -s tests -v
python3 scripts/build_cs_kernel.py
kaggle kernels push -p kaggle/openvla_cs -t 21600
kaggle kernels status stepanivanov233/multinet-openvla-jaco-cs8-pilot
kaggle kernels output stepanivanov233/multinet-openvla-jaco-cs8-pilot -p artifacts/cs_pilot/output
```

Исходники: `src/consensus_selection/__init__.py`,
`experiments/offline/cs_pilot.py`, общий runtime в `kaggle/openvla_baseline/baseline.py`.
`kaggle/openvla_cs/cs.py` автоматически собирается из этих файлов и содержит их
SHA256. Изменять нужно исходники и пересобирать bundle.

Проверено локально: формула и крайние случаи CS, маска и постоянные координаты,
action-token decoding, парное агрегирование и bootstrap, импорт bundle;
цикл оценки baseline совпадает с исходным по AST. GPU-интеграция подтверждена завершённым CS v2 и проверкой его outputs.
Сохранённый статус: `artifacts/cs_pilot/v2/status.json`.

При текущих ~0.7 секунды на один greedy вызов ориентир для 20 эпизодов и
8 кандидатов — 2–3 часа вычислений плюс подготовка и очередь. Тайм-аут 6 часов —
верхний предел запуска, не прогноз. API может не отдавать текущие output-файлы
до завершения; пустой ответ `kernels output` не означает отсутствие inference.

Документация параметров генерации:
[Transformers 4.40.1](https://huggingface.co/docs/transformers/v4.40.1/en/main_classes/text_generation).
