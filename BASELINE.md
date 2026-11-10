# Baseline: MultiNet / OpenVLA на Jaco Play

Завершённый запуск: Kaggle `stepanivanov233/multinet-openvla-jaco-baseline`,
версия 3, `COMPLETE` (10 ноября 2026 года). Оценён **один** датасет.

| Метрика | Наш результат | Авторский reference | Отклонение |
| --- | ---: | ---: | ---: |
| AMSE | 0.237987188510227 | 0.23864495486236428 | −0,2756255% |
| NAMSE | 0.2276125782619673 | 0.22824167050949942 | −0,2756255% |

Покрытие: **109 эпизодов / 7838 шагов**, официальный Jaco Play 0.1.0 test.
Время evaluation: 5592,59 с, или 93,21 мин. Это близкое численное
воспроизведение одного набора, не точная репликация всей таблицы MultiNet.

## Закреплённый протокол

- MultiNet commit `546a1c77f9a4cc0544c5e4b41565949778d32441`.
- OpenVLA `openvla/openvla-7b`, revision
  `47a0ec7fc4ec123775a391911046cf33cf9ed83f`.
- Авторские загрузчик, prompt, crop area 0.9, action conversion и статистики.
- Greedy decoding `do_sample=False`, seed 7, без квантования.
- 2 × Tesla T4, FP16, eager attention; у авторов L4/BF16/Flash Attention 2.
- AMSE — среднее квадратов ошибки по 4 координатам и всем кадрам.
  NAMSE — среднее min–max-нормированных timestep MSE этого метода.

Первый Jaco test split состоит из 8 TFRecord shards; их hashes сохранены
в manifest. Ошибка coverage приводит к `coverage_mismatch`, а не к тихому
сохранению результата как полного.

## Проверка

[Исходные outputs](artifacts/baseline_audit/current/output/) содержат все
предсказания. [Первичный audit](artifacts/baseline_audit/verified_jaco_result.json)
сохранён без изменения; его поле `cs_saved_steps_matching_baseline=117`
относится к частичному CS v1 на момент первой проверки. Последующий
[CS v2 audit](artifacts/cs_pilot/verified_v2_result.json) подтвердил совпадение
уже **всех 1487** paired greedy действий с baseline.

```sh
python scripts/verify_baseline_results.py
```

Скрипт без ML-зависимостей заново проверяет SHA-256, coverage, последовательность
episode/step IDs, реальные boundaries, Jaco conversion, конечность действий,
каждое MSE, AMSE/NAMSE и сравнение с закреплённым reference.

Поле `offline_final_action_exact_match_percent` — диагностическое совпадение
последнего действия с демонстрацией. Оно **не является online success rate**.

## История исполнения

V1 остановился на установке пакетов для Python 3.13, v2 — на отсутствии wheel
для `promise`. Inference результатов у них нет. V3 использовал отдельную среду
Python 3.11 и разрешил сборку `promise`; он завершил весь test split.

[Точный исходник v3](artifacts/baseline_audit/submitted_v3/) сохранён отдельно.
Текущий [runtime](kaggle/openvla_baseline/baseline.py) выделен для повторного
использования в CS-пилоте. Второй датасет не запускался.

Команды GPU-воспроизведения — в [инструкции](docs/REPRODUCIBILITY.md).
Интерпретация baseline вместе с CS — в [общем отчёте](docs/RESULTS.md).
