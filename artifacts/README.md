# Архив экспериментов

В git включены завершённый baseline v3 и CS pilot v2 с исходными действиями,
provenance и независимыми audit reports. Все числа можно проверить без доступа
к приватным Kaggle kernels. Скачанные outputs не редактировались.

В архив также включены завершённые motion-запуски 11 ноября: Jaco v1
(40 эпизодов / 2924 шага) и исправленный NYU v2 (49 / 2209), с исходными
кандидатами и независимыми аудитами. NYU v1 с ошибкой сохранён отдельно;
его частичные результаты не считаются завершённым экспериментом.

## Основные файлы

| Путь | Содержание |
| --- | --- |
| [baseline_audit/current/output/](baseline_audit/current/output/) | Все 7838 предсказаний, result, manifest, environment freeze, исходный лог |
| [baseline_audit/current/status.json](baseline_audit/current/status.json) | Последний сохранённый статус baseline `COMPLETE` |
| [baseline_audit/verified_jaco_result.json](baseline_audit/verified_jaco_result.json) | Исторический независимый аудит полного baseline |
| [baseline_audit/submitted_v3/](baseline_audit/submitted_v3/) | Исходник и metadata именно завершённого baseline v3, скачанные из Kaggle |
| [baseline_audit/source_pin.json](baseline_audit/source_pin.json) | MultiNet commit и hashes оригинальных файлов |
| [baseline_audit/jaco_dataset_info.json](baseline_audit/jaco_dataset_info.json) | Метаданные официального Jaco split |
| [cs_pilot/v2/output/](cs_pilot/v2/output/) | 1487 наблюдений × 8 кандидатов, итоговые метрики, episode totals, protocol, manifest, freeze, лог |
| [cs_pilot/submitted_v2/](cs_pilot/submitted_v2/) | Точный отправленный bundle CS v2; все три embedded source-модуля совпадают с текущими исходниками |
| [cs_pilot/v2/status.json](cs_pilot/v2/status.json) | Последний сохранённый статус CS `COMPLETE` |
| [cs_pilot/verified_v2_result.json](cs_pilot/verified_v2_result.json) | Независимый аудит CS, contrasts, per-coordinate errors и диагностика |
| [checksums.json](checksums.json) | SHA-256 всех включённых исходных артефактов |
| [overnight_motion/jaco/output/](overnight_motion/jaco/output/) | Завершённый motion Jaco: 2924 × 8 кандидатов и девять методов |
| [overnight_motion/jaco/verified_result.json](overnight_motion/jaco/verified_result.json) | Независимый пересчёт Jaco и совпадение greedy с исходным baseline |
| [overnight_motion/nyu_door/v2/output/](overnight_motion/nyu_door/v2/output/) | Завершённый motion NYU: 2209 × 8 кандидатов, девять методов и аудит границ |
| [overnight_motion/nyu_door/v2/verified_result.json](overnight_motion/nyu_door/v2/verified_result.json) | Независимый пересчёт NYU, bootstrap и сравнение greedy с reference |

Файлы `upstream_reference_NOT_OUR_RESULT.json` содержат **авторские** результаты
MultiNet. Их нельзя выдавать за новый запуск. Поле `cs_saved_steps_matching_baseline`
в раннем baseline audit равно 117; позже CS v2 audit подтвердил совпадение
всех 1487 кадров. Папка `current` — историческое имя завершённого baseline v3,
не обещание автоматически обновляемого состояния.

## Формат данных

### Baseline `predictions.jsonl`

Одна строка на шаг: `episode`, `step` (с нуля), `image_sha256`, `instruction`,
`raw_prediction` (7D OpenVLA), `prediction`, `target` (4D Jaco), `mse`,
`inference_seconds`, `is_last`.

### CS `candidates.jsonl`

Одна строка на наблюдение: те же IDs, image hash, инструкция, target и boundary;
`greedy_raw`; восемь `candidates` с `raw_action` и `token_ids`;
`normalized_candidates`; `selection` с bandwidth/scores/medoid costs;
`selected_indices`; итоговые `predictions`, `mse`, `seconds`; `pool_seed`.

Raw 7D order: translation x/y/z, rotation x/y/z, gripper. Jaco evaluation order:
`[gripper, x, y, z]`. Rotation в этом датасете постоянна. Перевод захвата:
`g < 0.05 → +1`, `g > 0.95 → −1`, иначе `0`. Targets хранятся для evaluator;
не передавать их селектору.

`manifest.json` фиксирует модель, revision, dataset, split, source hashes,
TFRecord hashes, action statistics, device map и версии среды. В CS
`generation_config` — defaults модели; реальные sampling overrides находятся
в `protocol`. `bundle_sources.json` связывает запуск с локальными модулями.

## Проверка целостности

```sh
python scripts/verify_artifacts.py
python scripts/verify_baseline_results.py
python scripts/verify_cs_results.py
python scripts/verify_motion_results.py
```

SHA-256 проверяет целостность опубликованного архива; метрики отдельно
пересчитываются из действий. Ни веса, ни изображения/TFRecords, ни локальные PDF,
ни credentials, ни поисковые выгрузки checkpoints не входят в этот архив.
Новые эксперименты сохранять в отдельные каталоги, не поверх этого снимка.
