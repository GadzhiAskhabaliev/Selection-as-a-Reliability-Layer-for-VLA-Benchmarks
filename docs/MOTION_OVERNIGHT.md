# Ночной эксперимент: CS по движению, захват от greedy

Протокол зафиксирован **11 ноября 2026 года**, до получения результатов.
Оба private Kaggle job отправлены 11 ноября в **00:18 МСК**, версия 1.
Проверка в **00:19 МСК**: оба имеют статус `RUNNING`. Этот API-статус
не подтверждает начало inference: среда ещё может готовиться или ждать ресурсы.
Итоговых метрик пока нет.

Квитанции и сохранённые статусы: [Jaco](../artifacts/overnight_motion/jaco/launch_receipt.json),
[NYU](../artifacts/overnight_motion/nyu_door/launch_receipt.json).
Это отдельный эксперимент после отрицательного одношагового CS-пилота.

## Что меняется

На каждом наблюдении генерируем восемь полных 7D-действий OpenVLA. CS сравнивает
только первые шесть нормированных координат — translation и rotation.
У Jaco rotation постоянна, поэтому фактически сравнивается x/y/z.
Затем берём движение выбранного кандидата, а координату gripper — из greedy.
Это составное действие: оно не обязательно совпадает с одним полным кандидатом.

**Захват не удаляется из действия или итоговой AMSE/NAMSE.** Основная метрика
остаётся полной ошибкой MultiNet. Дополнительно сохраняются ошибки движения,
захвата и каждой координаты. Online success rate здесь не измеряется.

## Два запуска

| Набор | Оцениваемые эпизоды | Шаги | Примерное время evaluation |
| --- | --- | ---: | --- |
| Jaco Play 0.1.0 test | IDs 20..59, 40 полных эпизодов | 2924 | 5–6 часов |
| NYU Door Opening 0.1.0 test | IDs 0..48, все 49 эпизодов | 2209 | около 4 часов |

Индексы с нуля, в фиксированном порядке TFDS/MultiNet. Jaco IDs 0..19, на которых
появилась гипотеза о захвате, исключены. Полный greedy baseline Jaco уже был
просмотрен; это новые эпизоды для оценки CS, а не никогда не виденные данные.
NYU выбирается по наличию официального test split, подходящего action adapter
и небольшому объёму, без просмотра результата CS на этом датасете.

Время оценено по предыдущему Jaco-пилоту; скорость NYU ещё не измерена.
Установка, скачивание и очередь идут сверх этой оценки. Каждый private job
запрашивает T4 accelerator (ранее давал 2 × T4); лимит выполнения — 8 часов.
Пиковая память, фактические устройства и длительность сохраняются в outputs.

- [Jaco Kaggle job](https://www.kaggle.com/code/stepanivanov233/multinet-openvla-jaco-motion-cs8)
- [NYU Kaggle job](https://www.kaggle.com/code/stepanivanov233/multinet-openvla-nyu-door-motion-cs8)

## Контроли и настройки

На общем pool сохраняются девять методов:

1. Greedy.
2. Single sample и Random-8.
3. Обычные Medoid-8 и CS-8 по всему нормированному действию.
4. Single sample и Random-8 с заменой захвата на greedy.
5. Motion Medoid-8 и Motion CS-8 с захватом от greedy.

Основной метод — `motion_cs8_greedy_gripper`. Основные контроли — greedy и
`motion_medoid8_greedy_gripper`; остальные позволяют отделить эффект sampling,
обработки захвата и правила выбора. Эти дополнительные сравнения не требуют
новой генерации pool и почти не добавляют GPU-стоимости.

OpenVLA checkpoint/revision, MultiNet commit, preprocessing, FP16/eager и frozen
normalization остаются теми же, что в завершённом пилоте. `N=8`, `R=1`,
`temperature=1.0`, `top_k=10`, `top_p=1.0`, seed 7. Настройки не подбираются
по текущим результатам. Один seed ограничивает силу обобщения.

Официальные test split содержат 109 и 49 эпизодов соответственно; полный split
скачивается, но Jaco inference выполняется только на выбранных 40 эпизодах.
Loader и action converters остаются из закреплённого MultiNet.

## Проверки и артефакты

Перед длинной оценкой первые три наблюдения проверяются на совпадение decoder
с upstream greedy и на независимый NumPy-пересчёт геометрии. На каждом шаге
проверяется сохранение greedy-захвата после action conversion, корректность
токенов и размерностей. Targets читаются только после выбора действий.
Несоответствие boundary или ожидаемого coverage приводит к ошибке.

Каждый job сохраняет `candidates.jsonl`, `result.json`, `progress.json`,
`episode_errors.json`, `protocol.json`, `manifest.json`, `bundle_sources.json`,
`startup_checks.json` и environment freeze. При незавершённом запуске `progress`
нельзя выдавать за полный результат. Итог содержит paired episode-bootstrap
интервалы: 2000 resamples, seed 200010.

Конфигурации: [Jaco](../configs/motion/jaco.json), [NYU](../configs/motion/nyu_door.json).
Код: [motion_pilot.py](../experiments/offline/motion_pilot.py).
Исходные завершённые baseline и CS v2 не изменены.

```sh
python -m unittest discover -s tests -v
python scripts/build_motion_kernels.py
# Перед повторным push заменить account/slug в metadata на свой.
kaggle kernels push -p kaggle/motion_jaco -t 28800
kaggle kernels push -p kaggle/motion_nyu_door -t 28800
```

`push` запускает новый job; не использовать его для проверки статуса.
