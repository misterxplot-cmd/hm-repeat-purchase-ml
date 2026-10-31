# hm-repeat-purchase-ml

Reproducible customer repeat-purchase prediction on H&M transaction history with PostgreSQL, temporal validation,
logistic regression and CatBoost.

Таблицы между маркерами `<!-- AUTO:… -->` генерирует `hmrp report` из сохранённых артефактов run; вручную
они не редактируются.

## Статус

<!-- AUTO:status:start -->
<!-- AUTO:status:end -->

Режимы: **full** — все клиенты полного `transactions_train.csv`; **smoke** — детерминированный 1% реальных
клиентов (та же логика и окна); **fixture** — синтетические данные только в тестах. Незапущенные этапы — в разделе
[Ограничения](#ограничения-и-незапущенные-этапы).

## Задача

PLACEHOLDER_TASK

## Данные

PLACEHOLDER_DATA

<!-- AUTO:full:data:start -->
<!-- AUTO:full:data:end -->

## Архитектура

PLACEHOLDER_ARCH

## Признаки

PLACEHOLDER_FEATURES

## Временные границы (full)

<!-- AUTO:full:splits:start -->
<!-- AUTO:full:splits:end -->

## Установка и запуск

PLACEHOLDER_COMMANDS

## Результаты full

### Development walk-forward (validation AP по fold)

<!-- AUTO:full:dev:start -->
<!-- AUTO:full:dev:end -->

### Выбор модели (до final test)

<!-- AUTO:full:selection:start -->
<!-- AUTO:full:selection:end -->

### Калибровка

<!-- AUTO:full:calibration:start -->
<!-- AUTO:full:calibration:end -->

### Final test

<!-- AUTO:full:test:start -->
<!-- AUTO:full:test:end -->

<!-- AUTO:full:bootstrap:start -->
<!-- AUTO:full:bootstrap:end -->

PLACEHOLDER_INTERPRETATION

## Результаты smoke (1% клиентов; не full)

<!-- AUTO:smoke:test:start -->
<!-- AUTO:smoke:test:end -->

## Ограничения и незапущенные этапы

PLACEHOLDER_LIMITS

## Отчёты и артефакты

PLACEHOLDER_LINKS
