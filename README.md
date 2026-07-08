# Cryoarchive Incident: Morse Audio Recognition

Решение Kaggle-контеста по распознаванию зашумленных аудиосигналов азбуки Морзе. Задача была сведена к end-to-end распознаванию последовательностей: модель получает аудиозапись, представленную спектрограммой, и возвращает строку из цифр, дефисов и пробелов без предварительной ручной сегментации сигнала.

## Задача

По условию контеста резервный аудиоканал передает коды доступа в виде Морзе, но сигнал искажен шумом, плавающим темпом и нестабильной длительностью пауз. Для каждого `.wav` файла нужно предсказать текстовую последовательность.

Целевая метрика: **Levenshtein Mean** - среднее расстояние Левенштейна между предсказанными и истинными строками. Такая метрика напрямую штрафует пропуски, вставки и замены символов, поэтому решение оптимизировалось не только по loss, но и по качеству финальной строки.

Алфавит распознавания:

```text
0123456789- 
```

## Ключевая идея решения

Вместо классического пайплайна "детектировать точки/тире -> собрать символы Морзе -> декодировать текст" использован нейросетевой sequence recognition подход:

1. Аудио переводится в нормализованную спектрограмму.
2. Нейросеть извлекает временные признаки из спектрограммы.
3. CTC-loss обучает модель сопоставлять последовательность кадров с целевой строкой переменной длины.
4. На инференсе CTC-декодер схлопывает повторы и blank-токены в итоговый текст.

Такой подход устойчивее к изменению скорости передачи и не требует вручную подбирать границы между сигналами Морзе.

## Preprocessing

Для каждого аудиофайла строилась спектрограмма и сохранялась в `.pt`, чтобы ускорить эксперименты:

- ресемплинг до `8000 Hz`;
- моно-сведение и нормализация амплитуды;
- `torchaudio.transforms.Spectrogram(n_fft=512, hop_length=128)`;
- перевод мощности в dB через `AmplitudeToDB(top_db=80)`;
- ограничение спектра первыми `70` частотными бинами;
- динамическая частотная маска вокруг наиболее энергичной полосы;
- percentile-thresholding для подавления слабого шума;
- стандартизация спектрограммы.

Код preprocessing находится в [`src/preprocessing.py`](src/preprocessing.py).

## Модели и эксперименты

В проекте проверялись несколько архитектур для распознавания последовательностей:

| Подход | Идея | Статус |
|---|---|---|
| TCN | 2D-CNN encoder + dilated 1D convolutions | baseline для временного моделирования |
| CRNN 2D | 2D-CNN + BiGRU | первый рабочий end-to-end пайплайн |
| CRNN 1D GRU | Conv1D по спектральным бинам + BiGRU + CTC | основная и лучшая ветка |
| CRNN 1D LSTM | Conv1D + BiLSTM + CTC | конкурентный вариант |
| Conformer | Conv1D + torchaudio Conformer + CTC | проверка attention/convolution архитектуры |

Обучение велось через `CTCLoss(blank=0, zero_infinity=True)`, `AdamW`, `ReduceLROnPlateau`, gradient clipping и сохранение лучших чекпойнтов. Для валидации использовался holdout split `80/20` с `random_state=42`.

## Лучшее решение

Лучший подтвержденный результат дала Optuna-настройка CRNN-модели с bidirectional GRU:

- encoder: несколько `Conv1D + BatchNorm + ReLU + Dropout` блоков;
- sequence model: многослойная bidirectional GRU;
- decoder: `LayerNorm + Linear` в CTC-алфавит;
- objective Optuna: минимизация validation Levenshtein Mean;
- число trials: `15`;
- финальное дообучение лучшей конфигурации: `20` эпох.

Лучшие найденные гиперпараметры:

```python
{
    "kernel_size": 3,
    "hidden_size": 160,
    "num_layers": 4,
    "dropout": 0.1934,
    "lr": 0.000969,
    "weight_decay": 5.08e-06,
    "grad_clip": 1.008,
}
```

Финальный сабмит: [`submits/submission_final.csv`](submits/submission_final.csv)  
Лучший сохраненный чекпойнт: [`best_model/morse_gru_optuna_train_best_every.pt`](best_model/morse_gru_optuna_train_best_every.pt)

## Результаты

Подтвержденные результаты на локальной валидации:

| Модель | Критерий выбора | Результат |
|---|---:|---:|
| GRU + Optuna | Levenshtein Mean | **0.2387** |
| GRU baseline | best validation CTC loss | 0.1240 |
| LSTM baseline | best validation CTC loss | 0.1216 |
| Conformer baseline | best validation CTC loss | 0.1441 |

Важно: baseline-модели в таблице сравнивались по validation CTC loss, а Optuna-ветка - напрямую по Levenshtein Mean, поэтому основным качественным результатом проекта считается именно GRU + Optuna.

Итоговый Kaggle score лучшего сабмита на тестовой выборке: **1.33720**.

## Что было сделано

- Построен полный пайплайн распознавания аудио: preprocessing, датасет, batching переменных длин, обучение, валидация и генерация Kaggle submission.
- Реализован CTC-декодинг для строк переменной длины без ручной сегментации Морзе.
- Проведены эксперименты с TCN, CRNN, LSTM, GRU и Conformer-архитектурами.
- Добавлена оптимизация гиперпараметров через Optuna по целевой для контеста метрике.
- Подготовлены финальные сабмиты и сохранены веса лучших моделей.

## Структура проекта

```text
.
├── src/
│   ├── preprocessing.py        # аудио -> спектрограмма
│   ├── morse_dataset.py        # Dataset для спектрограмм и текстовых меток
│   ├── collate_fn.py           # padding batch'ей переменной длины
│   ├── encoding_decoding.py    # CTC alphabet, encode/decode
│   ├── train_model.py          # training loop с CTC и validation
│   ├── evaluate.py             # Levenshtein evaluation
│   ├── morse_crnn.py           # 2D CRNN baseline
│   ├── morse_tcn.py            # TCN baseline
│   └── morse_conformer.py      # 2D Conformer baseline
├── models/
│   ├── crnn.py                 # основная Conv1D + GRU/LSTM модель
│   └── conformer_1d.py         # Conv1D + Conformer модель
├── *.ipynb                     # EDA и эксперименты
├── submits/                    # Kaggle submissions
├── models_files/               # сохраненные чекпойнты экспериментов
└── best_model/                 # лучший чекпойнт
```

## Стек

Python, PyTorch, torchaudio, scikit-learn, Optuna, python-Levenshtein, pandas, NumPy, Matplotlib.
