# Третье ДЗ по курсу продвинутое машинное обучение

[Описание задания](https://docs.google.com/document/d/1FWCuz-3Q_85yQYEwz6xVkIuXUjQn60dVFxreJId-liM/edit)

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KernelA/made-adv-ml-hw2/blob/master/solution.ipynb)

## Требования для запуска

1. Python 3.8 или выше
2. Настроенная Anaconda (опционально)
3. Установленный [Jupyter Lab или Notebook](https://jupyter.org/)
4. [Настроенный Plotly для Jupyter Lab](https://plotly.com/python/getting-started/)
5. Git LFS (исходные данных хранятся в LFS)

## Как запустить

Можно запустить пример сразу в Google Colab.

### Локальная настройка

Создать новое окружение для Anaconda:
```
conda env create -n env_name python=3.8 pip -y
conda env update -n env_name --file ./requirements.txt
```

Если исходных данных нет в директории `data`, то выполнить команду:
```
git lfs pull
```

Запустить `jupyter lab` или `notebook` и открыть файл: `decrypt.ipynb`.

