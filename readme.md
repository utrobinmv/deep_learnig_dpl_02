# Проект deep learning image captioning

Димломный проект для курса МФТИ по NLP

До начала работы скачайте модели и словарь по ссылке https://disk.yandex.ru/d/k4FEpYmNfC6yqA и поместите её в папку tmp/

Использовался датасет для обучения скачайте их по ссылке https://disk.yandex.ru/d/VRZJJFpWnfoypA и поместите в папку data/

Для запуска бота,

добавьте токен бота в файл var/token.py

```
pip install -r requirements.txt
```

For start telegram bot

```
python image_captionin_predict_models_bot.py
```



Дополнительно имеются ноутбуки:

1) Простая реализация модели image captioning

```
image_captioning-simple.ipynb
```

2. Усложенная модель с использованием attention

```
image_captioning-multi-attention.ipynb
```

