Homework1
==============================

Heart Disease Cleveland UCI classification task 

### Обучение запускается из директории src/models/ командой: 
- python train_model.py
#### Для использования модели KNN в конфиг-файле my_config#1.yaml ключ 'model' устанавливается как 'KNN', иначе 'LogisticRegression'. 
### Валидация запускается из директории src/models/ командой: 
- python predict_model.py
#### p.s. эти две команды нужно запускать последовательно, т.к. может возникнуть ошибка при использовании других данных (синтетических), ввиду того, что категориальные признаки не OneHotEncoded