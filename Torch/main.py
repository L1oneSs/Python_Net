import multiprocessing
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision as tv

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm

from torch.cuda.amp import autocast, GradScaler

f_loss = []
f_acc = []
total_time = 0
test_loss = []
test_acc = []


# Построение датасета
class Dataset2class(torch.utils.data.Dataset):
    def __init__(self, path_dir1: str, path_dir2: str):
        super().__init__()

        self.path_dir1 = path_dir1
        self.path_dir2 = path_dir2

        self.dir1_list = sorted(os.listdir(path_dir1))
        self.dir2_list = sorted(os.listdir(path_dir2))

    def __len__(self):
        return len(self.dir1_list) + len(self.dir2_list)

    def __getitem__(self, idx):

        if idx < len(self.dir1_list):
            class_id = 0
            img_path = os.path.join(self.path_dir1, self.dir1_list[idx])
        else:
            class_id = 1
            idx -= len(self.dir1_list)
            img_path = os.path.join(self.path_dir2, self.dir2_list[idx])

        # Читаем изображение из файла и преобразовываем его в формат PyTorch
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # Из типа BGR переводим в RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # приведение изображения к типу данных float32
        img = img.astype(np.float32)
        # нормализация значений пикселей изображения в диапазоне [0, 1]
        img = img / 255.0

        # Уменьшаем до 128х128 с помощью интерполяции(промежуточные значения)
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        # Изменение размерности изображения
        img = img.transpose((2, 0, 1))
        # Приведение к тензору PyTorch
        t_img = torch.from_numpy(img)
        t_class_id = torch.tensor(class_id)

        return {'img': t_img, 'label': t_class_id}

# Взятие изображений
train_dogs_path = './images/training/dogs'
train_cats_path = './images/training/cats'
test_dogs_path = './images/test/dogs'
test_cats_path = './images/test/cats'

train_ds_catsdogs = Dataset2class(train_dogs_path, train_cats_path)
test_ds_catsdogs = Dataset2class(test_dogs_path, test_cats_path)

# Кол-во примеров данных, используемых в одном пакете
batch_size = 16

# Создаем загрузчики данных
# num_workers=1 - данные загружаются в один поток
train_loader = torch.utils.data.DataLoader(
    train_ds_catsdogs, shuffle=True,
    batch_size=batch_size, num_workers=1, drop_last=True
)
test_loader = torch.utils.data.DataLoader(
    test_ds_catsdogs, shuffle=True,
    batch_size=batch_size, num_workers=1, drop_last=False
)

# Класс нейронной сети
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Выполнение нелинейного преобразования, отбрасывание мертвых нейронов
        # Функция активации
        self.act = nn.LeakyReLU(0.2)
        #  Слой максимальной подвыборки, уменьшающий размер изображения
        self.maxpool = nn.MaxPool2d(2, 2)
        # Сверточные слои
        self.conv0 = nn.Conv2d(3, 128, 3, stride=1, padding=0)
        self.conv1 = nn.Conv2d(128, 128, 3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(128, 128, 3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=0)
        # Слой адаптивной усредняющей подвыборки
        self.adaptivepool = nn.AdaptiveAvgPool2d((1, 1))
        # Слой, который преобразует тензор изображения в плоский вектор
        self.flatten = nn.Flatten()
        # Выходные тензор после применения линейного слоя имеет форму (batch_size, out_nc)
        self.linear1 = nn.Linear(256, 20)
        self.linear2 = nn.Linear(20, 2)

    # Реализуем сетку
    def forward(self, x):
        out = self.conv0(x)
        out = self.act(out)
        out = self.maxpool(out)

        out = self.conv1(out)
        out = self.act(out)
        out = self.maxpool(out)

        out = self.conv2(out)
        out = self.act(out)
        out = self.maxpool(out)

        out = self.conv3(out)
        out = self.act(out)

        out = self.adaptivepool(out)
        out = self.flatten(out)
        out = self.linear1(out)
        out = self.act(out)
        out = self.linear2(out)

        return out

def main():
    # Создаем модель
    global total_time
    model = ConvNet()

    for sample in train_loader:
        img = sample['img']
        label = sample['label']
        model(img)
        break

    # Функция потерь
    loss_fn = nn.CrossEntropyLoss()
    # Оптимизатор Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

    #  Функция вычисления точности предсказания модели на батче данных
    def accuracy(pred, label):
        answer = F.softmax(pred.detach()).numpy().argmax(1) == label.numpy().argmax(1)
        return answer.mean()

    # Перемещение модели на устройство
    device = 'cpu'
    model = model.to(device)
    loss_fn = loss_fn.to(device)

    # Обучение с автоматической смешанной точностью
    use_amp = True
    scaler = torch.cuda.amp.GradScaler()

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


    # Обучение модели
    epochs = 20

    # Подсчитываем количество изображений каждого класса в обучающей выборке
    n_dogs_train = len(os.listdir(train_dogs_path))
    n_cats_train = len(os.listdir(train_cats_path))

    # Визуализируем распределение классов
    plt.bar(['dogs', 'cats'], [n_dogs_train, n_cats_train])
    plt.title('Class distribution in the training set')
    plt.show()

    epoch_times = []  # Создаем список для записи времени выполнения на каждой эпохе
    test_loss = []  # test loss values
    test_acc = []  # test accuracy values

    for epoch in range(epochs):
        epoch_time_start = time.time()  # Записываем время начала эпохи
        loss_val = 0
        acc_val = 0
        for sample in (pbar := tqdm(train_loader)):
            # Получаем образец изображения и метку класса
            img, label = sample['img'], sample['label']
            # Переводим в one_hot
            label = F.one_hot(label, 2).float()
            # Перемещение на устройство
            img = img.to(device)
            label = label.to(device)
            # Обнуление градиентов оптимизатора
            optimizer.zero_grad()

            # Автоматическое ускорение вычислений
            with autocast(use_amp):
                pred = model(img)
                loss = loss_fn(pred, label)

            scaler.scale(loss).backward()
            loss_item = loss.item()
            loss_val += loss_item

            scaler.step(optimizer)
            scaler.update()

            # Вычисление текущей точности
            acc_current = accuracy(pred.cpu().float(), label.cpu().float())
            acc_val += acc_current

################################################################################################

        pbar.set_description(f'loss: {loss_item:.5f}\taccuracy: {acc_current:.3f}')
        print("Значение функции потерь: ", loss_val / len(train_loader))
        print("Точность: ", acc_val / len(train_loader))

        f_loss.append(loss_val / len(train_loader))
        f_acc.append(acc_val / len(train_loader))

        epoch_time_end = time.time()  # Записываем время окончания эпохи
        epoch_times.append(epoch_time_end - epoch_time_start)  # Записываем время выполнения эпохи в список
        total_time = total_time + (epoch_time_end - epoch_time_start)
        print("Время выполнения для данной эпохи:", epoch_times)
        print("Общее время выполнения:", total_time)

        # Итерируемся по первому пакету изображений из обучающей выборки
        for batch in train_loader:
            # Разбираем батч на изображения и метки классов
            imgs = batch['img']
            labels = batch['label']

            # Выводим первые 4 изображения из батча
            plt.figure(figsize=(10, 10))
            for i in range(4):
                plt.subplot(2, 2, i + 1)
                plt.imshow(imgs[i].permute(1, 2, 0))
                if labels[i] == 0:
                    plt.title('Label: DOG')
                else:
                    plt.title('Label: CAT')
            plt.show()

            break  # Прерываем цикл после первого пакета

    for i in range(1,21):
        # Вызов функции для тестирования модели
        test_loss_val, test_acc_val = test_model(model, test_loader)
        print("Тест :", i)
        # Вывод результатов тестирования
        print("Значение функции потерь для теста: ", test_loss_val)
        print("Значение функции точности для теста: ", test_acc_val)
        print("")


def test_model(model, test_loader):

    #  Функция вычисления точности предсказания модели на батче данных
    def accuracy(pred, label):
        answer = F.softmax(pred.detach()).numpy().argmax(1) == label.numpy().argmax(1)
        return answer.mean()

    # Функция потерь
    loss_fn = nn.CrossEntropyLoss()

    device = 'cpu'
    model = model.to(device)
    loss_fn = loss_fn.to(device)
    model.eval()  # Установка режима оценки модели
    test_loss_val = 0
    test_acc_val = 0

    with torch.no_grad():
        for sample in test_loader:
            img, label = sample['img'], sample['label']
            label = F.one_hot(label, 2).float()
            img = img.to(device)
            label = label.to(device)

            pred = model(img)
            loss = loss_fn(pred, label)
            acc = accuracy(pred.cpu().float(), label.cpu().float())

            test_loss_val += loss.item()
            test_acc_val += acc

    test_loss_val /= len(test_loader)
    test_acc_val /= len(test_loader)

    return test_loss_val, test_acc_val



if __name__ == '__main__':
    try:
        main()
        multiprocessing.freeze_support()
        multiprocessing.Process().start()
    except KeyboardInterrupt:
        print("Вы прервали программу")
