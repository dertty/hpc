# 2. Программы с багами
`omp_bugreduction_fixed.c` – код для скалярного произведения двух векторов.   
![img.png](img.png)
`omp_bugparfor_fixed.c` – устранены ошибки.  
![img_1.png](img_1.png)
# 3. Написать параллельную программу, использующую метод Монте-Карло для оценки числа pi.
Случайным образом кидаете точку в единичный квадрат. В этот же квадрат вписан круг. Если точка попала в круг, увеличиваете счетчик. Затем находите отношение точек, попавших в круг к общему числу точек. Зная площади квадрата и круга, находите приблизительно число pi.
`pi.c` – код поиска чичла pi.  
![img_2.png](img_2.png)