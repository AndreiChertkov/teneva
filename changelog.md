# changelog

> This is a draft version of the changelog (for developers now). In the future, the English version will be integrated into the github system.



## Version 0.13.0


### GLB (global changes)

- Добавлен файл `workflow.md` с описанием рекомендаций (для разработчиков) по текстовому описанию коммитов и развертыванию программного продукта

- Удалена поддержка python 3.6 и 3.7 для снятия проблем с версиями старых библиотек (примечание: Google Colab уже перешел на python 3.8)

- Удален модуль "collection", входящие в него подмодули перенесены в модуль "core". Это позволит более последовательно реализовать jax версию кода

- Удален пакет "numba" из "requirements.txt". Этот пакет не всегда корректно устанавливается и требует наличия дополнительных библиотек, при этом использовалась "numba" лишь в одной функции "core.act_one.getter", вместо которой может применяться функция "core.act_one.get_many", которая работает значительно быстрее. Добавлено также примечание о скором удалении функции "core.act_one.getter" в последующих версиях

- Удален пакет "matplotlib" из "requirements.txt" (планируется сделать его опциональным, поскольку он требуется только для бенчмарков)

- Уточнена зависимость от "scipy" (1.8+) и для "numpy" (1.22+)

- Начата разработка jax-версии кода, добавлен "jax" в "requirements.txt", существенно расширен "README.md" с комментариями по "jax" версии (поддерживаются только тензоры, имеющие постоянный размер мод и TT-ранги, при этом используется специальный формат для хранения TT-тензоров: список из трех массивов, соответствующих первому ядру, всем внутренним ядрам и последнему ядру). При работе в рамках jax-версии, удобно осуществлять импорт "import teneva.core_jax as teneva". Примечание: для возможности использования ортогонализации и svd, необходимо также, что ранг тензора не превосходил размер моды (этого всегда можно добиться, например, склеивая ядра при необходимости), в будущем это нужно будет явно обозначить как третье ограничение нового формата: (1) n = const, (2) r = const, (3) r <= n

- Модуль "func" помечен как устаревший. Планируется его заменить в ближайшем будущем на специализированные модули для бенчмарков и бейзлайнов


### RNM (renames of modules, functions, etc.)

- "core.tensors.tensor_poly" -> "core.tensors.poly"

- "core.tensors.tensor_const" -> "core.tensors.const"

- "core.tensors.tensor_delta" -> "core.tensors.delta"

- "core.tensors.tensor_rand" -> "core.tensors.rand_custom". Данная функция по умолчанию генерирует TT-тензор из стандартного нормального распределения, но может быть задана произвольная функция для семплирования в аргументе "f". Предполагается для создания случайных тензоров по умолчанию использовать в дальнейшем новую функцию "core.tensors.rand", которая семплирует из равномерного распределения, а функцию "core.tensors.rand_custom" применять только при необходимости

- "core.stat.sample_ind_rand" -> "core.sample.sample"

- "core.stat.sample_ind_rand_square" -> "core.sample.sample_square"

- "core.grid.sample_lhs" -> "core.sample.sample_lhs"

- "core.grid.sample_tt" -> "core.sample.sample_tt"


### UPG (upgrade of functions)

- "core.vis.show". Добавлена проверка корректности (размеры ядер) переданного TT-тензора


### NEW (new functions)

- "core.tensors.rand". Семплирует из равномерного распределения с пределами по умолчанию от -1 до +1 (с использованием функции "core.tensors.rand_custom")

- "core.tensors.rand_norm". Семплирует из нормального распределения с заданной дисперсией и средним (с использованием функции "core.tensors.rand_custom")

- "core_jax.act_one.copy"

- "core_jax.act_one.get"

- "core_jax.act_one.get_many"

- "core_jax.act_one.get_stab"

- "core_jax.act_one.mean"

- "core_jax.act_one.mean_stab"

- "core_jax.act_one.sum"

- "core_jax.act_one.sum_stab"

- "core_jax.maxvol.maxvol"

- "core_jax.maxvol.maxvol_rect" (draft!)

- "core_jax.svd.matrix_skeleton"

- "core_jax.svd.svd

- "core_jax.tensors.rand"

- "core_jax.tensors.rand_norm"

- "core_jax.transformation.full"

- "core_jax.transformation.orthogonalize_rtl"

- "core_jax.transformation.orthogonalize_trl_stab"

- "core_jax.vis.show". Проверка корректности переданного TT-тензора и краткая печать его формы


### DEM (demo in jupyter)

- Масштабно обновлены все демонстрационные примеры для всех функций из "core" (с учетом также переименований функций)

- Созданы демонстрационные примеры (в дальнейшем будут уточняться) для всех новых функций из "core_jax"


### FIX (small fixes)

- Многое уточнено ;)


### BUG (fixes for bugs)

- ?


### STL (style)

- Уточнены стили для многих функций из "core"

- Убраны раздражающие кавычки в документации всех функций (названия переменных теперь указываются без кавычек, однако для имен функций стоит по-прежнему использовать двойные кавычки для выделения; при этом имя модуля можно как указывать, так и не указывать)

- Более удобно отображен тип выходного значения для случая нескольких элементов (например, "Returns: \n (np.ndarray, float)): описание`


### DOC (documentation)

- Масштабно обновлена документация (более аккуратный текст, комментарий о появлении jax версии, увеличены отступы между функциями и т.д.)


### DEV (development)

- "core.core". Несколько функций в модуле не имеют документацию и демо, добавлено "TODO"

- "core.cross_act.cross_act". Функция помечена как "DRAFT" (нужно тщательно проверить и уточнить одноранговый случай)
