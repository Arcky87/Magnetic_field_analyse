import re
import pandas as pd


def parse_magnetic_catalog(filepath):
    # читаем с нужной кодировкой
    with open(filepath, 'r', encoding='koi8-r', errors='ignore') as f:
        text = f.read()

    lines = text.splitlines()

    # шаблоны
    star_pattern = re.compile(r'HD\s+\d+')

    # поле + ошибка + (опционально фаза)
    field_pattern = re.compile(
        r'([+-]?\s*\d+)\s*\+-\s*(\d+)'
        r'(?:\s*G)?'  # иногда есть G, иногда нет
        r'(?:\]?\((\d+)\))?'  # (фаза) после ]
    )

    data = {}
    current_star = None

    for line in lines:
        # ищем имя звезды
        star_match = star_pattern.search(line)
        if star_match:
            current_star = star_match.group().strip()
            if current_star not in data:
                data[current_star] = []

        if current_star is None:
            continue

        # ищем все измерения в строке
        for match in field_pattern.finditer(line):
            B = int(match.group(1).replace(' ', ''))
            sigma = int(match.group(2))

            phase_raw = match.group(3)
            phase = float(phase_raw) if phase_raw is not None else -1.0

            data[current_star].append([phase, B, sigma])

    # превращаем в DataFrame
    result = {}
    for star, values in data.items():
        if len(values) == 0:
            continue

        df = pd.DataFrame(values, columns=["phase", "B", "sigma"])
        result[star] = df

    return result

if __name__ == '__main__':

    print('This file is part of Uncertainties')

    test = parse_magnetic_catalog('APst06-12')

    keys = test.keys()

    for star in keys:
        star_data = test[star]

        print('Star:', star)
        print(star_data)