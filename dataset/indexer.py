import os


path = r'data/final/'


def rename_files(dir):
    file_names = os.listdir(dir)

    for i, name in enumerate(file_names):
        name = os.path.join(dir, name)
        target = os.path.join(dir, f'{i}.' + name.split('.')[-1])

        os.rename(name, target)


def rename_dir(dir):
    file_names = os.listdir(dir)

    for i, name in enumerate(file_names):
        if '.' in name:
            continue

        name = os.path.join(dir, name)
        target = os.path.join(dir, f'{i}')
        os.rename(name, target)


if __name__ == '__main__':
    rename_dir(path)

    # dirs = os.listdir(path)
    # for d in dirs:
    #     rename_files(os.path.join(path, d))
